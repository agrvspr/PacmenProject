"""Genetic algorithm level generator."""

import json
import os
import random
from collections import deque

SIZE = 20
TILE = 3
TILES_PER_SIDE = (SIZE - 2) // TILE
TILE_GENES = TILES_PER_SIDE ** 2
CHEESE_COUNT = 4
PLACEMENT_GENES = CHEESE_COUNT + 1 + 2
GENOME_LENGTH = TILE_GENES + PLACEMENT_GENES

WALL = "#"
EMPTYSPACE = " "
GOALPIECE = "I"
ENDGOAL = "G"

NORTH, EAST, SOUTH, WEST, ROOM = 1, 2, 4, 8, 16
GENE_VALUES = 32

DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))

BEST_GENOME_PATH = os.path.join(os.path.dirname(__file__), "best_level.json")

def decode_walls(genome):
    """
    Build a wall/floor grid from tile genes, before entities are placed.
    Args: genome - list of ints, the full genome array.
    Returns: grid - 2D list of WALL/EMPTYSPACE strings.
    """
    grid = [[WALL] * SIZE for _ in range(SIZE)]

    for index in range(TILE_GENES):
        gene = genome[index] & (GENE_VALUES - 1)
        tile_y, tile_x = divmod(index, TILES_PER_SIDE)
        top = 1 + tile_y * TILE
        left = 1 + tile_x * TILE

        cy, cx = top + 1, left + 1
        grid[cy][cx] = EMPTYSPACE

        if gene & ROOM:
            for oy in range(3):
                for ox in range(3):
                    grid[top + oy][left + ox] = EMPTYSPACE

        if gene & NORTH:
            grid[top][cx] = EMPTYSPACE
        if gene & SOUTH:
            grid[top + 2][cx] = EMPTYSPACE
        if gene & WEST:
            grid[cy][left] = EMPTYSPACE
        if gene & EAST:
            grid[cy][left + 2] = EMPTYSPACE

    return grid


def open_cells(grid):
    return [(x, y)
            for y in range(SIZE)
            for x in range(SIZE)
            if grid[y][x] != WALL]


def largest_component(grid):
    """
    Find the largest connected region of floor cells via BFS.
    Args: grid - 2D list of WALL/EMPTYSPACE strings.
    Returns: sorted list of (x, y) cells in the largest component.
    """
    seen = set()
    best = []
    for start in open_cells(grid):
        if start in seen:
            continue
        component = []
        queue = deque([start])
        seen.add(start)
        while queue:
            x, y = queue.popleft()
            component.append((x, y))
            for dx, dy in DIRS:
                nxt = (x + dx, y + dy)
                if nxt not in seen and _in_bounds(*nxt) and grid[nxt[1]][nxt[0]] != WALL:
                    seen.add(nxt)
                    queue.append(nxt)
        if len(component) > len(best):
            best = component
    return sorted(best)


def _in_bounds(x, y):
    return 0 <= x < SIZE and 0 <= y < SIZE


def decode(genome):
    """
    Decode a genome into walls, then place entities inside the largest region.
    Args: genome - list of ints, the full genome array.
    Returns: (grid, cheese, exit_cell, player_spawn, villain_spawn) or None.
    """
    grid = decode_walls(genome)
    region = largest_component(grid)
    if len(region) < PLACEMENT_GENES + 4:
        return None

    in_region = set(region)
    for x, y in open_cells(grid):
        if (x, y) not in in_region:
            grid[y][x] = WALL

    chosen = []
    for offset in range(PLACEMENT_GENES):
        gene = genome[TILE_GENES + offset]
        index = gene % len(region)
        for step in range(len(region)):
            candidate = region[(index + step) % len(region)]
            if candidate not in chosen:
                chosen.append(candidate)
                break

    if len(chosen) < PLACEMENT_GENES:
        return None

    cheese = chosen[:CHEESE_COUNT]
    exit_cell = chosen[CHEESE_COUNT]
    player_spawn = chosen[CHEESE_COUNT + 1]
    villain_spawn = chosen[CHEESE_COUNT + 2]

    for x, y in cheese:
        grid[y][x] = GOALPIECE
    grid[exit_cell[1]][exit_cell[0]] = ENDGOAL

    return grid, cheese, exit_cell, player_spawn, villain_spawn

def _neighbours(grid, x, y):
    return [(x + dx, y + dy) for dx, dy in DIRS
            if _in_bounds(x + dx, y + dy) and grid[y + dy][x + dx] != WALL]


def _distances(grid, start, blocked=()):
    """
    Compute BFS step counts from start to every reachable floor cell.
    Args: grid, start (x,y), blocked - optional cells to treat as walls.
    Returns: dict mapping (x, y) -> distance from start.
    """
    blocked = set(blocked)
    dist = {start: 0}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for nxt in _neighbours(grid, x, y):
            if nxt not in dist and nxt not in blocked:
                dist[nxt] = dist[(x, y)] + 1
                queue.append(nxt)
    return dist


def articulation_points(grid, region, root):
    """
    Find cut vertices (single-doorway cells) via iterative Tarjan's algorithm.
    Args: grid, region - set of reachable cells, root - BFS start cell.
    Returns: set of (x, y) cut-vertex cells.
    """
    disc, low, parent = {}, {}, {}
    cuts = set()
    timer = 0
    root_children = 0

    disc[root] = low[root] = timer
    timer += 1
    stack = [(root, iter(_neighbours(grid, *root)))]

    while stack:
        node, neighbours = stack[-1]
        descended = False

        for nxt in neighbours:
            if nxt not in region:
                continue
            if nxt not in disc:
                parent[nxt] = node
                disc[nxt] = low[nxt] = timer
                timer += 1
                if node == root:
                    root_children += 1
                stack.append((nxt, iter(_neighbours(grid, *nxt))))
                descended = True
                break
            if nxt != parent.get(node):
                low[node] = min(low[node], disc[nxt])

        if not descended:
            stack.pop()
            if stack:
                above = stack[-1][0]
                low[above] = min(low[above], low[node])
                if above != root and low[node] >= disc[above]:
                    cuts.add(above)

    if root_children > 1:
        cuts.add(root)
    return cuts


def entrances(grid, source, target):
    """
    Count independent neighbours of target reachable from source.
    Args: grid, source (x,y), target (x,y).
    Returns: int count of independent approaches to target.
    """
    if source == target:
        return len(_neighbours(grid, *target))
    reachable = _distances(grid, source, blocked={target})
    return sum(1 for n in _neighbours(grid, *target) if n in reachable)


def _longest_straight(grid):
    """
    Find the longest unbroken run of floor in any row or column.
    Args: grid - 2D list of WALL/EMPTYSPACE strings.
    Returns: int length of the longest straight run.
    """
    longest = 0
    for y in range(SIZE):
        run = 0
        for x in range(SIZE):
            run = run + 1 if grid[y][x] != WALL else 0
            longest = max(longest, run)
    for x in range(SIZE):
        run = 0
        for y in range(SIZE):
            run = run + 1 if grid[y][x] != WALL else 0
            longest = max(longest, run)
    return longest


def _wall_clumping(grid):
    """
    Compute the fraction of wall cells that touch another wall cell.
    Args: grid - 2D list of WALL/EMPTYSPACE strings.
    Returns: float ratio between 0.0 and 1.0.
    """
    walls = [(x, y) for y in range(1, SIZE - 1) for x in range(1, SIZE - 1)
             if grid[y][x] == WALL]
    if not walls:
        return 0.0
    touching = sum(
        1 for x, y in walls
        if any(_in_bounds(x + dx, y + dy) and grid[y + dy][x + dx] == WALL
               for dx, dy in DIRS)
    )
    return touching / len(walls)


def measure(genome):
    """
    Compute every quantity the fitness function scores, in one pass.
    Args: genome - list of ints, the full genome array.
    Returns: dict of stats, or None if undecodable/unsolvable.
    """
    decoded = decode(genome)
    if decoded is None:
        return None

    grid, cheese, exit_cell, player_spawn, villain_spawn = decoded

    reach = _distances(grid, player_spawn)
    targets = cheese + [exit_cell, villain_spawn]
    if any(cell not in reach for cell in targets):
        return None

    interior = (SIZE - 2) ** 2

    region = set(reach)
    degrees = {cell: sum(1 for n in _neighbours(grid, *cell) if n in region)
               for cell in region}
    dead_ends = sum(1 for d in degrees.values() if d == 1)
    junctions = sum(1 for d in degrees.values() if d >= 3)

    edges = sum(degrees.values()) // 2
    cycles = edges - len(region) + 1
    floor = sorted(region)

    cuts = articulation_points(grid, region, player_spawn)

    if len(cuts) > MAX_CHOKEPOINTS:
        return None

    important = cheese + [exit_cell]
    for cut in cuts:
        if cut == player_spawn:
            continue
        without = _distances(grid, player_spawn, blocked={cut})
        if any(target != cut and target not in without for target in important):
            return None

    tour = 0
    current = player_spawn
    remaining = list(cheese)
    legs = []
    ok = True
    while remaining:
        dist = _distances(grid, current)
        reachable = [c for c in remaining if c in dist]
        if not reachable:
            ok = False
            break
        nearest = min(reachable, key=lambda c: dist[c])
        legs.append(dist[nearest])
        tour += dist[nearest]
        current = nearest
        remaining.remove(nearest)
    if ok:
        dist = _distances(grid, current)
        if exit_cell not in dist:
            ok = False
        else:
            legs.append(dist[exit_cell])
            tour += dist[exit_cell]
    if not ok:
        return None

    spread = min(
        (_distances(grid, a).get(b, 0)
         for i, a in enumerate(cheese) for b in cheese[i + 1:]),
        default=0,
    )

    return {
        "grid": grid,
        "cheese": cheese,
        "exit": exit_cell,
        "player_spawn": player_spawn,
        "villain_spawn": villain_spawn,
        "floor_ratio": len(floor) / interior,
        "coverage": len({((y - 1) // TILE, (x - 1) // TILE)
                         for x, y in region if 1 <= x < SIZE - 1 and 1 <= y < SIZE - 1}),
        "dead_ends": dead_ends,
        "junctions": junctions,
        "cycles": cycles,
        "tour": tour,
        "leg_spread": max(legs) - min(legs) if legs else 0,
        "cheese_spread": spread,
        "cheese_entrances": [entrances(grid, player_spawn, c) for c in cheese],
        "exit_entrances": entrances(grid, player_spawn, exit_cell),
        "pockets": 0,
        "chokepoints": len(cuts),
        "guarded_targets": sum(1 for c in cheese + [exit_cell] if c in cuts),
        "spawn_gap": reach.get(villain_spawn, 0),
        "longest_straight": _longest_straight(grid),
        "clumping": _wall_clumping(grid),
    }

UNSOLVABLE = -10_000.0

MAX_CHOKEPOINTS = 14

TARGETS = {
    "coverage":         (36,   12.0),
    "chokepoints":      (0,     3.0),
    "floor_ratio":      (0.45, 60.0),
    "tour":             (85,    0.6),
    "cycles":           (30,    0.8),
    "dead_ends":        (2,     1.5),
    "junctions":        (90,    0.8),
    "cheese_spread":    (16,    2.0),
    "spawn_gap":        (16,    2.5),
    "leg_spread":       (8,     0.8),
    "longest_straight": (10,    2.0),
    "clumping":         (1.00, 20.0),
}


def fitness(genome, detail=False):
    stats = measure(genome)
    if stats is None:
        return (UNSOLVABLE, None) if detail else UNSOLVABLE

    score = 0.0
    breakdown = {}
    for key, (target, weight) in TARGETS.items():
        penalty = -abs(stats[key] - target) * weight
        breakdown[key] = penalty
        score += penalty

    single_entrance = sum(1 for count in stats["cheese_entrances"] if count < 2)
    if single_entrance or stats["exit_entrances"] < 2:
        return (UNSOLVABLE, None) if detail else UNSOLVABLE

    if stats["guarded_targets"]:
        return (UNSOLVABLE, None) if detail else UNSOLVABLE

    extra = sum(stats["cheese_entrances"]) - 2 * CHEESE_COUNT
    extra += stats["exit_entrances"] - 2
    breakdown["extra_entrances"] = min(extra, 6) * 4.0
    score += breakdown["extra_entrances"]

    return (score, {"stats": stats, "breakdown": breakdown}) if detail else score

def random_genome(rng):
    return [rng.randrange(GENE_VALUES) for _ in range(TILE_GENES)] + \
           [rng.randrange(1024) for _ in range(PLACEMENT_GENES)]


def viable_genome(rng, attempts=60):
    """
    Generate a random genome that passes the hard fitness gates.
    Args: rng - random.Random instance, attempts - max tries.
    Returns: a genome (list of ints), best-effort if none pass.
    """
    best, best_score = None, float("-inf")
    for _ in range(attempts):
        candidate = random_genome(rng)
        score = fitness(candidate)
        if score > UNSOLVABLE:
            return candidate
        if score > best_score:
            best, best_score = candidate, score
    return best


def tournament(population, scores, rng, size=3):
    contenders = rng.sample(range(len(population)), size)
    winner = max(contenders, key=lambda i: scores[i])
    return population[winner]


def crossover(a, b, rng):
    cut = rng.randrange(1, GENOME_LENGTH)
    return a[:cut] + b[cut:]


def mutate(genome, rng, rate=0.06):
    child = list(genome)
    for i in range(GENOME_LENGTH):
        if rng.random() < rate:
            if i < TILE_GENES:
                child[i] ^= 1 << rng.randrange(5)
            else:
                child[i] = rng.randrange(1024)
    return child


def evolve(seed=None, population_size=40, generations=40, elite=2, verbose=False):
    """
    Run the genetic algorithm to breed a level.
    Args: seed, population_size, generations, elite, verbose.
    Returns: (best_genome, history) where history is best fitness per generation.
    """
    rng = random.Random(seed)
    population = [viable_genome(rng) for _ in range(population_size)]
    cache = {}
    history = []
    best_genome, best_score = None, float("-inf")

    def score_of(genome):
        key = tuple(genome)
        if key not in cache:
            cache[key] = fitness(genome)
        return cache[key]

    for generation in range(generations):
        scores = [score_of(g) for g in population]

        ranked = sorted(range(population_size), key=lambda i: scores[i], reverse=True)
        if scores[ranked[0]] > best_score:
            best_score = scores[ranked[0]]
            best_genome = list(population[ranked[0]])
        history.append(best_score)

        if verbose:
            solvable = sum(1 for s in scores if s > UNSOLVABLE)
            print(f"  gen {generation + 1:3d}  best {best_score:9.1f}  "
                  f"solvable {solvable}/{population_size}")

        nxt = [list(population[i]) for i in ranked[:elite]]
        while len(nxt) < population_size:
            parent_a = tournament(population, scores, rng)
            parent_b = tournament(population, scores, rng)
            nxt.append(mutate(crossover(parent_a, parent_b, rng), rng))
        population = nxt

    return best_genome, history


# --- the drop-in board -------------------------------------------------------

class GeneticBoard:
    """A level bred by the GA, wearing DemoBoard's interface."""

    SIZE = SIZE
    GOALPIECE = GOALPIECE
    ENDGOAL = ENDGOAL
    WALL = WALL
    EMPTYSPACE = EMPTYSPACE

    def __init__(self, seed=None, genome=None, population_size=40, generations=40,
                 verbose=False):
        if genome is None:
            genome, self.history = evolve(seed=seed,
                                          population_size=population_size,
                                          generations=generations,
                                          verbose=verbose)
        else:
            self.history = []

        self.genome = genome
        self.fitness = fitness(genome)

        decoded = decode(genome)
        if decoded is None:
            raise ValueError("genome does not decode to a usable level")

        (self.grid, self.cheese, self.exit_cell,
         self.player_spawn, self.villain_spawn) = decoded

    def is_walkable(self, x, y):
        return (0 <= x < self.SIZE
                and 0 <= y < self.SIZE
                and self.grid[y][x] != self.WALL)

    def remaining_goal_items(self):
        return sum(row.count(self.GOALPIECE) for row in self.grid)

    def get_spawn_positions(self):
        """
        Return the player and villain spawn cells.
        Args: none.
        Returns: (player_spawn, villain_spawn) tuple.
        """
        return self.player_spawn, self.villain_spawn

    @classmethod
    def from_saved(cls, path=BEST_GENOME_PATH):
        with open(path) as handle:
            return cls(genome=json.load(handle)["genome"])

    def save(self, path=BEST_GENOME_PATH):
        with open(path, "w") as handle:
            json.dump({"genome": self.genome, "fitness": self.fitness}, handle)

    def render(self):
        rows = []
        for y in range(self.SIZE):
            row = ""
            for x in range(self.SIZE):
                if (x, y) == self.player_spawn:
                    row += "P"
                elif (x, y) == self.villain_spawn:
                    row += "V"
                else:
                    row += self.grid[y][x]
            rows.append(row)
        return "\n".join(rows)


if __name__ == "__main__":
    import sys

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    print(f"evolving a level (seed {seed})")
    board = GeneticBoard(seed=seed, verbose=True)
    print()
    print(board.render())
    print()
    score, detail = fitness(board.genome, detail=True)
    print(f"fitness {score:.1f}")
    for key, (target, _) in TARGETS.items():
        print(f"  {key:18s} {detail['stats'][key]:>8.2f}   target {target}")
    board.save()
    print(f"\nsaved {BEST_GENOME_PATH}")
