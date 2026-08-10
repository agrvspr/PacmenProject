"""
Genetic algorithm level generator.

Drop-in replacement for boardGeneration.DemoBoard: GeneticBoard exposes the
same SIZE / grid / tile constants / is_walkable / remaining_goal_items /
get_spawn_positions, so the game, the villain and both views need no changes.

    from board_ga import GeneticBoard
    board = GeneticBoard(seed=123)

Why a GA and not random-generate-and-reject: rejection sampling can find
*valid* levels, but it cannot aim at a *difficulty*. The fitness function
below scores a level on a dozen shaped criteria -- chase-ability, pacing,
escape routes, dead ends -- and selection climbs toward levels that satisfy
them together, which is a trade-off no rejection filter expresses.

GENOME (42 integers)
    genes 0..35   one per 3x3 tile of the 6x6 interior grid, value 0..15 read
                  as 4 bits: which of N/E/S/W that tile opens. Facing open
                  arms form a corridor; an arm facing a closed one is a
                  dead-end stub. Structure emerges from 36 small numbers, and
                  crossing two levels yields a level rather than noise.
    genes 36..41  placements as indices into the sorted open cells: 3 cheese,
                  1 exit, then the player and villain spawns.
"""

import json
import os
import random
from collections import deque

# --- board shape -------------------------------------------------------------

SIZE = 20                 # matches DemoBoard, border walls included
TILE = 3                  # each gene decodes to a TILE x TILE block
TILES_PER_SIDE = (SIZE - 2) // TILE      # 6
TILE_GENES = TILES_PER_SIDE ** 2         # 36
CHEESE_COUNT = 3
PLACEMENT_GENES = CHEESE_COUNT + 1 + 2   # cheese, exit, two spawns
GENOME_LENGTH = TILE_GENES + PLACEMENT_GENES

WALL = "#"
EMPTYSPACE = " "
GOALPIECE = "I"
ENDGOAL = "G"

NORTH, EAST, SOUTH, WEST, ROOM = 1, 2, 4, 8, 16
GENE_VALUES = 32          # five bits per tile

DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))

BEST_GENOME_PATH = os.path.join(os.path.dirname(__file__), "best_level.json")


# --- decoding ----------------------------------------------------------------

def decode_walls(genome):
    """Turn the tile genes into a grid of walls and floor, no entities yet."""
    grid = [[WALL] * SIZE for _ in range(SIZE)]

    for index in range(TILE_GENES):
        gene = genome[index] & (GENE_VALUES - 1)
        tile_y, tile_x = divmod(index, TILES_PER_SIDE)
        top = 1 + tile_y * TILE
        left = 1 + tile_x * TILE

        # The centre of every tile is always floor, so no tile is ever a solid
        # block that strands its neighbours' arms.
        cy, cx = top + 1, left + 1
        grid[cy][cx] = EMPTYSPACE

        # The room bit opens the whole 3x3. Without it a tile can free at most
        # 5 of its 9 cells, which caps the walkable share near 0.55 and starves
        # the level of the corner connections that make loops possible.
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
    The biggest connected region of floor.

    Entities are placed inside it rather than anywhere on the grid, so a level
    is solvable by construction instead of by luck -- the current random
    generator leaves roughly 0.6% of levels with a walled-off cheese or exit.
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
    Full decode: walls, then entities placed inside the largest component.

    Returns (grid, cheese, exit_cell, player_spawn, villain_spawn), or None if
    the layout has too little connected floor to place everything.
    """
    grid = decode_walls(genome)
    region = largest_component(grid)
    if len(region) < PLACEMENT_GENES + 4:
        return None

    chosen = []
    for offset in range(PLACEMENT_GENES):
        gene = genome[TILE_GENES + offset]
        index = gene % len(region)
        # Walk forward to the next free cell so two genes never collide.
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


# --- measurements the fitness function reads ---------------------------------

def _neighbours(grid, x, y):
    return [(x + dx, y + dy) for dx, dy in DIRS
            if _in_bounds(x + dx, y + dy) and grid[y + dy][x + dx] != WALL]


def _distances(grid, start):
    """BFS step counts from start to every reachable floor cell."""
    dist = {start: 0}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for nxt in _neighbours(grid, x, y):
            if nxt not in dist:
                dist[nxt] = dist[(x, y)] + 1
                queue.append(nxt)
    return dist


def _longest_straight(grid):
    """Longest unbroken run of floor in any row or column."""
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
    Fraction of wall cells that touch another wall cell.

    Scattered single blocks read as noise; walls in lines read as architecture.
    Purely cosmetic, weighted low, but it is what stops evolved levels from
    looking like static.
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
    Every quantity the fitness function scores, in one pass so the expensive
    BFS work is not repeated per criterion.

    Returns None when the level cannot be decoded or is not solvable, which
    the caller turns into a hard rejection.
    """
    decoded = decode(genome)
    if decoded is None:
        return None

    grid, cheese, exit_cell, player_spawn, villain_spawn = decoded

    reach = _distances(grid, player_spawn)
    targets = cheese + [exit_cell, villain_spawn]
    if any(cell not in reach for cell in targets):
        return None                      # hard gate: unwinnable level

    interior = (SIZE - 2) ** 2

    # Measure the region the player is actually confined to, not every scrap of
    # floor on the grid. Counting isolated pockets as well made `cycles` come
    # out negative, since edges - vertices + 1 only holds for one component.
    region = set(reach)
    degrees = {cell: sum(1 for n in _neighbours(grid, *cell) if n in region)
               for cell in region}
    dead_ends = sum(1 for d in degrees.values() if d == 1)
    junctions = sum(1 for d in degrees.values() if d >= 3)

    # Independent cycles in the reachable graph: edges - vertices + 1.
    # This is the most important structural number for a chase game -- a level
    # with no cycles is a tree, and on a tree a villain that gets between the
    # player and their goal can never be gone around.
    edges = sum(degrees.values()) // 2
    cycles = edges - len(region) + 1
    floor = sorted(region)

    # Greedy nearest-cheese tour, then the exit: an approximation of the route
    # a player actually walks, which is what sets the length of a match.
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
        "dead_ends": dead_ends,
        "junctions": junctions,
        "cycles": cycles,
        "tour": tour,
        "leg_spread": max(legs) - min(legs) if legs else 0,
        "cheese_spread": spread,
        "cheese_exposed": sum(1 for c in cheese if degrees.get(c, 0) >= 2),
        "exit_approaches": degrees.get(exit_cell, 0),
        "spawn_gap": reach.get(villain_spawn, 0),
        "longest_straight": _longest_straight(grid),
        "clumping": _wall_clumping(grid),
    }


# --- fitness -----------------------------------------------------------------
# Every term is a penalty for missing a target, so fitness is always <= 0 and
# a perfect level scores 0. Weights say how much each criterion matters
# relative to the others; they are the main thing to tune by eye.

UNSOLVABLE = -10_000.0

# Targets are calibrated against the range 400 random genomes actually produce,
# so none of them is unreachable -- an unreachable target is a constant penalty
# that only adds noise to selection. Ranges seen: floor 0.14-0.85, tour 10-125,
# cycles 14-185, dead ends 0-21, junctions 22-229, cheese spread 1-28,
# spawn gap 1-36, leg spread 2-36, longest straight 9-18, clumping 0.85-1.0.
TARGETS = {
    "floor_ratio":      (0.45, 60.0),   # a bit tighter than a random level
    "tour":             (85,    0.6),   # long enough that a match has shape
    "cycles":           (30,    0.8),   # loops to evade through, not a plaza
    "dead_ends":        (10,    1.5),   # some, for fog tension
    "junctions":        (90,    0.8),   # decision points
    "cheese_spread":    (16,    2.0),   # closest pair, so you cross the map
    "spawn_gap":        (16,    2.5),   # villain's head start
    "leg_spread":       (8,     0.8),   # even pacing between cheese
    "longest_straight": (10,    2.0),   # no long lethal shooting galleries
    "clumping":         (1.00, 20.0),   # walls should look built, not sprinkled
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

    # A cheese in a dead end is a death trap once the villain is fast, and the
    # exit needs more than one approach or the villain can camp a single
    # chokepoint and the level is unwinnable in practice rather than in theory.
    trapped = CHEESE_COUNT - stats["cheese_exposed"]
    breakdown["trapped_cheese"] = -80.0 * trapped
    score += breakdown["trapped_cheese"]

    if stats["exit_approaches"] < 2:
        breakdown["camped_exit"] = -120.0
        score += breakdown["camped_exit"]

    return (score, {"stats": stats, "breakdown": breakdown}) if detail else score


# --- the genetic algorithm ---------------------------------------------------

def random_genome(rng):
    return [rng.randrange(GENE_VALUES) for _ in range(TILE_GENES)] + \
           [rng.randrange(1024) for _ in range(PLACEMENT_GENES)]


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
                # Flip one arm rather than randomising the tile: small steps
                # let selection refine a layout instead of restarting it.
                child[i] ^= 1 << rng.randrange(5)
            else:
                child[i] = rng.randrange(1024)
    return child


def evolve(seed=None, population_size=40, generations=40, elite=2, verbose=False):
    """
    Run the GA and return (best_genome, history).

    history is the best fitness per generation, which is the curve worth
    plotting for a writeup: it is the evidence that selection is doing
    something a random search would not.
    """
    rng = random.Random(seed)
    population = [random_genome(rng) for _ in range(population_size)]
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
    """
    A level bred by the GA, wearing DemoBoard's interface.

    Evolving takes a moment, so a level can be built three ways: evolve now
    (default), reuse a genome you already have, or load the saved champion
    from best_level.json for an instant, reproducible start.
    """

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

    # --- the interface the rest of the game already expects ---

    def is_walkable(self, x, y):
        return (0 <= x < self.SIZE
                and 0 <= y < self.SIZE
                and self.grid[y][x] != self.WALL)

    def remaining_goal_items(self):
        return sum(row.count(self.GOALPIECE) for row in self.grid)

    def get_spawn_positions(self):
        """(player, villain), both inside the connected region by construction."""
        return self.player_spawn, self.villain_spawn

    # --- helpers ---

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
