
import random
from player import PlayerModel

class DemoBoard:
    """Simple board generator that randomizes tiles inside a bordered grid."""

    SIZE = 9
    GOALPIECE = "I"
    ENDGOAL = "G"
    WALL = "#"
    EMPTYSPACE = "x"
    DEFAULT_TILES = {GOALPIECE: 3, ENDGOAL: 1}
    DEFAULT_MIN_SPAWN_DISTANCE = 4
    DEFAULT_OBSTACLE_RATIO = 0.14

    def __init__(
        self,
        tiles=None,
        size=None,
        min_spawn_distance=DEFAULT_MIN_SPAWN_DISTANCE,
        obstacle_ratio=DEFAULT_OBSTACLE_RATIO,
    ):
        self.SIZE = size or self.SIZE
        self.grid = self.generate_random_board(
            tiles or self.DEFAULT_TILES,
            self.SIZE,
            obstacle_ratio=obstacle_ratio,
        )
        self.player_spawn, self.villain_spawn = self._pick_spawn_positions(min_spawn_distance)

    @classmethod
    def generate_random_board(cls, tiles, size=None, obstacle_ratio=None):
        """Return a size x size grid with the given tile counts placed randomly."""
        size = size or cls.SIZE
        obstacle_ratio = cls.DEFAULT_OBSTACLE_RATIO if obstacle_ratio is None else obstacle_ratio
        if size < 3:
            raise ValueError("Board size must be at least 3")

        grid = [[cls.EMPTYSPACE for _ in range(size)] for _ in range(size)]

        for index in range(size):
            grid[0][index] = cls.WALL
            grid[size - 1][index] = cls.WALL
            grid[index][0] = cls.WALL
            grid[index][size - 1] = cls.WALL

        open_cells = [
            (x, y)
            for y in range(1, size - 1)
            for x in range(1, size - 1)
        ]
        random.shuffle(open_cells)

        total_tiles = sum(tiles.values())
        interior_cells = len(open_cells)
        max_obstacles = max(0, interior_cells - total_tiles - 2)
        desired_obstacles = int(interior_cells * obstacle_ratio)
        obstacle_count = max(0, min(desired_obstacles, max_obstacles))

        for _ in range(obstacle_count):
            x, y = open_cells.pop()
            grid[y][x] = cls.WALL

        open_cells = [
            (x, y)
            for (x, y) in open_cells
            if grid[y][x] == cls.EMPTYSPACE
        ]
        random.shuffle(open_cells)

        if total_tiles > len(open_cells):
            raise ValueError("Not enough open cells for all tiles")

        for tile, count in tiles.items():
            for _ in range(count):
                x, y = open_cells.pop()
                grid[y][x] = tile

        return grid

    def _pick_spawn_positions(self, min_distance):
        """Pick player/villain spawns on empty cells with at least min Manhattan distance."""
        empty_cells = [
            (x, y)
            for y in range(1, self.SIZE - 1)
            for x in range(1, self.SIZE - 1)
            if self.grid[y][x] == self.EMPTYSPACE
        ]
        random.shuffle(empty_cells)

        if len(empty_cells) < 2:
            raise ValueError("Not enough empty cells to place player and villain")

        required_distance = max(1, min_distance)
        while required_distance >= 1:
            for i, player_cell in enumerate(empty_cells):
                for villain_cell in empty_cells[i + 1:]:
                    if self._manhattan(player_cell, villain_cell) >= required_distance:
                        return player_cell, villain_cell
            required_distance -= 1

        return empty_cells[0], empty_cells[1]

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_spawn_positions(self):
        return self.player_spawn, self.villain_spawn

    def is_walkable(self, x, y):
        return (
            0 <= x < self.SIZE
            and 0 <= y < self.SIZE
            and self.grid[y][x] != self.WALL
        )

    def remaining_goal_items(self):
        """
        Count GOALPIECE tiles still on the grid.
        """
        return sum(row.count(self.GOALPIECE) for row in self.grid)
class BoardState:
    GoalPiecesCollected = sum
    GameStarted = False
    FinalGoalActive = False

