
import random


class DemoBoard:
    """Simple board generator that randomizes tiles inside a bordered grid."""

    SIZE = 12
    GOALPIECE = "I"
    ENDGOAL = "G"
    WALL = "#"
    EMPTYSPACE = "X"

    def __init__(self, tiles=None, size=None):
        self.SIZE = size or self.SIZE
        self.grid = self.generate_random_board(tiles or {}, self.SIZE)

    @classmethod
    def generate_random_board(cls, tiles, size=None):
        """Return a size x size grid with the given tile counts placed randomly."""
        size = size or cls.SIZE
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
        if total_tiles > len(open_cells):
            raise ValueError("Not enough open cells for all tiles")

        for tile, count in tiles.items():
            for _ in range(count):
                x, y = open_cells.pop()
                grid[y][x] = tile

        return grid

    def is_walkable(self, x, y):
        return (
            0 <= x < self.SIZE
            and 0 <= y < self.SIZE
            and self.grid[y][x] != self.WALL
        )