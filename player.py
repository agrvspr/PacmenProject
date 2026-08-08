"""
Human-controlled player.

Coordinate convention follows controller.py: x = column, y = row, origin
top-left, so KEY_DOWN is (0, +1). A Board built for this indexes grid[y][x].

The player holds state, not rules. It does not check whether a move is legal
(Board decides) and does not know whether the game is over (Model decides).
"""

DOTS_REQUIRED = 3


class PlayerModel:
    """Position and collected dots. Stores nothing about the map: what the
    player can see is derived from position by the view."""

    def __init__(self, x, y, dots_required=DOTS_REQUIRED):
        self.x = x
        self.y = y
        self.start = (x, y)
        self.prev_pos = (x, y)
        self.dots_required = dots_required
        self.dots_collected = 0
        self.steps_taken = 0

    @property
    def pos(self):
        return (self.x, self.y)

    def move(self, dx, dy):
        """
        Apply a delta unconditionally; Controller.handle_key checks
        Board.is_walkable first. Records the previous cell so Model can
        detect the player and the villain trading places, which would let
        them pass through each other without ever sharing a cell.
        """
        self.prev_pos = self.pos
        self.x += dx
        self.y += dy
        self.steps_taken += 1

    def collect_dot(self):
        self.dots_collected += 1

    def has_all_dots(self):
        return self.dots_collected >= self.dots_required

    def reset(self, x=None, y=None):
        """
        Restore to the start of a match. Pass a position to move the start
        point; omit it to reuse the current one.
        """
        if (x is None) != (y is None):
            raise ValueError("reset() requires both x and y, or neither")
        if x is not None:
            self.start = (x, y)
        self.x, self.y = self.start
        self.prev_pos = self.start
        self.dots_collected = 0
        self.steps_taken = 0

    def __repr__(self):
        return (f"PlayerModel(x={self.x}, y={self.y}, "
                f"dots={self.dots_collected}/{self.dots_required}, "
                f"steps={self.steps_taken})")
