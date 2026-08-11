#Human controller player


class PlayerModel:
    #player location and control

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.start = (x, y)
        self.prev_pos = (x, y)
        self.steps_taken = 0

    @property
    def pos(self):
        return (self.x, self.y)

    def move(self, dx, dy):
        #applies movement delta
        self.prev_pos = self.pos
        self.x += dx
        self.y += dy
        self.steps_taken += 1

    def reset(self, x=None, y=None):
        #restarts game
        if (x is None) != (y is None):
            raise ValueError("reset() requires both x and y, or neither")
        if x is not None:
            self.start = (x, y)
        self.x, self.y = self.start
        self.prev_pos = self.start
        self.steps_taken = 0

    def __repr__(self):
        return (f"PlayerModel(x={self.x}, y={self.y}, "
                f"steps={self.steps_taken})")
