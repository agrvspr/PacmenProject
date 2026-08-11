import curses
from model.boardGeneration import DemoBoard, BoardState


from model.player import PlayerModel

DIRECTIONS = {
    curses.KEY_UP: (0, -1),
    curses.KEY_DOWN: (0, 1),
    curses.KEY_LEFT: (-1, 0),
    curses.KEY_RIGHT: (1, 0),
    ord("w"): (0, -1),
    ord("W"): (0, -1),
    ord("s"): (0, 1),
    ord("S"): (0, 1),
    ord("a"): (-1, 0),
    ord("A"): (-1, 0),
    ord("d"): (1, 0),
    ord("D"): (1, 0),
}

QUIT_KEYS = {ord("q"), ord("Q")}


class Controller:
    def __init__(self, board, player):
        self.board = board
        self.player = player

    def handle_key(self, key):
        """
        Move the player if key is a direction and the target cell is open.
        Args: key - a curses key code.
        Returns: bool True if the player actually moved.
        """
        if key not in DIRECTIONS:
            return False

        dx, dy = DIRECTIONS[key]
        new_x = self.player.x + dx
        new_y = self.player.y + dy

        if not self.board.is_walkable(new_x, new_y):
            return False


        self.player.move(dx, dy)
        return True

    def run(self,stdscr, render_callback):
        """
        Run the main render/input loop until a quit key is pressed.
        Args: stdscr - curses screen, render_callback - draw function.
        Returns: None.
        """
        curses.curs_set(0)
        stdscr.nodelay(False)
 
        while True:
            render_callback(stdscr)
            key = stdscr.getch()
 
            if key in QUIT_KEYS:
                break
 
            self.handle_key(key)
if __name__ == "__main__":
 
    class DemoPlayer:
        def __init__(self, x, y):
            self.x = x
            self.y = y
 
        def move(self, dx, dy):
            self.x += dx
            self.y += dy
 
    def render(stdscr):
        """
        Draw the board and player onto the curses screen.
        Args: stdscr - curses screen to draw on.
        Returns: None.
        """
        stdscr.clear()
        stdscr.addstr(0, 0, "Arrow keys / WASD to move, q to quit")
        for y in range(board.SIZE):
            row = ""
            for x in range(board.SIZE):
                if (x, y) == (player.x, player.y):
                    row += "P"
                else:
                    row += board.grid[y][x]
            stdscr.addstr(y + 2, 0, row)
        stdscr.refresh()
 
    board = DemoBoard()
    BoardState.GameStarted = True
    player = PlayerModel(4, 4)
    controller = Controller(board, player)
 
    curses.wrapper(lambda stdscr: controller.run(stdscr, render))