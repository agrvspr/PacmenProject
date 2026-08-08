import curses

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
        Given a single curses key code, attempt to move the player.
        Returns True if the player actually moved, False otherwise
        (invalid key, or move blocked by a wall/boundary).
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
        Main input loop. render_callback(stdscr) is called once before each
        key read, so it should draw the current board/player state.
        Press q to quit.
        """
        curses.curs_set(0)
        stdscr.nodelay(False)
 
        while True:
            render_callback(stdscr)
            key = stdscr.getch()
 
            if key in QUIT_KEYS:
                break
 
            self.handle_key(key)