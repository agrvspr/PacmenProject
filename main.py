"""
Main entry point that wires the Model (board, player, villain) together with
the movement Controller and a simple curses-based View.

Rules implemented here (not owned by any single existing file, so they live
in a small GameModel class):
  - Player walks a walled grid collecting GOALPIECE ("I") tiles.
  - Villain hunts the player with its Scout/Hunter split-brain AI, and
    closes in faster as goal items are collected (see Villain.move_period).
  - Player wins by collecting every goal item and then stepping onto
    ENDGOAL ("G").
  - Player loses the moment it shares a cell with the villain.

Controller.handle_key only knows how to move the player against the board;
it has no notion of items, the villain, or win/lose. GameModel is the glue
that runs after every successful player move.
"""
import curses
import random
import time

from boardGeneration import DemoBoard
from player import PlayerModel
from villain import Villain
from controller import Controller, DIRECTIONS, QUIT_KEYS


TOTAL_GOAL_ITEMS = 3

# How long the input loop waits for a keypress before redrawing. Short enough
# that the villain's movement looks smooth, long enough not to spin the CPU.
FRAME_MS = 30


def _open_cells(board):
    """All EMPTYSPACE cells -- i.e. not a wall, not a goal item, not the exit."""
    return [
        (x, y)
        for y in range(board.SIZE)
        for x in range(board.SIZE)
        if board.grid[y][x] == DemoBoard.EMPTYSPACE
    ]


def _pick_start_positions(board):
    """Pick two distinct random empty cells: one for the player, one for the villain."""
    cells = _open_cells(board)
    random.shuffle(cells)
    if len(cells) < 2:
        raise ValueError("Board doesn't have enough open space to place player and villain")
    player_cell = cells.pop()
    villain_cell = cells.pop()
    return player_cell, villain_cell


class GameModel:
    """
    Owns the board/player/villain trio and the rules connecting them:
    picking up goal items, running the villain's turn, and deciding when the
    game has been won or lost.
    """

    def __init__(self, board, player, villain, total_goal_items=TOTAL_GOAL_ITEMS):
        self.board = board
        self.player = player
        self.villain = villain
        self.total_goal_items = total_goal_items
        self.game_over = False
        self.won = False

    def _collect_item_if_present(self):
        x, y = self.player.x, self.player.y
        if self.board.grid[y][x] == DemoBoard.GOALPIECE:
            self.board.grid[y][x] = DemoBoard.EMPTYSPACE

    def _check_win(self):
        on_exit = self.board.grid[self.player.y][self.player.x] == DemoBoard.ENDGOAL
        if on_exit and self.board.remaining_goal_items() == 0:
            self.game_over = True
            self.won = True

    def _check_caught(self):
        """
        The villain wins by sharing a cell with the player. Checked both
        after the player moves and after the villain does, since either one
        can be the side that closes the gap.
        """
        if self.villain.caught_player(self.player):
            self.game_over = True
            self.won = False
            return True
        return False

    def on_player_moved(self):
        """Call this once, right after Controller.handle_key returns True."""
        if self.game_over:
            return

        # Walking into the villain counts. The villain no longer moves as
        # part of the player's turn, so this is the only place a player-side
        # collision can be noticed.
        if self._check_caught():
            return

        self._collect_item_if_present()
        self._check_win()

    def update(self, now):
        """
        Advance the villain's clock. Call this every frame, whether or not
        the player pressed anything -- the villain hunts on real time now, so
        standing still is no longer safe.
        """
        if self.game_over:
            return

        self.villain.update(self.player, now)
        self._check_caught()

    def status_line(self):
        if self.game_over:
            return "You win! Press q to quit." if self.won else "Caught! Press q to quit."
        return (
            f"Items left: {self.board.remaining_goal_items()} | "
            f"Villain: 1 move every {self.villain.move_period():.1f}s | "
            "WASD/Arrows to move, q to quit"
        )


def render(stdscr, model):
    board, player, villain = model.board, model.player, model.villain

    stdscr.clear()
    stdscr.addstr(0, 0, model.status_line())

    for y in range(board.SIZE):
        row_chars = []
        for x in range(board.SIZE):
            if (x, y) == (player.x, player.y):
                row_chars.append("P")
            elif (x, y) == (villain.x, villain.y):
                row_chars.append("V")
            else:
                row_chars.append(board.grid[y][x])
        stdscr.addstr(y + 2, 0, "".join(row_chars))

    stdscr.refresh()


def run_game(stdscr):
    board = DemoBoard(tiles={DemoBoard.GOALPIECE: TOTAL_GOAL_ITEMS, DemoBoard.ENDGOAL: 1})

    player_start, villain_start = board.get_spawn_positions()
    player = PlayerModel(*player_start)
    villain = Villain(*villain_start, board=board, total_goal_items=TOTAL_GOAL_ITEMS)

    controller = Controller(board, player)
    model = GameModel(board, player, villain, TOTAL_GOAL_ITEMS)

    curses.curs_set(0)
    # Wait at most FRAME_MS for a keypress instead of blocking on one, so the
    # loop keeps spinning and the villain's clock keeps running while the
    # player is thinking. getch() returns -1 when nothing was pressed.
    stdscr.timeout(FRAME_MS)

    while True:
        render(stdscr, model)
        key = stdscr.getch()

        if key in QUIT_KEYS:
            break

        if model.game_over:
            # Once the game has ended, only quitting does anything.
            continue

        if key in DIRECTIONS and controller.handle_key(key):
            model.on_player_moved()

        model.update(time.monotonic())


def main():
    curses.wrapper(run_game)


if __name__ == "__main__":
    main()