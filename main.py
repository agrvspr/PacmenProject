"""
Main entry point that wires the Model (board, player, villain) together with
the movement Controller and a simple curses-based View.

Rules implemented here (not owned by any single existing file, so they live
in a small GameModel class):
  - Player walks a walled grid collecting GOALPIECE ("I") tiles.
  - Villain hunts the player with its Scout/Hunter split-brain AI, and
    speeds up as goal items are collected (see Villain.current_speed).
  - Player wins by collecting every goal item and then stepping onto
    ENDGOAL ("G").
  - Player loses the moment it shares a cell with the villain.

Controller.handle_key only knows how to move the player against the board;
it has no notion of items, the villain, or win/lose. GameModel is the glue
that runs after every successful player move.
"""
import curses
import random

from boardGeneration import DemoBoard
from player import PlayerModel
from villain import Villain
from controller import Controller, DIRECTIONS, QUIT_KEYS


TOTAL_GOAL_ITEMS = 3


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

    def on_player_moved(self):
        """Call this once, right after Controller.handle_key returns True."""
        if self.game_over:
            return

        self._collect_item_if_present()
        self._check_win()
        if self.game_over:
            return

        if self.villain.take_turn(self.player):
            self.game_over = True
            self.won = False

    def status_line(self):
        if self.game_over:
            return "You win! Press q to quit." if self.won else "Caught! Press q to quit."
        return (
            f"Items left: {self.board.remaining_goal_items()} | "
            f"Villain speed: {self.villain.current_speed()} | "
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
    stdscr.nodelay(False)

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


def main():
    curses.wrapper(run_game)


if __name__ == "__main__":
    main()