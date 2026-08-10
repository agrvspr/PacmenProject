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

from boardGeneration import DemoBoard  # noqa: F401  (old random generator)
from board_ga import GeneticBoard
from player import PlayerModel
from villain import Villain
from controller import Controller, DIRECTIONS, QUIT_KEYS
from game_model import GameModel, TOTAL_GOAL_ITEMS

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


def render(stdscr, model):
    board, player, villain = model.board, model.player, model.villain
    max_y, max_x = stdscr.getmaxyx()

    stdscr.clear()
    status = model.status_line()[: max(0, max_x - 1)]
    if max_y > 0 and max_x > 0:
        stdscr.addstr(0, 0, status)

    # Rows begin at y=2, so usable board rows are whatever still fits below.
    drawable_rows = max(0, max_y - 2)
    visible_rows = min(board.SIZE, drawable_rows)
    visible_cols = min(board.SIZE, max(0, max_x - 1))

    if visible_rows < board.SIZE or visible_cols < board.SIZE:
        resize_hint = "Resize terminal to view full board"
        hint = resize_hint[: max(0, max_x - 1)]
        if max_y > 1 and max_x > 0:
            stdscr.addstr(1, 0, hint)

    for y in range(visible_rows):
        row_chars = []
        for x in range(visible_cols):
            if (x, y) == (player.x, player.y):
                row_chars.append("P")
            elif (x, y) == (villain.x, villain.y):
                row_chars.append("V")
            else:
                row_chars.append(board.grid[y][x])
        if max_x > 0:
            stdscr.addstr(y + 2, 0, "".join(row_chars))

    stdscr.refresh()


def run_game(stdscr):
    # Old random generator -- kept for comparison. It scatters obstacles with
    # no reachability check, so about 0.6% of its levels have a walled-off
    # cheese or exit and cannot be won.
    # board = DemoBoard(tiles={DemoBoard.GOALPIECE: TOTAL_GOAL_ITEMS, DemoBoard.ENDGOAL: 1})

    # Genetic algorithm generator: breeds a level against the fitness function
    # in board_ga. Takes about a second, and every level it returns is solvable
    # by construction because entities are placed inside one connected region.
    board = GeneticBoard(population_size=24, generations=18)

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