"""
Former main, now using app.py since we have a GUI now.
"""
import curses
import random
import time

from model.boardGeneration import DemoBoard  # noqa: F401
from model.board_ga import GeneticBoard
from model.player import PlayerModel
from model.villain import Villain
from controller.controller import Controller, DIRECTIONS, QUIT_KEYS
from model.game_model import GameModel, TOTAL_GOAL_ITEMS

FRAME_MS = 30


def _open_cells(board):
    return [
        (x, y)
        for y in range(board.SIZE)
        for x in range(board.SIZE)
        if board.grid[y][x] == DemoBoard.EMPTYSPACE
    ]


def _pick_start_positions(board):
    """
    Pick two distinct empty cells for the player and the villain.
    Args: board - board with SIZE and grid attributes.
    Returns: (player_cell, villain_cell) tuple of (x, y) positions.
    """
    cells = _open_cells(board)
    random.shuffle(cells)
    if len(cells) < 2:
        raise ValueError("Board doesn't have enough open space to place player and villain")
    player_cell = cells.pop()
    villain_cell = cells.pop()
    return player_cell, villain_cell


def render(stdscr, model):
    """
    Draw the status line and visible board onto the curses screen.
    Args: stdscr - curses screen, model - GameModel to render.
    Returns: None.
    """
    board, player, villain = model.board, model.player, model.villain
    max_y, max_x = stdscr.getmaxyx()

    stdscr.clear()
    status = model.status_line()[: max(0, max_x - 1)]
    if max_y > 0 and max_x > 0:
        stdscr.addstr(0, 0, status)

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
    """
    Set up a fresh game and run the curses input/render loop until quit.
    Args: stdscr - curses screen.
    Returns: None.
    """
    board = GeneticBoard(population_size=24, generations=18)

    player_start, villain_start = board.get_spawn_positions()
    player = PlayerModel(*player_start)
    villain = Villain(*villain_start, board=board, total_goal_items=TOTAL_GOAL_ITEMS)

    controller = Controller(board, player)
    model = GameModel(board, player, villain, TOTAL_GOAL_ITEMS)

    curses.curs_set(0)
    stdscr.timeout(FRAME_MS)

    while True:
        render(stdscr, model)
        key = stdscr.getch()

        if key in QUIT_KEYS:
            break

        if model.game_over:
            continue

        if key in DIRECTIONS and controller.handle_key(key):
            model.on_player_moved()

        model.update(time.monotonic())


def main():
    curses.wrapper(run_game)


if __name__ == "__main__":
    main()