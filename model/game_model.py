"""
GameModel: the rules glue between Board, PlayerModel, and Villain.
"""

from model.boardGeneration import DemoBoard

TOTAL_GOAL_ITEMS = 4


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
        """Call this once, right after a successful player move."""
        if self.game_over:
            return
        if self._check_caught():
            return

        self._collect_item_if_present()
        self._check_win()

    def update(self, now):
        """
        Advance the villain's clock. Call this every frame, whether or not
        the player pressed anything. The villain hunts on real time now, so
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
