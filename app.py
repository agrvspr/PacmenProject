"""
Flask server exposing the real game (boardGeneration.DemoBoard,
player.PlayerModel, villain.Villain, game_model.GameModel) over a small
JSON API, so a browser page can play against the actual Scout/Hunter
villain AI instead of a re-implemented stand-in.

Run with:
    pip install -r requirements.txt
    python app.py
then open http://127.0.0.1:5000 in a browser.
"""
import threading
import time

from flask import Flask, jsonify, request, send_from_directory

from model.boardGeneration import DemoBoard  # noqa: F401  (old random generator)
from model.board_ga import GeneticBoard
from model.player import PlayerModel
from model.villain import Villain
from model.game_model import GameModel, TOTAL_GOAL_ITEMS

app = Flask(__name__, static_folder="static", static_url_path="")

# A single in-memory game shared by anyone hitting this server. Fine for a
# local single-player demo; a lock keeps concurrent requests (a keypress and
# the polling loop landing at the same moment) from stepping on each other.
_lock = threading.Lock()
_state = {"model": None}

_DIRECTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


def _serialize(model):
    board = model.board
    return {
        "size": board.SIZE,
        "grid": ["".join(row) for row in board.grid],
        "player": {"x": model.player.x, "y": model.player.y},
        "villain": {"x": model.villain.x, "y": model.villain.y},
        "items_left": board.remaining_goal_items(),
        "total_items": model.total_goal_items,
        "move_period": model.villain.move_period(),
        "game_over": model.game_over,
        "won": model.won,
        "status": model.status_line(),
    }


def _new_model():
    # Old random generator -- kept for comparison, see board_ga for why it was
    # replaced (no reachability check, so some levels are unwinnable).
    # board = DemoBoard(tiles={DemoBoard.GOALPIECE: TOTAL_GOAL_ITEMS, DemoBoard.ENDGOAL: 1})

    board = GeneticBoard(population_size=24, generations=18)
    player_start, villain_start = board.get_spawn_positions()
    player = PlayerModel(*player_start)
    villain = Villain(*villain_start, board=board, total_goal_items=TOTAL_GOAL_ITEMS)
    return GameModel(board, player, villain, TOTAL_GOAL_ITEMS)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/new_game", methods=["POST"])
def new_game():
    with _lock:
        model = _new_model()
        _state["model"] = model
        return jsonify(_serialize(model))


@app.route("/api/state", methods=["GET"])
def get_state():
    with _lock:
        model = _state.get("model")
        if model is None:
            return jsonify({"error": "no active game"}), 400
        # Advances the villain's real-time clock exactly the way main.py's
        # frame loop does, one step at most per call.
        model.update(time.monotonic())
        return jsonify(_serialize(model))


@app.route("/api/move", methods=["POST"])
def move():
    with _lock:
        model = _state.get("model")
        if model is None:
            return jsonify({"error": "no active game"}), 400

        payload = request.get_json(silent=True) or {}
        direction = payload.get("direction")
        delta = _DIRECTIONS.get(direction)

        if delta and not model.game_over:
            dx, dy = delta
            new_x, new_y = model.player.x + dx, model.player.y + dy
            if model.board.is_walkable(new_x, new_y):
                model.player.move(dx, dy)
                model.on_player_moved()

        model.update(time.monotonic())
        return jsonify(_serialize(model))


@app.route("/api/quit", methods=["POST"])
def quit_game():
    with _lock:
        _state["model"] = None
        return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
