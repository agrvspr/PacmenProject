import random

ACTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


def _sign(n):
    return (n > 0) - (n < 0)


def remaining_items(board, total):
    """
    Count goal items remaining on the board, falling back to total.
    Args: board, total - fallback count if board lacks the method.
    Returns: int number of goal items still present.
    """
    getter = getattr(board, "remaining_goal_items", None)
    return total if getter is None else getter()


class PriorityQueue:
    """
    Priority queue implementation, taken from Genetic Algorithm homework
    """
    def __init__(self):
        self.list = []

    def is_empty(self):
        return len(self.list) == 0

    def push(self, p, v):
        self.list.append((p, v))
        self.list.sort(key=lambda n: n[0], reverse=False)

    def pop(self):
        return self.list.pop(0)[1]


class Node:
    """
    A* search node, matching the shape of the Genetic Algorithm homework's
    A* (state / cost / path / get_neighbors / is_goal), adapted to a plain
    (x, y) grid state with a board instead of a Level with player health.
    """
    def __init__(self, state, board, cost, path=None):
        self.state = state
        self.board = board
        self.cost = cost
        self.path = path if path is not None else []

    def is_goal(self, goal):
        return self.state[0] == goal[0] and self.state[1] == goal[1]

    def get_neighbors(self):
        x, y = self.state
        candidates = ((x + dx, y + dy) for dx, dy in ACTIONS.values())
        return [c for c in candidates if self.board.is_walkable(*c)]


class ScoutBrain:
    """
    The smart one, that knows where the player is and the whole board.
    Computes a path to the player through A* and only reveals a small window to the dumb brain.
    """
    def __init__(self, board, villain, total_goal_items, hint_size=5, max_hint_size=None):
        self.board = board
        self.villain = villain
        self.total_goal_items = total_goal_items
        self.hint_size = hint_size
        self.max_hint_size = max_hint_size if max_hint_size is not None else hint_size + total_goal_items

    def _heuristic(self, a, b):
        """
        Compute Manhattan distance between two points (admissible heuristic).
        Args: a, b - (x, y) tuples.
        Returns: int Manhattan distance.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, player):
        """
        Run A* from the villain to the player using Manhattan distance heuristic.
        Args: player - object with x, y attributes.
        Returns: list of (x,y) cells from villain to player, or None if unreachable.
        """
        start = (self.villain.x, self.villain.y)
        goal = (player.x, player.y)

        pq = PriorityQueue()
        visited = set()
        start_node = Node(start, self.board, cost=0)
        pq.push(self._heuristic(start, goal), start_node)

        while not pq.is_empty():
            node = pq.pop()

            if node.state not in visited:
                visited.add(node.state)
                path = node.path[:] + [node.state]

                if node.is_goal(goal):
                    return path

                for move in node.get_neighbors():
                    new_node = Node(move, self.board, len(path), path)
                    pq.push(new_node.cost + self._heuristic(move, goal), new_node)

        return None

    def current_hint_size(self):
        """
        Compute the hint window size, which grows as more items are collected.
        Args: none.
        Returns: int hint size
        """
        collected_items = self.total_goal_items - self.board.remaining_goal_items()
        hint_size = min(self.hint_size + collected_items, self.max_hint_size)
        return hint_size

    def get_hint(self, player):
        """
        Return the next few cells of the A* path toward the player.
        Args: player - object with x, y attributes.
        Returns: list of (x,y) cells, empty if player is unreachable.
        """
        path = self.find_path(player)
        if not path or len(path) < 2:
            return []
        return path[1:1 + self.current_hint_size()]


class HunterBrain:
    """
    The dumb one, that only sees a small window around itself and moves randomly.
    """

    def __init__(self, epsilon=0.3, learning_rate=0.1, gamma=0.9):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.q_table = {}
        self.last_state = None
        self.last_action = None

    def encode_state(self, villain_pos, hint_region):
        """
        Encode villain position and hint region into a hashable Q-learning state.
        Args: villain_pos (x,y), hint_region - list of (x,y) cells.
        Returns: tuple state usable as a Q-table key.
        """
        if not hint_region:
            return (0, 0, 0, 0, False)

        x, y = villain_pos
        near_x, near_y = hint_region[0]
        far_x, far_y = hint_region[-1]
        return (
            _sign(near_x - x), _sign(near_y - y),
            _sign(far_x - x), _sign(far_y - y),
            True,
        )

    def choose_action(self, state):
        """
        Pick an action via epsilon-greedy selection over the Q-table.
        Args: state - hashable state key.
        Returns: action string, one of ACTIONS' keys.
        """
        self.last_state = state

        if random.random() < self.epsilon:
            action = random.choice(list(ACTIONS))
        else:
            values = {a: self.q_table.get((state, a), 0.0) for a in ACTIONS}
            best = max(values.values())
            action = random.choice([a for a, q in values.items() if q == best])

        self.last_action = action
        return action

    def update(self, reward, next_state):
        """
        Apply a Q-learning update for the last chosen action.
        Args: reward - float reward, next_state - resulting state.
        Returns: None; does nothing if no prior action exists.
        """
        if self.last_state is None or self.last_action is None:
            return

        key = (self.last_state, self.last_action)
        old = self.q_table.get(key, 0.0)
        future = max(self.q_table.get((next_state, a), 0.0) for a in ACTIONS)
        self.q_table[key] = old + self.learning_rate * (reward + self.gamma * future - old)


class Villain:
    """
    Villian model. Nefarious actions will be committed.
    """

    CATCH_REWARD = 10.0
    ON_HINT_STEP_REWARD = 1.0
    ON_HINT_REWARD = 0.5
    OFF_HINT_PENALTY = -0.5
    BLOCKED_PENALTY = -1.0
    MOVE_PERIODS = (1.0, 0.7, 0.4, 0.1)
    EPSILON_LEVELS = (0.2, 0.1, 0.05, 0.0)

    def __init__(self, x, y, board, total_goal_items=3, move_periods=None, epsilon_levels=None):
        self.x = x
        self.y = y
        self.board = board
        self.total_goal_items = total_goal_items
        self.move_periods = move_periods or self.MOVE_PERIODS
        self.epsilon_levels = epsilon_levels or self.EPSILON_LEVELS
        self.last_move_time = None
        self.scout = ScoutBrain(board, self, total_goal_items)
        self.hunter = HunterBrain(epsilon=self.epsilon_levels[0])

    @property
    def pos(self):
        return (self.x, self.y)

    def _collected_items(self):
        remaining = remaining_items(self.board, self.total_goal_items)
        return self.total_goal_items - remaining

    def move_period(self):
        """
        Look up seconds between moves based on items collected so far.
        Args: none.
        Returns: float seconds, clamped to the fastest table entry.
        """
        remaining = remaining_items(self.board, self.total_goal_items)
        collected = self.total_goal_items - remaining
        return self.move_periods[min(collected, len(self.move_periods) - 1)]

    def _current_epsilon(self):
        """
        Look up the Hunter's epsilon based on items collected so far.
        Args: none.
        Returns: float epsilon, clamped to the lowest table entry.
        """
        collected = self._collected_items()
        return self.epsilon_levels[min(collected, len(self.epsilon_levels) - 1)]

    def update(self, player, now):
        """
        Advance the villain's move clock, moving at most once per call.
        Args: player - object with x, y attributes, now - current time.
        Returns: bool True if the player was caught this move.
        """
        if self.last_move_time is None:
            self.last_move_time = now
            return False

        if now - self.last_move_time < self.move_period():
            return False

        self.last_move_time = now
        self._take_single_move(player)
        return self.caught_player(player)

    def _take_single_move(self, player):
        """
        Execute one villain turn: chase directly, or hint-guided Hunter move.
        Args: player - object with x, y attributes.
        Returns: None.
        """
        if remaining_items(self.board, self.total_goal_items) == 0:
            self._chase_directly(player)
            return

        self.hunter.epsilon = self._current_epsilon()
        hint_region = self.scout.get_hint(player)
        state = self.hunter.encode_state((self.x, self.y), hint_region)
        action = self.hunter.choose_action(state)

        dx, dy = ACTIONS[action]
        new_x, new_y = self.x + dx, self.y + dy

        moved = False
        if self.board.is_walkable(new_x, new_y):
            self.x, self.y = new_x, new_y
            moved = True

        reward = self._compute_reward(moved, hint_region, player)
        next_state = self.hunter.encode_state((self.x, self.y), hint_region)
        self.hunter.update(reward, next_state)

    def _chase_directly(self, player):
        """
        Step along the shortest path straight at the player (no hint/Hunter).
        Args: player - object with x, y attributes.
        Returns: None.
        """
        path = self.scout.find_path(player)
        if path and len(path) >= 2:
            self.x, self.y = path[1]

    def distance_to(self, player):
        """
        Compute Manhattan distance from the villain to the player.
        Args: player - object with x, y attributes.
        Returns: int Manhattan distance.
        """
        return abs(self.x - player.x) + abs(self.y - player.y)

    def _compute_reward(self, moved, hint_region, player):
        """
        Compute the Hunter's reward for its last move based on the hint.
        Args: moved - bool, hint_region - list of (x,y), player - object.
        Returns: float reward value.
        """
        if self.caught_player(player):
            return self.CATCH_REWARD

        if not moved:
            return self.BLOCKED_PENALTY

        if not hint_region:
            return 0.0

        if self.pos == hint_region[0]:
            return self.ON_HINT_STEP_REWARD
        if self.pos in hint_region:
            return self.ON_HINT_REWARD
        return self.OFF_HINT_PENALTY

    def caught_player(self, player):
        """
        Check if the villain occupies the same cell as the player.
        Args: player - object with x, y attributes.
        Returns: bool True if caught.
        """
        return (self.x, self.y) == (player.x, player.y)