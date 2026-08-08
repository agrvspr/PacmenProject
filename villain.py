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
    How many goal items are still on the board.

    The board does not expose this yet, so fall back to `total` (nothing
    collected) rather than crashing. Delete the fallback once the board
    grows a remaining_goal_items() method -- this is the single place the
    villain asks, so there is only one name to agree on.
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
        self.state = state  # (x, y)
        self.board = board  # so get_neighbors() needs no arguments, same as theirs
        self.cost = cost    # number of steps taken from start to reach this state
        # NOTE: default is None, not [], to avoid the classic Python mutable-
        # default-argument bug (a `path=[]` default is shared across every
        # call that doesn't pass path explicitly, and can leak state between
        # unrelated Node instances).
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
    def __init__(self, board, villain, total_goal_items, hint_size=5, min_hint_size=1):
        self.board = board
        self.villain = villain
        self.total_goal_items = total_goal_items
        self.hint_size = hint_size
        self.min_hint_size = min_hint_size

    def _heuristic(self, a, b):
        """Manhattan distance - admissible since movement is cardinal-only."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, player):
        """
        A* from the villain's current position to the player's, using
        Manhattan distance as the heuristic. Returns the full route
        including the villain's own cell at index 0, or None if the player
        cannot be reached.
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
                    # len(path) is the step count from start to `move`, since
                    # path already holds every cell up to and including node.
                    new_node = Node(move, self.board, len(path), path)
                    pq.push(new_node.cost + self._heuristic(move, goal), new_node)

        return None

    def current_hint_size(self):
        """
        Returns the current hint size based on how many goal items have been collected.
        The more items collected, the smaller the hint size.
        """
        remaining = remaining_items(self.board, self.total_goal_items)
        collected_items = self.total_goal_items - remaining
        return max(self.hint_size - collected_items, self.min_hint_size)

    def get_hint(self, player):
        """
        The next few cells of the A* route toward the player, trimmed to
        current_hint_size(). The villain's own cell is dropped, so the first
        entry is where the Scout would step next. Empty when the player is
        unreachable, which leaves the Hunter to guess.
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
        Turn villain_pos and hint_region into a hashable state for Q-learning.

        Two directions, both as compass signs: the Scout's immediate next
        step, and the furthest cell it revealed. The near direction has to be
        here because that is what the reward pays out on; the far one gives
        the Hunter a sense of where the route is heading. In a straight
        corridor they agree, but around a corner the hint may run down and
        then east, and a state built on only one of them cannot tell that
        case apart from a straight run.

        Deliberately coarse otherwise: encoding absolute coordinates would
        give one state per cell, so the Hunter would have to relearn the same
        lesson in every corridor.
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
        Epsilon-greedy: explore at random with probability epsilon, else take
        the best known action, breaking ties randomly so the Hunter does not
        always favour whichever direction happens to come first.
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
        Standard Q-learning update on the action returned by the last
        choose_action(). Does nothing if no action has been taken yet.
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

    CATCH_REWARD = 10.0        # catching the player dwarfs any single step
    ON_HINT_STEP_REWARD = 1.0  # stepped exactly where the Scout pointed
    ON_HINT_REWARD = 0.5       # stepped somewhere further along the hint
    OFF_HINT_PENALTY = -0.5    # legal move, but ignored the hint
    BLOCKED_PENALTY = -1.0     # walked into a wall

    def __init__(self, x, y, board, total_goal_items=3, max_speed=3, base_speed=1):
        self.x = x
        self.y = y
        self.board = board
        self.total_goal_items = total_goal_items
        self.max_speed = max_speed
        self.base_speed = base_speed
        self.scout = ScoutBrain(board, self, total_goal_items)
        self.hunter = HunterBrain()

    @property
    def pos(self):
        return (self.x, self.y)

    def current_speed(self):
        """
        Moves per turn, increasing by 1 for every goal item collected so
        far (read live from the board), capped at self.max_speed.
        """
        remaining = remaining_items(self.board, self.total_goal_items)
        collected = self.total_goal_items - remaining
        return min(self.max_speed, self.base_speed + collected)

    def take_turn(self, player):
        """
        Runs the villain's full turn, which may be multiple single moves
        depending on current_speed(). Returns True if the player was caught,
        and stops immediately when that happens so the villain does not keep
        moving past a catch.
        """
        for _ in range(self.current_speed()):
            self._take_single_move(player)
            if self.caught_player(player):
                return True
        return False

    def _take_single_move(self, player):
        """
        One move + one Hunter learning update:
          1. Ask Scout for a hint region based on current positions.
          2. Ask Hunter to choose an action from that hint.
          3. Check the resulting move against board.is_walkable().
          4. Apply the move if legal, otherwise stay put.
          5. Compute a reward (based on whether the move followed the
             Scout's hint) and let Hunter learn from it.
        """
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

    def distance_to(self, player):
        """Manhattan distance from the villain's current position to the player."""
        return abs(self.x - player.x) + abs(self.y - player.y)

    def _compute_reward(self, moved, hint_region, player):
        """
        Rewards the villain for following the Scout's hint, and penalizes it
        for a blocked move or for wandering off the hint.

        Scoring the hint rather than the true distance to the player is what
        makes the two-brain split real: the Hunter never sees where the
        player actually is, so the only way to earn reward is to learn that
        the Scout's window is worth trusting. Rewarding true distance would
        hand the Hunter the answer directly and leave the Scout decorative.

        moved:       whether the move was legal and applied
        hint_region: the cells the Scout revealed BEFORE this move
        """
        if self.caught_player(player):
            return self.CATCH_REWARD

        if not moved:
            return self.BLOCKED_PENALTY

        # No hint means the player is unreachable, so there is nothing to
        # follow and nothing to judge the move against.
        if not hint_region:
            return 0.0

        if self.pos == hint_region[0]:
            return self.ON_HINT_STEP_REWARD
        if self.pos in hint_region:
            return self.ON_HINT_REWARD
        return self.OFF_HINT_PENALTY

    def caught_player(self, player):
        """
        Check if the villain has caught the player.
        """
        return (self.x, self.y) == (player.x, player.y)
