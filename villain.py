import random

ACTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}

class PriorityQueue:
    """
    Priority queue implementation, taken from Genetic Algorithm homework
    """
    def __init__(self):
        self.list = []
    
    def is_empty(self):
        return len(self.list) == 0 
    def push(self, p, v):
        self.list.append((p,v))
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
        A* from current position to player x and y
        heuristic is manhattan distance
        """
        start = (self.villain.x, self.villain.y)
        goal = (player.x, player.y)
 
        pq = PriorityQueue()
        visited = []
        start_node = Node(start, self.board, cost=0)
        pq.push(self._heuristic(start_node, goal), start_node)
 
        while not pq.is_empty():
            node = pq.pop()
 
            if node.state not in visited:
                visited.append(node.state)
                path = node.path[:] + [node.state]
 
                if node.is_goal(goal):
                    return path
 
                for move in node.get_neighbors():
                    new_node = Node(move, self.board, len(path) + 1, path)
                    pq.push(new_node.cost + self._heuristic(new_node, goal), new_node)
 
        return None

    def current_hint_size(self):
        """
        Returns the current hint size based on how many goal items have been collected.
        The more items collected, the smaller the hint size.
        """
        collected_items = self.total_goal_items - self.board.remaining_goal_items()
        hint_size = max(self.hint_size - collected_items, self.min_hint_size)
        return hint_size

    def get_hint(self, player):
        """
        Returns a list of (x, y) tuples representing the hint path to the player.
        The length of the list is determined by current_hint_size().
        """


class HunterBrain:
    """
    The dumb one, that only sees a small window around itself and moves randomly.
    """

    def __init__(self, epsilon=0.3, learning_rate = 0.1, gamma=0.9):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.q_table = {} 
        self.last_state = None
        self.last_action = None

    def encode_state(self, villain_pos, hint_region):
        """
        Turn villain_pos and hint_region into a hashable state representation for Q-learning.
        """

    def choose_action(self, state):
        """
        Choose an action based on epsilon-greedy policy.
        """

    def update(self, reward, next_state):
        """
        Update the Q-table based on the last action taken and the received reward.
        """

class Villain:
    """
    Villian model. Nefarious actions will be committed.
    """

    def __init__(self, x, y, board, total_goal_items=3, max_speed=3, base_speed=1):
        self.x = x
        self.y = y
        self.board = board
        self.max_speed = max_speed
        self.base_speed = base_speed
        self.scout = ScoutBrain(board, self, total_goal_items)
        self.hunter = HunterBrain()

    def current_speed(self):
        """
        Moves per turn, increasing by 1 for every goal item collected so
        far (read live from the board), capped at self.max_speed.
        """
        remaining = self.board.goal_items_remaining()
        collected = self.total_goal_items - remaining
        return min(self.max_speed, self.base_speed + collected)

    def take_turn(self, player):
        """
        Runs the villain's full turn, which may be multiple single moves
        depending on current_speed(). Stops early if the villain catches
        the player mid-turn, so it doesn't keep moving past a catch.
        """
        for _ in range(self.current_speed()):
            if self.caught_player(player):
                break
            self._take_single_move(player)

    def _take_single_move(self, player):
        """
        One move + one Hunter learning update:
          1. Ask Scout for a hint region based on current positions.
          2. Ask Hunter to choose an action from that hint.
          3. Check the resulting move against board.is_walkable().
          4. Apply the move if legal, otherwise stay put.
          5. Compute a reward (based on distance to player) and let
             Hunter learn from it.
        """
        hint_region = self.scout.get_hint(player)
        state = self.hunter.encode_state((self.x, self.y), hint_region)
        action = self.hunter.choose_action(state)
 
        prev_distance = self.distance_to(player)  # measured BEFORE moving
 
        dx, dy = ACTIONS[action]
        new_x, new_y = self.x + dx, self.y + dy
 
        moved = False
        if self.board.is_walkable(new_x, new_y):
            self.x, self.y = new_x, new_y
            moved = True
 
        reward = self._compute_reward(moved, prev_distance, player)
        next_state = self.hunter.encode_state((self.x, self.y), hint_region)
        self.hunter.update(reward, next_state)
 
    def distance_to(self, player):
        """Manhattan distance from the villain's current position to the player."""
        return abs(self.x - player.x) + abs(self.y - player.y)
 
    def _compute_reward(self, moved, prev_distance, player):
        """
        Rewards the villain for closing the distance to the player, and
        penalizes it for a blocked move or for moving further away.
 
        moved:         whether the move was legal and applied
        prev_distance: villain-to-player Manhattan distance BEFORE this move
        """
        if not moved:
            return -1.0
        
        new_distance = self.distance_to(player)
        if new_distance < prev_distance:
            return 1.0
        elif new_distance > prev_distance:
            return -0.5
        return 0.0

    def caught_player(self, player):
        """
        Check if the villain has caught the player.
        """
        return (self.x, self.y) == (player.x, player.y)