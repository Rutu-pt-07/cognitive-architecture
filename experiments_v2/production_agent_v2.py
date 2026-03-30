"""
SOAR-inspired Production Rule Agent v2 — RIGID SYMBOLIC COGNITION.

This agent has TWO critical limitations that make it fail in changed environments:
  1. FIXED direction preferences learned from the original environment layout
  2. LOCAL perception only (sees 1 cell in each direction)
  3. Short visited memory — gets stuck in loops near obstacles

The rules encode knowledge about the ORIGINAL environment. When the environment
changes, these rules become wrong and the agent degrades.
"""
import random

ACTION_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
ALL_ACTIONS = [0, 1, 2, 3]


class ProductionAgentV2:
    """
    Production rule agent with environment-specific rules.

    The rules are designed for the ORIGINAL environment layout:
    - Upper half: prefer DOWN then RIGHT
    - Lower half: prefer RIGHT then DOWN
    - Specific waypoint-based rules for known obstacle patterns

    When the environment changes, these hardcoded preferences cause failures.
    """
    def __init__(self, visit_memory=8):
        self.name = "Production Rule Agent"
        self.visit_memory = visit_memory
        self.visited = []
        self.rule_log = []

    def reset(self):
        self.visited = []
        self.rule_log = []

    def _perceive_adjacent(self, state, env):
        """LOCAL perception — only sees 4 adjacent cells."""
        r, c = state
        perception = {}
        for action, (dr, dc) in ACTION_DELTAS.items():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < env.grid_size and 0 <= nc < env.grid_size):
                perception[action] = "wall"
            elif (nr, nc) in env.obstacles:
                perception[action] = "obstacle"
            elif (nr, nc) == env.goal:
                perception[action] = "goal"
            else:
                perception[action] = "free"
        return perception

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_preferred_actions(self, state):
        """
        FIXED direction preference rules — encoded for the ORIGINAL environment.
        These are the 'production rules' that work well in the original layout
        but become WRONG when obstacles change.
        """
        r, c = state
        # Rule set: go DOWN-RIGHT through the center corridor
        # This is the optimal strategy for the ORIGINAL environment
        if r < 4:
            # Upper portion: go DOWN first, then RIGHT
            return [1, 3, 2, 0]  # DOWN, RIGHT, LEFT, UP
        elif r < 7:
            # Middle portion: go RIGHT first to avoid known obstacles at cols 3,5
            if c < 5:
                return [3, 1, 0, 2]  # RIGHT, DOWN, UP, LEFT
            else:
                return [1, 3, 0, 2]  # DOWN, RIGHT, UP, LEFT
        else:
            # Lower portion: go DOWN and RIGHT toward goal
            return [1, 3, 2, 0]  # DOWN, RIGHT, LEFT, UP

    def act(self, state, env):
        perception = self._perceive_adjacent(state, env)
        r, c = state

        # Rule 0: If goal is adjacent, move to it
        for action, status in perception.items():
            if status == "goal":
                self.rule_log.append(f"R0(goal) {state}")
                return action

        preferred = self._get_preferred_actions(state)

        # Rule 1: Follow preferred direction order, avoid visited cells
        for action in preferred:
            if perception.get(action) == "free":
                dr, dc = ACTION_DELTAS[action]
                next_s = (r + dr, c + dc)
                if next_s not in self.visited[-self.visit_memory:]:
                    self.rule_log.append(f"R1(preferred_unvisited) {state}->{ACTION_NAMES[action]}")
                    return action

        # Rule 2: Follow preferred direction (allow revisit)
        for action in preferred:
            if perception.get(action) == "free":
                self.rule_log.append(f"R2(preferred_revisit) {state}->{ACTION_NAMES[action]}")
                return action

        # Rule 3: Any free cell
        for action in ALL_ACTIONS:
            if perception.get(action) == "free":
                self.rule_log.append(f"R3(any_free) {state}->{ACTION_NAMES[action]}")
                return action

        # Rule 4: Random (completely stuck)
        action = random.choice(ALL_ACTIONS)
        self.rule_log.append(f"R4(random) {state}")
        return action

    def run_episode(self, env):
        self.reset()
        state = env.reset()
        total_reward = 0
        path = [state]
        while not env.done:
            action = self.act(state, env)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            self.visited.append(next_state)
            state = next_state
            path.append(state)
        return {
            "total_reward": total_reward,
            "steps": env.steps,
            "success": state == env.goal,
            "path": path,
        }
