"""
SOAR-inspired Production Rule Agent.

Uses a fixed set of IF-THEN production rules to navigate the grid world.
This agent does NOT learn — it applies rules in priority order each step.
"""

import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


class ProductionAgent:
    """
    A rule-based agent inspired by the SOAR cognitive architecture.

    Production rules are evaluated in priority order:
      1. Move toward goal if the direct path is clear
      2. Avoid obstacles by choosing an alternative clear direction
      3. Random fallback if all preferred moves are blocked
    """

    def __init__(self):
        self.name = "Production Rule Agent"
        self.rule_log = []  # log of which rules fired

    def reset(self):
        """Reset internal logs for a new episode."""
        self.rule_log = []

    def _is_safe(self, state, obstacles, grid_size):
        """Check if a state is within bounds and free of obstacles."""
        r, c = state
        return (0 <= r < grid_size and 0 <= c < grid_size
                and (r, c) not in obstacles)

    def _manhattan_distance(self, a, b):
        """Manhattan distance between two (row, col) positions."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def act(self, state, env):
        """
        Select an action based on production rules.

        Parameters
        ----------
        state : tuple (row, col)
        env   : GridWorld instance (used to read obstacles & grid size)

        Returns
        -------
        action : int
        """
        goal = env.goal
        obstacles = env.obstacles
        grid_size = env.grid_size
        r, c = state

        # ── Rule 1: Move directly toward the goal if path is clear ──
        goal_actions = []
        if goal[0] < r:
            goal_actions.append(0)  # UP
        elif goal[0] > r:
            goal_actions.append(1)  # DOWN
        if goal[1] < c:
            goal_actions.append(2)  # LEFT
        elif goal[1] > c:
            goal_actions.append(3)  # RIGHT

        for action in goal_actions:
            dr, dc = config.ACTION_DELTAS[action]
            next_state = (r + dr, c + dc)
            if self._is_safe(next_state, obstacles, grid_size):
                self.rule_log.append(
                    f"Rule 1 (move toward goal): state={state}, "
                    f"action={config.ACTION_NAMES[action]}"
                )
                return action

        # ── Rule 2: Choose any safe action that reduces distance ──
        candidates = []
        for action in config.ACTIONS:
            dr, dc = config.ACTION_DELTAS[action]
            next_state = (r + dr, c + dc)
            if self._is_safe(next_state, obstacles, grid_size):
                dist = self._manhattan_distance(next_state, goal)
                candidates.append((dist, action))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            best_action = candidates[0][1]
            self.rule_log.append(
                f"Rule 2 (avoid obstacle, pick best): state={state}, "
                f"action={config.ACTION_NAMES[best_action]}"
            )
            return best_action

        # ── Rule 3: Random fallback (all moves are blocked/bad) ──
        action = random.choice(config.ACTIONS)
        self.rule_log.append(
            f"Rule 3 (random fallback): state={state}, "
            f"action={config.ACTION_NAMES[action]}"
        )
        return action

    def run_episode(self, env):
        """
        Run a single episode in the given environment.

        Returns
        -------
        dict with keys: total_reward, steps, success, path
        """
        self.reset()
        state = env.reset()
        total_reward = 0
        path = [state]

        while not env.done:
            action = self.act(state, env)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
            path.append(state)

        return {
            "total_reward": total_reward,
            "steps": env.steps,
            "success": state == env.goal,
            "path": path,
            "rule_log": list(self.rule_log),
        }
