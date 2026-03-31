"""
Grid World Environment for Cognitive Robotics Experiment.

A 5x5 grid with obstacles, a start position, and a goal position.
The agent receives rewards/penalties based on its actions.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


class GridWorld:
    """
    A 2D grid-world environment.

    The agent starts at START_STATE and must navigate to GOAL_STATE
    while avoiding obstacles. Hitting a wall or obstacle incurs penalties.
    """

    def __init__(self, obstacles=None):
        self.grid_size = config.GRID_SIZE
        self.start = config.START_STATE
        self.goal = config.GOAL_STATE
        self.obstacles = obstacles if obstacles is not None else list(config.OBSTACLES)
        self.max_steps = config.MAX_STEPS
        self.state = None
        self.steps = 0
        self.done = False

    def reset(self):
        """Reset the environment to the start state."""
        self.state = self.start
        self.steps = 0
        self.done = False
        return self.state

    def is_valid(self, state):
        """Check if a state is within grid bounds (not obstacle check)."""
        r, c = state
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def step(self, action):
        """
        Execute an action and return (next_state, reward, done, info).

        Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() first.")

        dr, dc = config.ACTION_DELTAS[action]
        r, c = self.state
        new_state = (r + dr, c + dc)

        self.steps += 1
        info = {"action": config.ACTION_NAMES[action]}

        # Wall collision — stay in place, penalty
        if not self.is_valid(new_state):
            reward = config.REWARD_WALL
            info["event"] = "wall_collision"
            # state doesn't change

        # Obstacle collision — stay in place, penalty
        elif new_state in self.obstacles:
            reward = config.REWARD_OBSTACLE
            info["event"] = "obstacle_collision"
            # state doesn't change

        # Goal reached
        elif new_state == self.goal:
            self.state = new_state
            reward = config.REWARD_GOAL
            self.done = True
            info["event"] = "goal_reached"

        # Normal step
        else:
            self.state = new_state
            reward = config.REWARD_STEP
            info["event"] = "step"

        # Timeout
        if self.steps >= self.max_steps and not self.done:
            self.done = True
            info["event"] = "timeout"

        return self.state, reward, self.done, info

    def get_grid_matrix(self):
        """
        Return a 2D numpy array representing the grid.
        0 = empty, 1 = obstacle, 2 = start, 3 = goal
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for obs in self.obstacles:
            grid[obs] = 1
        grid[self.start] = 2
        grid[self.goal] = 3
        return grid

    def render_text(self, path=None):
        """Return a text representation of the grid."""
        symbols = {0: ".", 1: "X", 2: "S", 3: "G"}
        grid = self.get_grid_matrix()
        lines = []
        for r in range(self.grid_size):
            row_str = ""
            for c in range(self.grid_size):
                if path and (r, c) in path and (r, c) not in [self.start, self.goal]:
                    row_str += " * "
                elif (r, c) == self.state and self.state not in [self.start, self.goal]:
                    row_str += " A "
                else:
                    row_str += f" {symbols[grid[r, c]]} "
            lines.append(row_str)
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"GridWorld(size={self.grid_size}, start={self.start}, "
            f"goal={self.goal}, obstacles={self.obstacles})"
        )
