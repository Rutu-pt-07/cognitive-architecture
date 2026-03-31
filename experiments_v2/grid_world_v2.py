"""
Grid World Environment v2 — 10x10 with corridors and traps.
"""
import numpy as np


class GridWorld:
    def __init__(self, grid_size, start, goal, obstacles, max_steps,
                 reward_goal=10, reward_step=-1, reward_obstacle=-10, reward_wall=-5):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.obstacles = list(obstacles)
        self.max_steps = max_steps
        self.reward_goal = reward_goal
        self.reward_step = reward_step
        self.reward_obstacle = reward_obstacle
        self.reward_wall = reward_wall
        self.action_deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        self.action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        self.state = None
        self.steps = 0
        self.done = False

    def reset(self):
        self.state = self.start
        self.steps = 0
        self.done = False
        return self.state

    def is_valid(self, state):
        r, c = state
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode done. Call reset().")
        dr, dc = self.action_deltas[action]
        r, c = self.state
        new_state = (r + dr, c + dc)
        self.steps += 1
        info = {"action": self.action_names[action]}

        if not self.is_valid(new_state):
            reward = self.reward_wall
            info["event"] = "wall"
        elif new_state in self.obstacles:
            reward = self.reward_obstacle
            info["event"] = "obstacle"
        elif new_state == self.goal:
            self.state = new_state
            reward = self.reward_goal
            self.done = True
            info["event"] = "goal"
        else:
            self.state = new_state
            reward = self.reward_step
            info["event"] = "step"

        if self.steps >= self.max_steps and not self.done:
            self.done = True
            info["event"] = "timeout"
        return self.state, reward, self.done, info

    def get_grid_matrix(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for obs in self.obstacles:
            grid[obs] = 1
        grid[self.start] = 2
        grid[self.goal] = 3
        return grid
