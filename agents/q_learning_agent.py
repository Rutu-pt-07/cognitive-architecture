"""
Q-Learning Agent for Cognitive Robotics Experiment.

Learns an optimal policy through interaction with the grid world
using temporal-difference (TD) learning with ε-greedy exploration.
"""

import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class QLearningAgent:
    """
    Tabular Q-Learning agent.

    Maintains a Q-table of shape (grid_size, grid_size, num_actions)
    and updates it via the Bellman equation after each step.
    """

    def __init__(self, alpha=None, gamma=None, epsilon=None,
                 epsilon_min=None, epsilon_decay=None):
        self.name = "Q-Learning Agent"
        self.grid_size = config.GRID_SIZE
        self.num_actions = len(config.ACTIONS)

        # Hyperparameters (use config defaults if not provided)
        self.alpha = alpha if alpha is not None else config.ALPHA
        self.gamma = gamma if gamma is not None else config.GAMMA
        self.epsilon = epsilon if epsilon is not None else config.EPSILON_START
        self.epsilon_min = epsilon_min if epsilon_min is not None else config.EPSILON_MIN
        self.epsilon_decay = epsilon_decay if epsilon_decay is not None else config.EPSILON_DECAY

        # Q-table: state (r, c) × action
        self.q_table = np.zeros(
            (self.grid_size, self.grid_size, self.num_actions)
        )

        # Training history
        self.rewards_history = []
        self.steps_history = []

    def reset_q_table(self):
        """Reset Q-table to zeros and clear history."""
        self.q_table = np.zeros(
            (self.grid_size, self.grid_size, self.num_actions)
        )
        self.epsilon = config.EPSILON_START
        self.rewards_history = []
        self.steps_history = []

    def act(self, state, explore=True):
        """
        Select an action using ε-greedy policy.

        Parameters
        ----------
        state   : tuple (row, col)
        explore : bool — if False, always exploit (greedy)

        Returns
        -------
        action : int
        """
        if explore and np.random.random() < self.epsilon:
            return np.random.choice(config.ACTIONS)
        r, c = state
        return int(np.argmax(self.q_table[r, c]))

    def update(self, state, action, reward, next_state, done):
        """
        Bellman update:
          Q(s,a) ← Q(s,a) + α [ r + γ·max Q(s',·) − Q(s,a) ]
        """
        r, c = state
        nr, nc = next_state
        best_next = np.max(self.q_table[nr, nc]) if not done else 0
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[r, c, action]
        self.q_table[r, c, action] += self.alpha * td_error

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min,
                          self.epsilon * self.epsilon_decay)

    def train(self, env, num_episodes=None):
        """
        Train the agent on the given environment.

        Parameters
        ----------
        env          : GridWorld instance
        num_episodes : int (defaults to config.NUM_EPISODES)

        Returns
        -------
        dict with rewards_history and steps_history
        """
        if num_episodes is None:
            num_episodes = config.NUM_EPISODES

        for ep in range(num_episodes):
            state = env.reset()
            total_reward = 0

            while not env.done:
                action = self.act(state, explore=True)
                next_state, reward, done, info = env.step(action)
                self.update(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state

            self.decay_epsilon()
            self.rewards_history.append(total_reward)
            self.steps_history.append(env.steps)

            if (ep + 1) % 200 == 0:
                avg_r = np.mean(self.rewards_history[-50:])
                print(f"  Episode {ep + 1}/{num_episodes} | "
                      f"Avg Reward (last 50): {avg_r:.2f} | "
                      f"ε: {self.epsilon:.4f}")

        return {
            "rewards_history": list(self.rewards_history),
            "steps_history": list(self.steps_history),
        }

    def run_episode(self, env, explore=False):
        """
        Run a single episode (evaluation mode by default).

        Returns
        -------
        dict with keys: total_reward, steps, success, path
        """
        state = env.reset()
        total_reward = 0
        path = [state]

        while not env.done:
            action = self.act(state, explore=explore)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
            path.append(state)

        return {
            "total_reward": total_reward,
            "steps": env.steps,
            "success": state == env.goal,
            "path": path,
        }

    def save_q_table(self, filepath):
        """Save Q-table to a .npy file."""
        np.save(filepath, self.q_table)

    def load_q_table(self, filepath):
        """Load Q-table from a .npy file."""
        self.q_table = np.load(filepath)
