"""
Central configuration for the Cognitive Robotics Experiment.
All parameters for the grid world, agents, and experiment are defined here.
"""

# ─── Grid World ───────────────────────────────────────────────
GRID_SIZE = 5
START_STATE = (0, 0)
GOAL_STATE = (4, 4)

# Obstacle positions (row, col)
OBSTACLES = [(1, 1), (2, 3), (3, 1), (3, 3)]

# Modified obstacles for adaptability test
MODIFIED_OBSTACLES = [(0, 3), (1, 2), (2, 1), (3, 4)]

# Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
ACTIONS = [0, 1, 2, 3]
ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
ACTION_DELTAS = {
    0: (-1, 0),   # UP
    1: (1, 0),    # DOWN
    2: (0, -1),   # LEFT
    3: (0, 1),    # RIGHT
}

# ─── Rewards ──────────────────────────────────────────────────
REWARD_GOAL = 10
REWARD_STEP = -1
REWARD_OBSTACLE = -10
REWARD_WALL = -5

# Maximum steps per episode (prevents infinite loops)
MAX_STEPS = 100

# ─── Q-Learning Hyperparameters ───────────────────────────────
ALPHA = 0.1          # Learning rate
GAMMA = 0.95         # Discount factor
EPSILON_START = 1.0  # Initial exploration rate
EPSILON_MIN = 0.01   # Minimum exploration rate
EPSILON_DECAY = 0.995  # Decay per episode
NUM_EPISODES = 1000  # Training episodes

# ─── Experiment ───────────────────────────────────────────────
EVAL_EPISODES = 100  # Episodes for evaluation runs
RETRAIN_EPISODES = 500  # Episodes for retraining after env change

# ─── Paths ────────────────────────────────────────────────────
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiment_v1", "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
