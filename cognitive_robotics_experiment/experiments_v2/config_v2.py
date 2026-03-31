"""
Upgraded Configuration for Cognitive Robotics Experiment v2.
10x10 grid with corridors, dead-end traps, and misleading paths.
"""
import os

GRID_SIZE = 10
START_STATE = (0, 0)
GOAL_STATE = (9, 9)

# Original obstacles: maze with corridors, greedy path works well
OBSTACLES_ORIGINAL = [
    (1, 2), (1, 8),
    (2, 2), (2, 5),
    (3, 5),
    (4, 0), (4, 1), (4, 7),
    (5, 3), (5, 7),
    (6, 3),
    (7, 1), (7, 5), (7, 6),
    (8, 1), (8, 8),
    (9, 4),
]

# Modified obstacles: FUNNEL TRAP for greedy agents
# Greedy agent goes DOWN from (0,0), enters pocket at rows 3-5,
# gets trapped — all exits require moving AWAY from goal.
# Only viable path: RIGHT along row 0, then DOWN right side.
#
#   0 1 2 3 4 5 6 7 8 9
# 0 S . . . . . . . . .   <- must go RIGHT here
# 1 . . . X X X . . . .   <- wall blocks center-right at row 1
# 2 . . . X . . X . . .   <- pocket forms
# 3 X . . X . . X . . .   <- left+center walls — pocket deepens
# 4 X . X X . . X . . .   <- dead-end: (4,1) surrounded on 3 sides
# 5 X X X . . . . . . .   <- bottom of trap: wall seals left side
# 6 . . . . X . . . X .   <- scattered obstacles
# 7 . . . . X . . . . .   <- forces path east
# 8 . . . . . . . . . .   <- open
# 9 . . . . . . . . . G   <- goal
OBSTACLES_MODIFIED = [
    # LEFT FUNNEL TRAP (rows 3-5, cols 0-2): greedy agent enters, can't exit
    (3, 0), (4, 0), (5, 0),        # Left wall
    (4, 2), (5, 1), (5, 2),        # Bottom/right of pocket
    # CENTER WALL blocks direct paths
    (1, 3), (1, 4), (1, 5),        # Row 1 wall
    (2, 3), (2, 6),                # Row 2 funnels into pocket
    (3, 3), (3, 6),                # Row 3 continues wall
    (4, 3), (4, 6),                # Row 4 continues wall
    # LOWER SCATTERED — force non-trivial path on right side
    (6, 4), (6, 8),
    (7, 4),
]

ACTIONS = [0, 1, 2, 3]
ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
ACTION_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

REWARD_GOAL = 10
REWARD_STEP = -1
REWARD_OBSTACLE = -10
REWARD_WALL = -5
MAX_STEPS = 200

ALPHA = 0.1
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
NUM_EPISODES = 1000
RETRAIN_EPISODES = 1000
EVAL_EPISODES = 100

PERCEPTION_RANGE = 1
VISIT_MEMORY = 8

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_v2")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
