"""
Evaluation utilities for the Cognitive Robotics Experiment.

Provides functions to evaluate agents over multiple episodes and
aggregate metrics (steps to goal, total reward, success rate).
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def evaluate_agent(agent, env, num_episodes=None, label=None):
    """
    Evaluate an agent over multiple episodes.

    Parameters
    ----------
    agent        : ProductionAgent or QLearningAgent
    env          : GridWorld instance
    num_episodes : int (defaults to config.EVAL_EPISODES)
    label        : str (for display)

    Returns
    -------
    dict with aggregated metrics and per-episode details
    """
    if num_episodes is None:
        num_episodes = config.EVAL_EPISODES

    if label is None:
        label = getattr(agent, "name", "Agent")

    rewards = []
    steps_list = []
    successes = []
    all_paths = []

    for ep in range(num_episodes):
        result = agent.run_episode(env)
        rewards.append(result["total_reward"])
        steps_list.append(result["steps"])
        successes.append(result["success"])
        all_paths.append(result["path"])

    metrics = {
        "agent_name": label,
        "num_episodes": num_episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_steps": float(np.mean(steps_list)),
        "std_steps": float(np.std(steps_list)),
        "success_rate": float(np.mean(successes)),
        "rewards": rewards,
        "steps": steps_list,
        "successes": successes,
        "best_path": all_paths[int(np.argmax(rewards))],
    }

    return metrics


def print_metrics(metrics, header=None):
    """Pretty-print evaluation metrics."""
    if header:
        print(f"\n{'=' * 55}")
        print(f"  {header}")
        print(f"{'=' * 55}")

    print(f"  Agent         : {metrics['agent_name']}")
    print(f"  Episodes      : {metrics['num_episodes']}")
    print(f"  Mean Reward   : {metrics['mean_reward']:.2f} "
          f"(± {metrics['std_reward']:.2f})")
    print(f"  Mean Steps    : {metrics['mean_steps']:.2f} "
          f"(± {metrics['std_steps']:.2f})")
    print(f"  Success Rate  : {metrics['success_rate'] * 100:.1f}%")
    print(f"{'─' * 55}")


def compare_agents(metrics_list):
    """
    Print a side-by-side comparison table.

    Parameters
    ----------
    metrics_list : list of metric dicts from evaluate_agent()
    """
    print(f"\n{'=' * 65}")
    print(f"  AGENT COMPARISON")
    print(f"{'=' * 65}")
    header = f"{'Metric':<20}"
    for m in metrics_list:
        header += f" | {m['agent_name']:>18}"
    print(header)
    print(f"{'─' * 65}")

    rows = [
        ("Mean Reward", "mean_reward", ".2f"),
        ("Mean Steps", "mean_steps", ".2f"),
        ("Success Rate (%)", "success_rate", ".1%"),
    ]

    for label, key, fmt in rows:
        row = f"{label:<20}"
        for m in metrics_list:
            val = m[key]
            row += f" | {val:>18{fmt}}"
        print(row)

    print(f"{'=' * 65}\n")
