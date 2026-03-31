"""
Main Experiment Runner — Cognitive Robotics Experiment.

Workflow:
  1. Run production rule agent on the original environment
  2. Train Q-learning agent on the original environment
  3. Evaluate both agents — compare performance
  4. Modify environment (change obstacles)
  5. Evaluate both agents on the new environment
  6. Retrain Q-learning agent and re-evaluate
  7. Generate all plots and logs
"""

import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

from environment.grid_world import GridWorld
from agents.production_agent import ProductionAgent
from agents.q_learning_agent import QLearningAgent
from evaluation import evaluate_agent, print_metrics, compare_agents
from utils.visualization import plot_grid, plot_grid_comparison, plot_q_values
from utils.plotting import (
    plot_learning_curve, plot_steps_curve,
    plot_comparison_bar, plot_success_rate_bar,
    plot_adaptation_comparison,
)

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


def ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)


def save_log(data, filename):
    """Save a JSON log file."""
    path = os.path.join(config.LOGS_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  [log saved] {path}")


def main():
    ensure_dirs()
    np.random.seed(42)

    print("=" * 60)
    print("  COGNITIVE ROBOTICS EXPERIMENT")
    print("  Symbolic (Production Rule) vs Learning (Q-Learning)")
    print("=" * 60)

    # ────────────────────────────────────────────────────────────
    # PHASE 1: Original Environment
    # ────────────────────────────────────────────────────────────
    print("\n>> PHASE 1: Original Environment Setup")
    env = GridWorld()
    print(f"  Grid: {config.GRID_SIZE}x{config.GRID_SIZE}")
    print(f"  Start: {config.START_STATE}  →  Goal: {config.GOAL_STATE}")
    print(f"  Obstacles: {config.OBSTACLES}")
    print(f"\n{env.render_text()}\n")

    # ── Production Agent ──
    print("-" * 60)
    print(">> Running Production Rule Agent...")
    prod_agent = ProductionAgent()
    prod_metrics = evaluate_agent(prod_agent, env, label="Production Rule")
    print_metrics(prod_metrics, header="Production Rule Agent — Original Env")

    # Save best path visualization
    plot_grid(env, path=prod_metrics["best_path"],
              title="Production Agent — Best Path (Original)",
              save_path=os.path.join(config.PLOTS_DIR, "prod_path_original.png"))

    # ── Q-Learning Agent ──
    print(">> Training Q-Learning Agent...")
    q_agent = QLearningAgent()
    train_result = q_agent.train(env, num_episodes=config.NUM_EPISODES)

    # Learning curves
    plot_learning_curve(train_result["rewards_history"],
                        title="Q-Learning: Reward per Episode (Original Env)",
                        save_path=os.path.join(config.PLOTS_DIR,
                                               "q_learning_curve.png"))
    plot_steps_curve(train_result["steps_history"],
                     title="Q-Learning: Steps per Episode (Original Env)",
                     save_path=os.path.join(config.PLOTS_DIR,
                                            "q_steps_curve.png"))

    # Q-value heatmap
    plot_q_values(q_agent.q_table,
                  title="Q-Values After Training (Original Env)",
                  save_path=os.path.join(config.PLOTS_DIR, "q_values_heatmap.png"))

    # Evaluate trained Q-agent
    q_metrics = evaluate_agent(q_agent, env, label="Q-Learning")
    print_metrics(q_metrics, header="Q-Learning Agent — Original Env")

    # Q-agent best path
    plot_grid(env, path=q_metrics["best_path"],
              title="Q-Learning Agent — Best Path (Original)",
              save_path=os.path.join(config.PLOTS_DIR, "q_path_original.png"))

    # ── Comparison (Original Env) ──
    compare_agents([prod_metrics, q_metrics])

    plot_comparison_bar([prod_metrics, q_metrics],
                        title="Agent Comparison — Original Environment",
                        save_path=os.path.join(config.PLOTS_DIR,
                                               "comparison_original.png"))
    plot_success_rate_bar([prod_metrics, q_metrics],
                          title="Success Rate — Original Environment",
                          save_path=os.path.join(config.PLOTS_DIR,
                                                 "success_rate_original.png"))

    # Save metrics log
    save_log({
        "phase": "original",
        "production": {k: v for k, v in prod_metrics.items()
                       if k not in ["rewards", "steps", "successes", "best_path"]},
        "q_learning": {k: v for k, v in q_metrics.items()
                       if k not in ["rewards", "steps", "successes", "best_path"]},
    }, "phase1_original_metrics.json")

    # ────────────────────────────────────────────────────────────
    # PHASE 2: Modified Environment (Adaptability Test)
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(">> PHASE 2: Modified Environment (Adaptability Test)")
    print("=" * 60)
    print(f"  New Obstacles: {config.MODIFIED_OBSTACLES}")

    env_modified = GridWorld(obstacles=config.MODIFIED_OBSTACLES)
    print(f"\n{env_modified.render_text()}\n")

    # Evaluate both agents on modified env WITHOUT retraining
    print(">> Evaluating Production Agent on modified environment...")
    prod_metrics_mod = evaluate_agent(prod_agent, env_modified,
                                       label="Production Rule")
    print_metrics(prod_metrics_mod,
                  header="Production Rule Agent — Modified Env (no change)")

    print(">> Evaluating Q-Learning Agent on modified environment (no retrain)...")
    q_metrics_mod_before = evaluate_agent(q_agent, env_modified,
                                           label="Q-Learning (stale)")
    print_metrics(q_metrics_mod_before,
                  header="Q-Learning Agent — Modified Env (NOT retrained)")

    # ── Retrain Q-agent on modified env ──
    print(">> Retraining Q-Learning Agent on modified environment...")
    q_agent.epsilon = config.EPSILON_START  # reset exploration
    retrain_result = q_agent.train(env_modified,
                                    num_episodes=config.RETRAIN_EPISODES)

    plot_learning_curve(retrain_result["rewards_history"],
                        title="Q-Learning: Reward (Retraining on Modified Env)",
                        save_path=os.path.join(config.PLOTS_DIR,
                                               "q_retrain_curve.png"))

    q_metrics_mod_after = evaluate_agent(q_agent, env_modified,
                                          label="Q-Learning (retrained)")
    print_metrics(q_metrics_mod_after,
                  header="Q-Learning Agent — Modified Env (RETRAINED)")

    # ── Comparison (Modified Env) ──
    compare_agents([prod_metrics_mod, q_metrics_mod_before, q_metrics_mod_after])

    # Side-by-side grid comparison
    plot_grid_comparison(
        env, env_modified,
        path_original=q_metrics["best_path"],
        path_modified=q_metrics_mod_after["best_path"],
        title="Environment Change & Agent Adaptation",
        save_path=os.path.join(config.PLOTS_DIR, "env_comparison.png"),
    )

    # Adaptation bar charts
    plot_adaptation_comparison(
        before_metrics=[prod_metrics, q_metrics],
        after_metrics=[prod_metrics_mod, q_metrics_mod_after],
        title="Agent Adaptation After Environment Change",
        save_path=os.path.join(config.PLOTS_DIR, "adaptation_comparison.png"),
    )

    # Q-value heatmap after retrain
    plot_q_values(q_agent.q_table,
                  title="Q-Values After Retraining (Modified Env)",
                  save_path=os.path.join(config.PLOTS_DIR,
                                         "q_values_retrained.png"))

    # Save phase 2 log
    save_log({
        "phase": "modified",
        "production": {k: v for k, v in prod_metrics_mod.items()
                       if k not in ["rewards", "steps", "successes", "best_path"]},
        "q_learning_before_retrain": {
            k: v for k, v in q_metrics_mod_before.items()
            if k not in ["rewards", "steps", "successes", "best_path"]
        },
        "q_learning_after_retrain": {
            k: v for k, v in q_metrics_mod_after.items()
            if k not in ["rewards", "steps", "successes", "best_path"]
        },
    }, "phase2_modified_metrics.json")

    # ────────────────────────────────────────────────────────────
    # SUMMARY
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"  Plots saved to : {config.PLOTS_DIR}")
    print(f"  Logs saved to  : {config.LOGS_DIR}")
    print(f"  Total plots    : {len(os.listdir(config.PLOTS_DIR))}")
    print("=" * 60)

    plt.close("all")


if __name__ == "__main__":
    main()
