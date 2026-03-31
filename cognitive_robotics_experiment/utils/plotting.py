"""
Plotting utilities — charts and graphs.

Creates learning curves, comparison bar charts, and
before/after adaptation visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def _smooth(data, window=50):
    """Simple moving average for smoothing noisy curves."""
    if len(data) < window:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def plot_learning_curve(rewards_history, title="Q-Learning: Reward per Episode",
                        save_path=None):
    """
    Plot raw and smoothed reward over training episodes.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    episodes = range(1, len(rewards_history) + 1)

    ax.plot(episodes, rewards_history, alpha=0.25, color="#3498DB",
            linewidth=0.8, label="Raw reward")

    smoothed = _smooth(rewards_history)
    smooth_x = range(50, 50 + len(smoothed))
    ax.plot(smooth_x, smoothed, color="#E74C3C", linewidth=2.2,
            label="Smoothed (window=50)")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [saved] {save_path}")

    return fig


def plot_steps_curve(steps_history, title="Q-Learning: Steps per Episode",
                     save_path=None):
    """
    Plot steps-to-goal over training episodes.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    episodes = range(1, len(steps_history) + 1)

    ax.plot(episodes, steps_history, alpha=0.25, color="#2ECC71",
            linewidth=0.8, label="Raw steps")

    smoothed = _smooth(steps_history)
    smooth_x = range(50, 50 + len(smoothed))
    ax.plot(smooth_x, smoothed, color="#8E44AD", linewidth=2.2,
            label="Smoothed (window=50)")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Steps to Goal", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [saved] {save_path}")

    return fig


def plot_comparison_bar(metrics_list, title="Agent Performance Comparison",
                        save_path=None):
    """
    Grouped bar chart comparing agents across key metrics.
    """
    agent_names = [m["agent_name"] for m in metrics_list]
    metrics_keys = [
        ("mean_reward", "Mean Reward"),
        ("mean_steps", "Mean Steps"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#3498DB", "#E74C3C", "#2ECC71", "#F39C12"]

    for idx, (key, ylabel) in enumerate(metrics_keys):
        ax = axes[idx]
        values = [m[key] for m in metrics_list]
        bars = ax.bar(agent_names, values,
                      color=colors[:len(agent_names)],
                      edgecolor="#2C3E50", linewidth=1.2, width=0.5)

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(ylabel, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [saved] {save_path}")

    return fig


def plot_success_rate_bar(metrics_list, title="Success Rate Comparison",
                          save_path=None):
    """Bar chart comparing success rates."""
    fig, ax = plt.subplots(figsize=(7, 5))
    agent_names = [m["agent_name"] for m in metrics_list]
    rates = [m["success_rate"] * 100 for m in metrics_list]
    colors = ["#3498DB", "#E74C3C", "#2ECC71", "#F39C12"]

    bars = ax.bar(agent_names, rates,
                  color=colors[:len(agent_names)],
                  edgecolor="#2C3E50", linewidth=1.2, width=0.5)

    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [saved] {save_path}")

    return fig


def plot_adaptation_comparison(before_metrics, after_metrics,
                               title="Adaptation After Environment Change",
                               save_path=None):
    """
    Grouped bar chart showing before/after environment change performance.
    """
    agent_names = [m["agent_name"] for m in before_metrics]
    n = len(agent_names)
    x = np.arange(n)
    width = 0.3

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (key, ylabel) in enumerate([("mean_reward", "Mean Reward"),
                                          ("success_rate", "Success Rate")]):
        ax = axes[idx]
        before_vals = [m[key] for m in before_metrics]
        after_vals = [m[key] for m in after_metrics]

        if key == "success_rate":
            before_vals = [v * 100 for v in before_vals]
            after_vals = [v * 100 for v in after_vals]

        bars1 = ax.bar(x - width / 2, before_vals, width,
                       label="Before Change", color="#3498DB",
                       edgecolor="#2C3E50", linewidth=1)
        bars2 = ax.bar(x + width / 2, after_vals, width,
                       label="After Change", color="#E74C3C",
                       edgecolor="#2C3E50", linewidth=1)

        ax.set_xticks(x)
        ax.set_xticklabels(agent_names, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(ylabel, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [saved] {save_path}")

    return fig
