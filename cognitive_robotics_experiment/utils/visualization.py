"""
Visualization utilities — grid rendering.

Creates matplotlib visualizations of the grid world, agent paths,
and Q-value heatmaps.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def plot_grid(env, path=None, title="Grid World", save_path=None, ax=None):
    """
    Render the grid world with optional agent path overlay.

    Color scheme:
      - White   : empty cell
      - Red     : obstacle
      - Green   : start
      - Gold    : goal
      - Blue    : agent path
    """
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    grid = env.get_grid_matrix()  # 0=empty, 1=obstacle, 2=start, 3=goal
    n = env.grid_size

    # Draw cells
    colors_map = {
        0: "#FFFFFF",   # empty
        1: "#E74C3C",   # obstacle (red)
        2: "#2ECC71",   # start (green)
        3: "#F1C40F",   # goal (gold)
    }

    for r in range(n):
        for c in range(n):
            color = colors_map[grid[r, c]]
            rect = plt.Rectangle((c, n - 1 - r), 1, 1,
                                 facecolor=color, edgecolor="#2C3E50",
                                 linewidth=1.5)
            ax.add_patch(rect)

            # Cell labels
            if grid[r, c] == 2:
                ax.text(c + 0.5, n - 1 - r + 0.5, "S",
                        ha="center", va="center", fontsize=14,
                        fontweight="bold", color="#1A5276")
            elif grid[r, c] == 3:
                ax.text(c + 0.5, n - 1 - r + 0.5, "G",
                        ha="center", va="center", fontsize=14,
                        fontweight="bold", color="#7D6608")
            elif grid[r, c] == 1:
                ax.text(c + 0.5, n - 1 - r + 0.5, "✖",
                        ha="center", va="center", fontsize=16,
                        color="#922B21")

    # Draw path
    if path and len(path) > 1:
        path_x = [c + 0.5 for (r, c) in path]
        path_y = [n - 1 - r + 0.5 for (r, c) in path]
        ax.plot(path_x, path_y, "o-", color="#3498DB", linewidth=2.5,
                markersize=7, markeredgecolor="#1F618D", markerfacecolor="#85C1E9",
                zorder=5, alpha=0.85)

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(range(n))
    ax.set_yticklabels(range(n - 1, -1, -1))
    ax.tick_params(length=0)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#2ECC71", edgecolor="#2C3E50", label="Start"),
        mpatches.Patch(facecolor="#F1C40F", edgecolor="#2C3E50", label="Goal"),
        mpatches.Patch(facecolor="#E74C3C", edgecolor="#2C3E50", label="Obstacle"),
    ]
    if path:
        from matplotlib.lines import Line2D
        legend_elements.append(
            Line2D([0], [0], color="#3498DB", marker="o", linewidth=2,
                   markersize=6, label="Path")
        )
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9,
              framealpha=0.9)

    if save_path and show:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [saved] {save_path}")

    if show:
        plt.tight_layout()

    return ax


def plot_grid_comparison(env_original, env_modified,
                         path_original=None, path_modified=None,
                         title="Environment Comparison",
                         save_path=None):
    """
    Side-by-side grid plots showing original vs modified environment.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    plot_grid(env_original, path=path_original,
              title="Original Environment", ax=axes[0])
    plot_grid(env_modified, path=path_modified,
              title="Modified Environment", ax=axes[1])
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [saved] {save_path}")
    return fig


def plot_q_values(q_table, title="Q-Value Heatmap (max per state)",
                  save_path=None):
    """
    Heatmap of max Q-value per state.
    """
    grid_size = q_table.shape[0]
    max_q = np.max(q_table, axis=2)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(max_q, cmap="RdYlGn", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Max Q-value")

    for r in range(grid_size):
        for c in range(grid_size):
            ax.text(c, r, f"{max_q[r, c]:.1f}",
                    ha="center", va="center", fontsize=10,
                    color="black" if abs(max_q[r, c]) < 3 else "white")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [saved] {save_path}")

    return fig
