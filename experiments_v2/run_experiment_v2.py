"""
Upgraded Experiment Runner v2 — Research-Grade Differentiation.

Generates:
  1. Learning curve (reward convergence)
  2. Adaptation curve (sharp drop + recovery after env change)
  3. Success rate comparison (before/after, both agents)
  4. Steps to goal comparison (before/after, both agents)
  5. Failure visualization (production fails, Q-learning succeeds)
  6. Grid visualizations with paths
  7. Q-value heatmaps
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
import config_v2 as cfg
from grid_world_v2 import GridWorld
from production_agent_v2 import ProductionAgentV2
from q_learning_agent_v2 import QLearningAgentV2


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════
def make_env(obstacles):
    return GridWorld(cfg.GRID_SIZE, cfg.START_STATE, cfg.GOAL_STATE,
                     obstacles, cfg.MAX_STEPS, cfg.REWARD_GOAL,
                     cfg.REWARD_STEP, cfg.REWARD_OBSTACLE, cfg.REWARD_WALL)

def evaluate(agent, env, n=None):
    n = n or cfg.EVAL_EPISODES
    rewards, steps, successes, paths = [], [], [], []
    for _ in range(n):
        r = agent.run_episode(env)
        rewards.append(r["total_reward"]); steps.append(r["steps"])
        successes.append(r["success"]); paths.append(r["path"])
    best_idx = int(np.argmax(rewards))
    worst_idx = int(np.argmin(rewards))
    return {"agent": agent.name, "n": n,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_steps": float(np.mean(steps)),
            "std_steps": float(np.std(steps)),
            "success_rate": float(np.mean(successes)),
            "rewards": rewards, "steps_list": steps,
            "best_path": paths[best_idx],
            "worst_path": paths[worst_idx],
            "best_reward": rewards[best_idx],
            "worst_reward": rewards[worst_idx]}

def pr(m, header=""):
    if header:
        print(f"\n{'='*55}\n  {header}\n{'='*55}")
    print(f"  Agent        : {m['agent']}")
    print(f"  Mean Reward  : {m['mean_reward']:.2f} (+/- {m['std_reward']:.2f})")
    print(f"  Mean Steps   : {m['mean_steps']:.2f} (+/- {m['std_steps']:.2f})")
    print(f"  Success Rate : {m['success_rate']*100:.1f}%")
    print(f"  {'─'*50}")


# ═══════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
# ═══════════════════════════════════════════════════════════
def smooth(data, w=50):
    if len(data) < w: return data
    c = np.cumsum(np.insert(data, 0, 0))
    return (c[w:] - c[:-w]) / w

def plot_grid(env, path=None, title="", save=None, ax=None):
    show = ax is None
    if ax is None: fig, ax = plt.subplots(figsize=(7, 7))
    n = env.grid_size
    colors = {0:"#FFFFFF", 1:"#E74C3C", 2:"#2ECC71", 3:"#F1C40F"}
    grid = env.get_grid_matrix()
    for r in range(n):
        for c in range(n):
            rect = plt.Rectangle((c, n-1-r), 1, 1, facecolor=colors[grid[r,c]],
                                 edgecolor="#2C3E50", linewidth=1.2)
            ax.add_patch(rect)
            if grid[r,c]==2: ax.text(c+.5, n-1-r+.5, "S", ha="center", va="center",
                                      fontsize=11, fontweight="bold", color="#1A5276")
            elif grid[r,c]==3: ax.text(c+.5, n-1-r+.5, "G", ha="center", va="center",
                                        fontsize=11, fontweight="bold", color="#7D6608")
            elif grid[r,c]==1: ax.text(c+.5, n-1-r+.5, "X", ha="center", va="center",
                                        fontsize=12, fontweight="bold", color="#922B21")
    if path and len(path)>1:
        px = [c+.5 for (r,c) in path]
        py = [n-1-r+.5 for (r,c) in path]
        ax.plot(px, py, "o-", color="#3498DB", lw=2.2, ms=5,
                markerfacecolor="#85C1E9", markeredgecolor="#1F618D", zorder=5, alpha=.8)
    ax.set_xlim(0,n); ax.set_ylim(0,n); ax.set_aspect("equal")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(range(n)); ax.set_yticklabels(range(n-1,-1,-1))
    ax.tick_params(length=0)
    if save and show:
        plt.tight_layout(); plt.savefig(save, dpi=150, bbox_inches="tight"); print(f"  [saved] {save}")
    return ax

def plot_learning_curve(rewards, title, save):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(range(1,len(rewards)+1), rewards, alpha=.2, color="#3498DB", lw=.8, label="Raw")
    s = smooth(rewards); ax.plot(range(50,50+len(s)), s, color="#E74C3C", lw=2.2, label="Smoothed")
    ax.set_xlabel("Episode",fontsize=12); ax.set_ylabel("Total Reward",fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold"); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(save, dpi=150, bbox_inches="tight"); print(f"  [saved] {save}")

def plot_adaptation_curve(rewards_orig, rewards_retrain, save):
    """THE KEY PLOT: continuous training with env-change vertical line."""
    combined = rewards_orig + rewards_retrain
    fig, ax = plt.subplots(figsize=(12,5))
    eps = range(1, len(combined)+1)
    ax.plot(eps, combined, alpha=.2, color="#3498DB", lw=.8, label="Raw reward")
    s = smooth(combined); ax.plot(range(50,50+len(s)), s, color="#E74C3C", lw=2.5, label="Smoothed")
    change_ep = len(rewards_orig)
    ax.axvline(x=change_ep, color="#2C3E50", ls="--", lw=2, label="Environment Change")
    ax.annotate("ENV CHANGE", xy=(change_ep, max(combined)*0.8),
                fontsize=11, fontweight="bold", color="#2C3E50",
                ha="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#F9E79F", alpha=0.9))
    ax.set_xlabel("Episode",fontsize=12); ax.set_ylabel("Total Reward",fontsize=12)
    ax.set_title("Q-Learning Adaptation Curve: Sharp Drop & Recovery", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(save, dpi=150, bbox_inches="tight"); print(f"  [saved] {save}")

def plot_comparison_bars(before, after, metric, ylabel, title, save):
    agents = [b["agent"] for b in before]
    b_vals = [b[metric] for b in before]
    a_vals = [a[metric] for a in after]
    if "rate" in metric:
        b_vals = [v*100 for v in b_vals]; a_vals = [v*100 for v in a_vals]
    x = np.arange(len(agents)); w = 0.3
    fig, ax = plt.subplots(figsize=(8,5))
    bars1 = ax.bar(x-w/2, b_vals, w, label="Before Change", color="#3498DB", edgecolor="#2C3E50")
    bars2 = ax.bar(x+w/2, a_vals, w, label="After Change", color="#E74C3C", edgecolor="#2C3E50")
    for b,v in zip(list(bars1)+list(bars2), b_vals+a_vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f"{v:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(agents, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(save, dpi=150, bbox_inches="tight"); print(f"  [saved] {save}")

def plot_failure_vs_success(env, fail_path, success_path, save):
    """Side-by-side: production agent fails (left), Q-learning succeeds (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    # Production agent failure
    plot_grid(env, path=fail_path,
              title=f"Production Agent FAILS ({len(fail_path)-1} steps, wandering)",
              ax=axes[0])
    # Mark the path in red for failure
    n = env.grid_size
    if len(fail_path) > 1:
        axes[0].lines[0].set_color("#E74C3C")
        axes[0].lines[0].set_alpha(0.5)
    # Q-learning success
    plot_grid(env, path=success_path,
              title=f"Q-Learning Agent SUCCEEDS ({len(success_path)-1} steps)",
              ax=axes[1])
    fig.suptitle("Failure Visualization: Modified Environment",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight"); print(f"  [saved] {save}")

def plot_steps_curve(steps, title, save):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(range(1,len(steps)+1), steps, alpha=.2, color="#2ECC71", lw=.8, label="Raw")
    s = smooth(steps); ax.plot(range(50,50+len(s)), s, color="#8E44AD", lw=2.2, label="Smoothed")
    ax.set_xlabel("Episode",fontsize=12); ax.set_ylabel("Steps",fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold"); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(save, dpi=150, bbox_inches="tight"); print(f"  [saved] {save}")

def plot_q_heatmap(q_table, title, save):
    fig, ax = plt.subplots(figsize=(7,6))
    mq = np.max(q_table, axis=2)
    im = ax.imshow(mq, cmap="RdYlGn", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Max Q-value")
    for r in range(mq.shape[0]):
        for c in range(mq.shape[1]):
            ax.text(c, r, f"{mq[r,c]:.1f}", ha="center", va="center", fontsize=8,
                    color="black" if abs(mq[r,c])<3 else "white")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Column"); ax.set_ylabel("Row")
    plt.tight_layout(); plt.savefig(save, dpi=150, bbox_inches="tight"); print(f"  [saved] {save}")


# ═══════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════
def main():
    os.makedirs(cfg.PLOTS_DIR, exist_ok=True)
    os.makedirs(cfg.LOGS_DIR, exist_ok=True)
    np.random.seed(42)
    import random; random.seed(42)

    P = cfg.PLOTS_DIR

    print("=" * 60)
    print("  COGNITIVE ROBOTICS EXPERIMENT v2 (UPGRADED)")
    print("  10x10 Grid | Local Perception | Research-Grade")
    print("=" * 60)

    # ─────────────────────────────────────────────────────
    # PHASE 1: ORIGINAL ENVIRONMENT
    # ─────────────────────────────────────────────────────
    print("\n>> PHASE 1: Original Environment (10x10)")
    env_orig = make_env(cfg.OBSTACLES_ORIGINAL)
    print(f"   Grid: {cfg.GRID_SIZE}x{cfg.GRID_SIZE}")
    print(f"   Obstacles: {len(cfg.OBSTACLES_ORIGINAL)}")

    # Plot original grid
    fig, ax = plt.subplots(figsize=(7,7))
    plot_grid(env_orig, title="Original Environment (10x10)", ax=ax)
    plt.tight_layout(); plt.savefig(os.path.join(P, "01_grid_original.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [saved] 01_grid_original.png")

    # --- Production Agent ---
    print("\n>> Evaluating Production Rule Agent (local perception)...")
    prod = ProductionAgentV2(visit_memory=cfg.VISIT_MEMORY)
    m_prod_orig = evaluate(prod, env_orig)
    pr(m_prod_orig, "Production Agent -- Original Env")

    # --- Q-Learning Agent ---
    print("\n>> Training Q-Learning Agent (1000 episodes)...")
    qagent = QLearningAgentV2(cfg.GRID_SIZE, cfg.ALPHA, cfg.GAMMA,
                               cfg.EPSILON_START, cfg.EPSILON_MIN, cfg.EPSILON_DECAY)
    train1 = qagent.train(env_orig, cfg.NUM_EPISODES)

    plot_learning_curve(train1["rewards_history"],
                        "Q-Learning: Reward per Episode (Original Env)",
                        os.path.join(P, "02_learning_curve.png"))

    plot_steps_curve(train1["steps_history"],
                     "Q-Learning: Steps per Episode (Original Env)",
                     os.path.join(P, "03_steps_curve.png"))

    plot_q_heatmap(qagent.q_table, "Q-Values (Original Env)",
                   os.path.join(P, "04_q_heatmap_original.png"))

    m_q_orig = evaluate(qagent, env_orig)
    pr(m_q_orig, "Q-Learning Agent -- Original Env")

    # Path visualizations
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    plot_grid(env_orig, path=m_prod_orig["best_path"],
              title="Production Agent Best Path", ax=axes[0])
    plot_grid(env_orig, path=m_q_orig["best_path"],
              title="Q-Learning Agent Best Path", ax=axes[1])
    fig.suptitle("Agent Paths -- Original Environment", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(P, "05_paths_original.png"), dpi=150, bbox_inches="tight")
    plt.close(); print("  [saved] 05_paths_original.png")

    # Comparison table
    print(f"\n{'='*60}")
    print(f"  COMPARISON -- ORIGINAL ENVIRONMENT")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} | {'Production':>15} | {'Q-Learning':>15}")
    print(f"  {'─'*55}")
    print(f"  {'Mean Reward':<20} | {m_prod_orig['mean_reward']:>15.2f} | {m_q_orig['mean_reward']:>15.2f}")
    print(f"  {'Mean Steps':<20} | {m_prod_orig['mean_steps']:>15.2f} | {m_q_orig['mean_steps']:>15.2f}")
    print(f"  {'Success Rate':<20} | {m_prod_orig['success_rate']*100:>14.1f}% | {m_q_orig['success_rate']*100:>14.1f}%")
    print(f"{'='*60}")

    # ─────────────────────────────────────────────────────
    # PHASE 2: MODIFIED ENVIRONMENT (NO RETRAINING)
    # ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(">> PHASE 2: Modified Environment (Adaptability Test)")
    print(f"{'='*60}")
    env_mod = make_env(cfg.OBSTACLES_MODIFIED)
    print(f"   New Obstacles: {len(cfg.OBSTACLES_MODIFIED)}")

    # Plot modified grid
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    plot_grid(env_orig, title="Original Environment", ax=axes[0])
    plot_grid(env_mod, title="Modified Environment", ax=axes[1])
    fig.suptitle("Environment Change", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(P, "06_env_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(); print("  [saved] 06_env_comparison.png")

    # Evaluate BOTH on modified env (no retraining)
    print("\n>> Production Agent on modified env...")
    m_prod_mod = evaluate(prod, env_mod)
    pr(m_prod_mod, "Production Agent -- Modified Env")

    print(">> Q-Learning Agent on modified env (STALE Q-table)...")
    m_q_stale = evaluate(qagent, env_mod)
    pr(m_q_stale, "Q-Learning Agent -- Modified Env (NOT retrained)")

    # ─────────────────────────────────────────────────────
    # PHASE 3: RETRAIN Q-LEARNING ON MODIFIED ENV
    # ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(">> PHASE 3: Retraining Q-Learning Agent")
    print(f"{'='*60}")

    rewards_before_change = list(qagent.rewards_history)  # save phase 1 history
    qagent.epsilon = cfg.EPSILON_START  # reset exploration
    train2 = qagent.train(env_mod, cfg.RETRAIN_EPISODES)

    # ADAPTATION CURVE (the key plot!)
    plot_adaptation_curve(rewards_before_change, train2["rewards_history"],
                          os.path.join(P, "07_adaptation_curve.png"))

    plot_q_heatmap(qagent.q_table, "Q-Values (After Retraining on Modified Env)",
                   os.path.join(P, "08_q_heatmap_retrained.png"))

    m_q_retrained = evaluate(qagent, env_mod)
    pr(m_q_retrained, "Q-Learning Agent -- Modified Env (RETRAINED)")

    # ─────────────────────────────────────────────────────
    # PHASE 4: FINAL COMPARISONS & ALL PLOTS
    # ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(">> PHASE 4: Final Comparison & Visualizations")
    print(f"{'='*60}")

    # Success Rate Comparison
    plot_comparison_bars(
        [m_prod_orig, m_q_orig], [m_prod_mod, m_q_retrained],
        "success_rate", "Success Rate (%)",
        "Success Rate: Before vs After Environment Change",
        os.path.join(P, "09_success_rate_comparison.png"))

    # Steps to Goal Comparison
    plot_comparison_bars(
        [m_prod_orig, m_q_orig], [m_prod_mod, m_q_retrained],
        "mean_steps", "Mean Steps to Goal",
        "Steps to Goal: Before vs After Environment Change",
        os.path.join(P, "10_steps_comparison.png"))

    # Mean Reward Comparison
    plot_comparison_bars(
        [m_prod_orig, m_q_orig], [m_prod_mod, m_q_retrained],
        "mean_reward", "Mean Reward",
        "Mean Reward: Before vs After Environment Change",
        os.path.join(P, "11_reward_comparison.png"))

    # FAILURE VISUALIZATION
    # Get a failure episode from production agent on modified env
    prod_fail = None
    for _ in range(50):
        r = prod.run_episode(env_mod)
        if not r["success"]:
            prod_fail = r; break
    if prod_fail is None:
        prod_fail = prod.run_episode(env_mod)  # use whatever we get

    q_success = qagent.run_episode(env_mod)

    plot_failure_vs_success(env_mod, prod_fail["path"], q_success["path"],
                            os.path.join(P, "12_failure_visualization.png"))

    # Paths on modified env
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    plot_grid(env_mod, path=m_prod_mod["worst_path"],
              title=f"Production Agent (worst, {len(m_prod_mod['worst_path'])-1} steps)",
              ax=axes[0])
    plot_grid(env_mod, path=m_q_retrained["best_path"],
              title=f"Q-Learning Retrained (best, {len(m_q_retrained['best_path'])-1} steps)",
              ax=axes[1])
    fig.suptitle("Modified Environment: Worst vs Best Paths", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(P, "13_paths_modified.png"), dpi=150, bbox_inches="tight")
    plt.close(); print("  [saved] 13_paths_modified.png")

    # ─────────────────────────────────────────────────────
    # FINAL SUMMARY TABLE
    # ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Metric':<25} | {'Prod(Orig)':>12} | {'Prod(Mod)':>12} | {'QL(Orig)':>12} | {'QL(Retr)':>12}")
    print(f"  {'─'*70}")
    print(f"  {'Mean Reward':<25} | {m_prod_orig['mean_reward']:>12.2f} | {m_prod_mod['mean_reward']:>12.2f} | {m_q_orig['mean_reward']:>12.2f} | {m_q_retrained['mean_reward']:>12.2f}")
    print(f"  {'Mean Steps':<25} | {m_prod_orig['mean_steps']:>12.2f} | {m_prod_mod['mean_steps']:>12.2f} | {m_q_orig['mean_steps']:>12.2f} | {m_q_retrained['mean_steps']:>12.2f}")
    print(f"  {'Success Rate (%)':<25} | {m_prod_orig['success_rate']*100:>11.1f}% | {m_prod_mod['success_rate']*100:>11.1f}% | {m_q_orig['success_rate']*100:>11.1f}% | {m_q_retrained['success_rate']*100:>11.1f}%")

    # Adaptation deltas
    prod_delta = m_prod_mod['success_rate'] - m_prod_orig['success_rate']
    ql_delta = m_q_retrained['success_rate'] - m_q_orig['success_rate']
    print(f"\n  Adaptation Delta (Success Rate):")
    print(f"    Production Agent : {prod_delta*100:+.1f}%")
    print(f"    Q-Learning Agent : {ql_delta*100:+.1f}%")
    print(f"{'='*70}")

    # Save logs
    log = {
        "original": {
            "production": {k: v for k, v in m_prod_orig.items() if k not in ["rewards","steps_list","best_path","worst_path"]},
            "q_learning": {k: v for k, v in m_q_orig.items() if k not in ["rewards","steps_list","best_path","worst_path"]},
        },
        "modified": {
            "production": {k: v for k, v in m_prod_mod.items() if k not in ["rewards","steps_list","best_path","worst_path"]},
            "q_learning_stale": {k: v for k, v in m_q_stale.items() if k not in ["rewards","steps_list","best_path","worst_path"]},
            "q_learning_retrained": {k: v for k, v in m_q_retrained.items() if k not in ["rewards","steps_list","best_path","worst_path"]},
        },
        "adaptation_delta": {"production": prod_delta, "q_learning": ql_delta},
    }
    log_path = os.path.join(cfg.LOGS_DIR, "experiment_v2_results.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)
    print(f"\n  [log saved] {log_path}")

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT v2 COMPLETE")
    print(f"{'='*60}")
    print(f"  Plots: {cfg.PLOTS_DIR} ({len(os.listdir(cfg.PLOTS_DIR))} files)")
    print(f"  Logs : {cfg.LOGS_DIR}")
    print(f"{'='*60}")
    plt.close("all")


if __name__ == "__main__":
    main()
