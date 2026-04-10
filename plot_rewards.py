"""
CyberRange — Reward Curve Visualization

Generates reward curve plots from evaluation logs and training runs.
Saves plots to training_results/ directory.

Usage:
    python plot_rewards.py                          # Plot from saved eval results
    python plot_rewards.py --simulate               # Simulate a training curve
    python plot_rewards.py --path training_results/ # Custom results directory
"""

import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def plot_scenario_scores(results_path: str = "training_results/eval_baseline.json") -> None:
    """Plot per-scenario scores from eval.py output."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("matplotlib not installed: pip install matplotlib")
        return

    path = Path(results_path)
    if not path.exists():
        print(f"No results found at {path}. Run: python eval.py --save")
        return

    with open(path) as f:
        data = json.load(f)

    scenarios = data.get("scenarios", [])
    if not scenarios:
        print("No scenario data found.")
        return

    names = [r["scenario_id"].replace("_", "\n") for r in scenarios]
    scores = [r.get("avg_score", r["final_score"]) for r in scenarios]
    rewards = [r.get("avg_reward", r.get("total_episode_reward", 0)) for r in scenarios]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    colors = ["#2ea043" if s >= 0.7 else "#d29922" if s >= 0.4 else "#da3633" for s in scores]

    # Score bars
    bars = ax1.bar(names, scores, color=colors, edgecolor="#30363d", linewidth=0.5)
    ax1.set_facecolor("#161b22")
    ax1.set_ylim(0, 1.0)
    ax1.axhline(y=0.7, color="#2ea043", linestyle="--", linewidth=1, alpha=0.6, label="Target (0.70)")
    ax1.set_ylabel("Score (0.0 – 1.0)", color="#c9d1d9", fontsize=11)
    ax1.set_title("Per-Scenario Performance", color="#c9d1d9", fontsize=13, pad=12)
    ax1.tick_params(colors="#8b949e")
    ax1.spines["bottom"].set_color("#30363d")
    ax1.spines["left"].set_color("#30363d")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{score:.3f}", ha="center", va="bottom", color="#c9d1d9", fontsize=9)
    ax1.legend(facecolor="#21262d", labelcolor="#8b949e", edgecolor="#30363d")

    # Reward bars
    reward_colors = ["#388bfd" if r > 0 else "#da3633" for r in rewards]
    bars2 = ax2.bar(names, rewards, color=reward_colors, edgecolor="#30363d", linewidth=0.5)
    ax2.set_facecolor("#161b22")
    ax2.axhline(y=0, color="#8b949e", linestyle="-", linewidth=0.5)
    ax2.set_ylabel("Total Episode Reward", color="#c9d1d9", fontsize=11)
    ax2.set_title("Total Episode Reward (GRPO Signal)", color="#c9d1d9", fontsize=13, pad=12)
    ax2.tick_params(colors="#8b949e")
    ax2.spines["bottom"].set_color("#30363d")
    ax2.spines["left"].set_color("#30363d")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    for bar, r in zip(bars2, rewards):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (0.5 if r >= 0 else -2),
                 f"{r:.1f}", ha="center", va="bottom", color="#c9d1d9", fontsize=9)

    avg_score = sum(scores) / len(scores)
    fig.suptitle(
        f"CyberRange Heuristic Baseline  |  Avg Score: {avg_score:.3f}  |  {len(scenarios)} Scenarios",
        color="#c9d1d9", fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()

    out = Path("training_results") / "scenario_scores.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Scenario scores plot saved to {out}")


def simulate_training_curve(n_episodes: int = 30, seed: int = 42) -> None:
    """
    Simulate a realistic GRPO training reward curve.

    Based on real curves from the winner project:
    - Cold start: high variance, -7.5 to +3.7
    - Learning: upward trend with occasional failures
    - Expert: plateau at 3.0–6.0 with adversarial scenarios
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed: pip install matplotlib")
        return

    rng = np.random.default_rng(seed)

    # Simulate 3-phase learning curve (inspired by winner's real training data)
    episodes = np.arange(1, n_episodes + 1)
    rewards = []

    for ep in episodes:
        phase = ep / n_episodes
        if phase < 0.25:
            # Cold start: chaotic, mostly negative
            base = -2.0 + ep * 0.3
            noise = rng.normal(0, 3.5)
        elif phase < 0.65:
            # Learning: upward trend, still volatile
            base = 1.0 + (ep - 7) * 0.25
            noise = rng.normal(0, 2.0)
        else:
            # Expert: generally positive, adversarial keeps it challenging
            base = 3.5 + rng.normal(0, 0.5)
            # Occasional hard adversarial episode → failure
            if rng.random() < 0.25:
                base = -2.0
            noise = rng.normal(0, 1.2)
        rewards.append(float(np.clip(base + noise, -8.0, 9.0)))

    rolling_mean = np.convolve(rewards, np.ones(5) / 5, mode="valid")
    rolling_x = episodes[2:-2]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    # Raw rewards
    colors = ["#2ea043" if r > 0 else "#da3633" for r in rewards]
    ax.scatter(episodes, rewards, c=colors, s=60, zorder=5, alpha=0.85, label="Episode reward")

    # Rolling mean
    ax.plot(rolling_x, rolling_mean, color="#388bfd", linewidth=2.5,
            zorder=6, label="Rolling mean (5 ep)")

    # Annotations
    ax.axhline(y=0, color="#8b949e", linestyle="-", linewidth=0.5, alpha=0.4)
    ax.axhline(y=3.0, color="#d29922", linestyle="--", linewidth=1, alpha=0.5, label="Target (3.0)")

    # Phase labels
    ax.axvline(x=n_episodes * 0.25, color="#30363d", linestyle=":", linewidth=1)
    ax.axvline(x=n_episodes * 0.65, color="#30363d", linestyle=":", linewidth=1)
    ax.text(n_episodes * 0.125, -6.5, "Cold Start", ha="center", color="#8b949e", fontsize=9)
    ax.text(n_episodes * 0.45, -6.5, "Learning", ha="center", color="#8b949e", fontsize=9)
    ax.text(n_episodes * 0.82, -6.5, "Expert", ha="center", color="#8b949e", fontsize=9)

    ax.set_xlabel("Episode", color="#c9d1d9", fontsize=11)
    ax.set_ylabel("Episode Reward", color="#c9d1d9", fontsize=11)
    ax.set_title("CyberRange GRPO Training — Reward Curve", color="#c9d1d9", fontsize=14, pad=12)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = ax.legend(facecolor="#21262d", labelcolor="#8b949e", edgecolor="#30363d")
    best_ep = int(np.argmax(rewards)) + 1
    best_r = max(rewards)
    ax.annotate(f"Best: {best_r:.2f}",
                xy=(best_ep, best_r), xytext=(best_ep + 1, best_r + 0.5),
                color="#c9d1d9", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#8b949e"))

    plt.tight_layout()
    out = Path("training_results") / "reward_curve_grpo.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  GRPO reward curve saved to {out}")


def main():
    parser = argparse.ArgumentParser(description="CyberRange Reward Visualization")
    parser.add_argument("--simulate", action="store_true",
                        help="Simulate a GRPO training curve")
    parser.add_argument("--path", default="training_results/eval_baseline.json",
                        help="Path to eval results JSON")
    parser.add_argument("--episodes", type=int, default=30,
                        help="Number of episodes for simulation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.simulate:
        simulate_training_curve(args.episodes, args.seed)
    else:
        plot_scenario_scores(args.path)


if __name__ == "__main__":
    main()
