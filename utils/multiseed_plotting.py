import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from collections import defaultdict


def load_multiseed_results(results_dir):
    """Load and group results byh algorithm and seed."""
    results_dir = Path(results_dir)
    algo_seed_results = defaultdict(lambda: defaultdict(list))

    for exp_dir in results_dir.iterdir():
        if exp_dir.is_dir():
            results_file = exp_dir / "results.json"
            if results_file.exists():
                try:
                    with open(results_file) as f:
                        data = json.load(f)
                        if not isinstance(data, dict):
                            continue
                        if "mean_reward" not in data or "config" not in data:
                            continue

                        algo = data["config"].get("algorithm") or data["config"].get(
                            "algorithim"
                        )
                        seed = data["config"].get("seed", 0)

                        algo_seed_results[algo][seed].append(data["mean_reward"])
                except:
                    continue

    return algo_seed_results


def plot_multiseed_comparison(results_dir, save_path="multiseed_comparison.png"):
    """Create bar chart with error bars from multi-seed results."""
    algo_seed_results = load_multiseed_results(results_dir)

    if not algo_seed_results:
        print("No results found!")
        return

    # Compute statistics for each algorithm
    algo_stats = {}
    for algo, seed_results in algo_seed_results.items():
        # Get all rewards across all seeds
        all_rewards = []
        for seed, rewards in seed_results.items():
            all_rewards.extend(rewards)

        algo_stats[algo] = {
            "mean": np.mean(all_rewards),
            "std": np.std(all_rewards),
            "n": len(all_rewards),
        }

    # Create plot
    algorithms = list(algo_stats.keys())
    means = [algo_stats[algo]["mean"] for algo in algorithms]
    stds = [algo_stats[algo]["std"] for algo in algorithms]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(algorithms))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=10,
        alpha=0.7,
        color="steelblue",
        ecolor="red",
        linewidth=2,
    )
    ax.set_xlabel("Algorithm", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Reward", fontsize=12, fontweight="bold")
    ax.set_title(
        "Algorithm Performance Comparison (Multiseed)", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels with n
    for i, (algo, mean, std) in enumerate(zip(algorithms, means, stds)):
        n = algo_stats[algo]["n"]
        label = f"{mean:.1f} +/- {std:.1f}\n(n={n})"
        ax.text(i, mean + std, label, ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Multiseed comparison plot saved to{save_path}")
    plt.close()


def plot_seed_distribution(results_dir, save_path="seed_distribution.png"):
    """Create box plot showing distribution across seeds."""
    algo_seed_results = load_multiseed_results(results_dir)

    if not algo_seed_results:
        print("No results found!")
        return

    # Prepare data for box plot
    data_to_plot = []
    labels = []

    for algo, seed_results in algo_seed_results.items():
        all_rewards = []
        for seed, rewards in seed_results.items():
            all_rewards.extend(rewards)
        data_to_plot.append(all_rewards)
        labels.append(algo)

    # Create box plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(
        data_to_plot, labels=labels, patch_artist=True, notch=True, showmeans=True
    )

    # Custom colors
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)

    ax.set_xlabel("Algorithm", fontsize=12, fontweight="bold")
    ax.set_ylabel("Reward", fontsize=12, fontweight="bold")
    ax.set_title("Reward Distribution Across Seeds", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Distribution plot saved to {save_path}")
    plt.close()


def create_multiseed_summary(results_dir, save_path="multiseed_summary.csv"):
    """Create detailed summary table of multi-seed results."""
    algo_seed_results = load_multiseed_results(results_dir)

    if not algo_seed_results:
        print("No results found!")
        return

    rows = []
    for algo, seed_results in algo_seed_results.items():
        all_rewards = []
        for seed, rewards in seed_results.items():
            all_rewards.extend(rewards)

        rows.append(
            {
                "Algorithm": algo,
                "Mean": np.mean(all_rewards),
                "Std": np.std(all_rewards),
                "Min": np.min(all_rewards),
                "Max": np.max(all_rewards),
                "N_runs": len(all_rewards),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("Mean", ascending=False)

    df.to_csv(save_path, index=False, float_format="%.2f")
    print(f"Multiseed summary saved to {save_path}")
    print("\nSummary:")
    print(df.to_string(index=False))

    return df
