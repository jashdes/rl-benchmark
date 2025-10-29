import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd


def load_results(results_dir):
    """Load all experiment results from a directory."""
    results_dir = Path(results_dir)
    all_results = []

    for exp_dir in results_dir.iterdir():
        if exp_dir.is_dir():
            results_file = exp_dir / "results.json"
            if results_file.exists():
                try:
                    with open(results_file) as f:
                        data = json.load(f)
                        # Skip if data is not a dict
                        if not isinstance(data, dict):
                            print(f"Warning: Skipping {exp_dir.name} - old format")
                            continue
                        # Skip if missing required fields
                        if "mean_reward" not in data or "config" not in data:
                            print(f"Warning: Skipping {exp_dir.name} - missing fields")
                            continue
                        data["experiment_name"] = exp_dir.name
                        print(f"✓ Loaded: {exp_dir.name}")
                        all_results.append(data)
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Skipping {exp_dir.name} - invalud results file")
                    continue
    print(f"\nTotal results loaded: {len(all_results)}")
    return all_results


def plot_algorithm_comparison(results_dir, save_path="algorithm_comparison.png"):
    """Create bar chart comparing algorithm performance."""
    results = load_results(results_dir)

    print(f"DEBUG plot: Loaded {len(results)} results")

    if not results:
        print("No results found!")
        return

    # Group by algorithm
    algo_results = {}
    for r in results:
        algo = r["config"].get("algorithm") or r["config"].get("algorithim")
        if not algo:
            print(
                f"Warning: No alogrithm found in {r.get('experiment_name', 'unknown')}"
            )
            continue
        if algo not in algo_results:
            algo_results[algo] = []
        algo_results[algo].append(r["mean_reward"])

    print(f"DEBUG plot: algo_results = {algo_results}")

    # Calculate means and stds
    algorithms = list(algo_results.keys())
    means = [np.mean(algo_results[algo]) for algo in algorithms]
    stds = [
        np.std(algo_results[algo]) if len(algo_results[algo]) > 1 else 0
        for algo in algorithms
    ]

    print(f"DEBUG plot: algorithms = {algorithms}")  # DEBUG
    print(f"DEBUG plot: means = {means}")

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(algorithms))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color="steelblue")

    ax.set_xlabel("Algorithm", fontsize=12)
    ax.set_ylabel("Mean Reward", fontsize=12)
    ax.set_title("Algotithm Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars.patches, means)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    print(f"DEBUG plot: About to save to {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Comparison plot saved to {save_path}")
    plt.close()


def create_results_table(results_dir, save_path="results_table.csv"):
    """Create a summary table of all experiments."""
    results = load_results(results_dir)

    if not results:
        print("No results found!")
        return

    # Extract key info
    data = []
    for r in results:
        print(f"DEBUG: Entering loop for {r.get('experiment_name', 'unknown')}")
        try:
            # Handle both spellings
            algo = r["config"].get("algorithm") or r["config"].get("algoorithim")
            print(f"DEBUG: Processing {algo} - {r['mean_reward']}")
            data.append(
                {
                    "Algorithm": algo,
                    "Environment": r["config"]["env_name"],
                    "Timesteps": r["config"]["total_timesteps"],
                    "Mean Reward": r["mean_reward"],
                    "Std Reward": r["std_reward"],
                    "Learning Rate": r["config"]["learning_rate"],
                    "Seed": r["config"]["seed"],
                }
            )
        except Exception as e:
            print(f"ERROR processing {r.get('experiment_name', 'unknown')}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(f"DEBUG: Total rows in data: {len(data)}")
    df = pd.DataFrame(data)
    print(f"DEBUG: DataFrame shape: {df.shape}")
    print(f"DEBUG: DataFrame before sort:\n{df}")

    df = df.sort_values(["Algorithm", "Mean Reward"], ascending=[True, False])

    # Save
    df.to_csv(save_path, index=False)
    print(f"Results table saved to {save_path}")
    print("\nResults Summary:")
    print(df.to_string(index=False))

    return df


def plot_learning_curves(results_dir, save_path="learning_currvs.png"):
    """Plot training curves for all algorithms (if tensorboard data available)."""
    print("Learning curve plorring from tensorboard logs - TODO")
    pass
