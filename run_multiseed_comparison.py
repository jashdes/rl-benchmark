from utils.trainer import RLTrainer
from utils.config import ExperimentConfig
import numpy as np


def run_multiseed_experiment(algorithm, env_name, n_seeds=5, **kwargs):
    """Run experiment with multiple random seeds."""
    print(f"\n{'='*60}")
    print(f"Running {algorithm} with {n_seeds} seeds")
    print(f"{'='*60}")

    results = []

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed + 1}/{n_seeds} ---")

        config = ExperimentConfig(
            algorithm=algorithm, env_name=env_name, seed=seed, **kwargs
        )

        trainer = RLTrainer(config)
        result = trainer.run()
        results.append(result["mean_reward"])

        print(f"Seed {seed}: {result['mean_reward']:.2f}")

    # Compute statistics
    mean = np.mean(results)
    std = np.std(results)

    print(f"\n{algorithm} Summary {n_seeds} seeds:")
    print(f"Mean: {mean:.2f} +/- {std:.2f}")
    print(f"Min: {np.min(results):.2f}")
    print(f"Max: {np.max(results):.2f}")

    return {"algorithm": algorithm, "results": results, "mean": mean, "std": std}


def compare_algorithms_multiseed(n_seeds=5):
    """Compare algorithms with multiple seeds."""

    algorithms = [
        ("PPO", {"total_timesteps": 25000, "n_eval_episodes": 10}),
        (
            "DQN",
            {"total_timesteps": 100000, "n_eval_episodes": 10, "learning_rate": 1e-3},
        ),
    ]

    all_results = {}

    for algo, params in algorithms:
        result = run_multiseed_experiment(
            algorithm=algo, env_name="CartPole-v1", n_seeds=n_seeds, **params
        )
        all_results[algo] = result

    # Print final comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    for algo, data in all_results.items():
        print(f"{algo}: {data['mean']:.2f} +/- {data['std']:.2f}")

    return all_results


if __name__ == "__main__":
    # Run with 5 seeds each
    results = compare_algorithms_multiseed(n_seeds=5)
