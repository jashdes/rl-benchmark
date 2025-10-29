from utils.trainer import RLTrainer
from utils.config import ExperimentConfig


def compare_algorithms():
    """Compare PPO, DQN, and SAC on CartPole."""

    algorithms = ["PPO", "DQN"]
    results = {}

    for algo in algorithms:
        print(f"\n{'='*50}")
        print(f"Running {algo}")
        print(f"{'='*50}")

        # Configure based on algo
        if algo == "DQN":
            config = ExperimentConfig(
                algorithm=algo,
                env_name="CartPole-v1",
                total_timesteps=100000,
                n_eval_episodes=10,
                seed=42,
                learning_rate=1e-3,
            )
        else:
            config = ExperimentConfig(
                algorithm=algo,
                env_name="CartPole-v1",
                total_timesteps=25000,
                n_eval_episodes=10,
                seed=42,
            )

        trainer = RLTrainer(config)
        results[algo] = trainer.run()

    # Print comparison
    print(f"\n{'='*50}")
    print("FINAL COMPARISON")
    print(f"{'='*50}")
    for algo, result in results.items():
        print(f"{algo}: {result['mean_reward']:.2f} +/- {result['std_reward']:.2f}")


if __name__ == "__main__":
    compare_algorithms()
