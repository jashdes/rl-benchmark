import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import torch

from utils.experiment import ExperimentLogger
from utils.config import ExperimentConfig


def run_experiment(config: ExperimentConfig):
    """Run a single RL experiment with proper logging."""

    # Create logger
    logger = ExperimentLogger(experiment_name=f"{config.algorithim}_{config.env_name}")
    logger.log_config(config.to_dict())

    # Set device
    if config.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = config.device
    print(f"Using device: {device}")

    # Create enviornment
    env = gym.make(config.env_name)

    # Create model (only PPO for now)
    if config.algorithim == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            verbose=1,
            tensorboard_log=f"{logger.get_exp_dir()}/tensorboard/",
            device=device,
            seed=config.seed,
        )
    else:
        raise ValueError(f"Algorithm {config.algorithim} not implemented yet")

    # Train
    print(f"\nStarting training: {config.algorithim} not implemented yet")
    print(f"Total timesteps: {config.total_timesteps}")
    model.learn(total_timesteps=config.total_timesteps)

    # Evaluate
    print("\nEvaluating trained policy...")
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=config.n_eval_episodes
    )

    # Log results
    results = {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "config": config.to_dict(),
    }
    logger.log_final_results(results)

    print(f"\nFinal Results:")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Save model
    model_path = f"{logger.get_exp_dir()}/model"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    env.close()
    return results


if __name__ == "__main__":
    # Define experiment configurations
    config = ExperimentConfig(
        algorithim="PPO",
        env_name="CartPole-v1",
        total_timesteps=25000,
        n_eval_episodes=10,
        seed=42,
    )

    # Run experiment
    results = run_experiment(config)
