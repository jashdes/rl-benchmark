from utils.trainer import RLTrainer
from utils.config import ExperimentConfig


if __name__ == "__main__":
    # Define experiment configuration
    config = ExperimentConfig(
        algorithim="PPO",
        env_name="CartPole-v1",
        total_timesteps=25000,
        n_eval_episodes=10,
        seed=42,
    )

    # Create trainer and run experiment
    trainer = RLTrainer(config)
    results = trainer.run()
