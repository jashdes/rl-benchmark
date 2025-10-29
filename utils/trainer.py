import gymnasium as gym
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.evaluation import evaluate_policy
import torch

from utils.experiment import ExperimentLogger
from utils.config import ExperimentConfig


class RLTrainer:
    """Unified trainer for RL algorithms"""

    ALGORITHMS = {
        "PPO": PPO,
        "DGN": DQN,
        "SAC": SAC,
    }

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = ExperimentLogger(
            experiment_name=f"{config.algorithim}_{config.env_name}"
        )
        self.logger.log_config(config.to_dict())

        # Set device
        if config.device == "auto":
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = config.device
        print(f"Using device: {self.device}")

        # Create enviornment
        self.env = gym.make(config.env_name)

        # Create model
        self.model = self._create_model()

    def _create_model(self):
        """Initialize the RL algorithm."""
        if self.config.algorithim not in self.ALGORITHMS:
            raise ValueError(
                f"Algorithm {self.config.algorithim} not supported."
                f"Choose from: {list(self.ALGORITHMS.keys())}"
            )

        AlgoClass = self.ALGORITHMS[self.config.algorithim]

        model = AlgoClass(
            "MlpPolicy",
            self.env,
            learning_rate=self.config.learning_rate,
            verbose=1,
            tensorboard_log=f"{self.logger.get_exp_dir()}/tensorboard/",
            device=self.device,
            seed=self.config.seed,
        )

        return model

    def train(self):
        """Train the model."""
        print(
            f"n\Strating training: {self.config.algorithim} on {self.config.env_name}"
        )
        print(f"Total timesteps: {self.config.total_timesteps}")

        self.model.learn(total_timesteps=self.config.total_timesteps)

        print("Training complete.")

    def evaluate(self):
        """Evaluate the trained model."""
        print("\nEvaluating trained policy...")
        mean_reward, std_reward = evaluate_policy(
            self.model, self.env, n_eval_episodes=self.config.n_eval_episodes
        )

        results = {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "config": self.config.to_dict(),
        }

        self.logger.log_final_results(results)

        print(f"\nFinal Results:")
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        return results

    def save_model(self):
        """Save the trained model."""
        model_path = f"{self.logger.get_exp_dir()}/model"
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

    def run(self):
        """Run complete training pipeline."""
        self.train()
        results = self.evaluate()
        self.save_model()
        self.env.close()
        return results
