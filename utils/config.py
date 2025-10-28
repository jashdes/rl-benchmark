from dataclasses import dataclass, asdict


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    algorithim: str
    env_name: str
    total_timesteps: int
    n_eval_episodes: int
    seed: int = 42
    device: str = "auto"

    # Algorithm-speicfic hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 2048  # For PPO
    batch_size: int = 64

    def to_dict(self):
        return asdict(self)
