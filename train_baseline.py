import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

env = gym.make("CartPole-v1")

model = PPO(
    "MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/", device=device
)

print("Starting training...")
model.learn(total_timesteps=25000)

print("\nEvaluating trained policy...")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

model.save("ppo_cartpole")
print("\nModel saved as 'ppo_cartpole'")

env.close()
