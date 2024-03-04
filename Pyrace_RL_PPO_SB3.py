import gymnasium as gym
import gym_race
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
import torch
import torch.nn as nn

# Define a custom neural network model
class CustomPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomPolicy, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

env = gym.make("Pyrace-v1")

# Wrap the environment in a vectorized environment

# Define the PPO agent with the custom policy
policy_kwargs = dict(
    net_arch=[64, 64],
    activation_fn=torch.nn.ReLU,
    ortho_init=False
)
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

# Train the agent
total_timesteps = 1000000  # You can adjust this number
render_interval = 5000  # Set the interval to render the environment (adjust as needed)
for t in range(total_timesteps):
    model.learn(1)
    if t % render_interval == 0:
        obs = env.reset()
        for _ in range(1000):
            action, _ = model.predict(obs)
            obs, _, done, _ = env.step(action)
            env.render()

# Save the trained model
model.save("ppo_pyrace")

# Evaluate the trained model
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}")

# You can use the model for inference later
# loaded_model = PPO.load("ppo_pyrace")
# obs = env.reset()
# for _ in range(1000):
#     action, _ = loaded_model.predict(obs)
#     obs, _, _, _ = env.step(action)

# Close the environment
env.close()
