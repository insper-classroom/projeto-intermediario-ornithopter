import os
import numpy as np
import pandas as pd
import gymnasium as gym
from cli_interface import parse_args
from algorithms import DQNAgent, DoubleDeepQNetworks
from hyperparameters import Hyperparameters
import flappy_bird_gymnasium

args = parse_args()

env_name = args.env_name

if env_name == "FlappyBird-v0":
    env = gym.make(env_name, pipe_gap=150, use_lidar=False, normalize_obs=False, score_limit=500)
else:
    env = gym.make(env_name)

np.random.seed(args.seed)

params = Hyperparameters(
    learning_rate=1e-4,
    gamma=0.99,
    epsilon=0.9,
    epsilon_min=0.05,
    epsilon_dec=1000,
    episodes=600,
    batch_size=128,
    memory_size=50_000,
    max_num_steps=1500,
)
# Use the reward threshold from the environment to determine mean reward convergence
mean_rw = env.spec.reward_threshold
is_flappy_bird = False
if env_name == "FlappyBird-v0":
    is_flappy_bird = True
    params.max_num_steps = np.inf
    params.epsilon_dec = 5_000
    mean_rw = 500
    params.episodes = 2_500
    params.epsilon_min = 0.01

print(f"Threshold reward for {env_name} is {mean_rw}")

if args.model == "dqn":
    params.target_update_rate = 0.005
    model = DQNAgent(env, params, mean_rw)
elif args.model == "ddqn":
    params.target_update_rate = 0.005
    model = DoubleDeepQNetworks(env, params, mean_rw)

WINDOW_SIZE = 50
device = model.device
print(f"Training model {model} {args.train_times} times using {device}")

result_df = pd.DataFrame()
res_folder = f"models/{env_name}/"
os.makedirs(res_folder, exist_ok=True)
for i in range(args.train_times):
    print(f"Train {i}")
    rewards = model.train()
    model.save_model(f"{res_folder}{args.model}_{i}.pt")
    curr_df = pd.DataFrame(
        {"episode": range(len(rewards)), "reward": rewards, "model": str(model)}
    )
    curr_df["reward_mean"] = curr_df["reward"].rolling(window=WINDOW_SIZE).mean()
    result_df = pd.concat([result_df, curr_df], ignore_index=True)
    path = os.path.join("results", f"{env_name}_{args.model}_{i}.csv")
    result_df.to_csv(path, index=False)

    if args.model == "dqn":
        model = DQNAgent(env, params, mean_rw)
    else:
        model = DoubleDeepQNetworks(env, params, mean_rw)

path = os.path.join("results", f"{env_name}_{args.model}.csv")
result_df.to_csv(path, index=False)
