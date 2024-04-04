import os
import numpy as np
import pandas as pd
import gymnasium as gym
from collections import deque
from cli_interface import parse_args
from algorithms import DeepQLearning, DQNAgent, DoubleDeepQNetworks
from hyperparameters import Hyperparameters

args = parse_args()

env_name = args.env_name
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
)

if args.model == "dql":
    memory = deque(maxlen=params.memory_size)
    model = DeepQLearning(env, params, memory)
elif args.model == "dqn":
    params.target_update_rate = 0.005
    model = DQNAgent(env, params)
elif args.model == "ddqn":
    params.target_update_rate = 0.005
    model = DoubleDeepQNetworks(env, params)

WINDOW_SIZE = 50

device = model.device
print(f"Training model {model} {args.train_times} times using {device}")

result_df = pd.DataFrame()
for _ in range(args.train_times):
    rewards = model.train()
    curr_df = pd.DataFrame(
        {"episode": range(len(rewards)), "reward": rewards, "model": str(model)}
    )
    curr_df["reward_mean"] = curr_df["reward"].rolling(window=WINDOW_SIZE).mean()
    result_df = pd.concat([result_df, curr_df], ignore_index=True)

path = os.path.join("results", f"{env_name}_{args.model}.csv")
result_df.to_csv(path, index=False)
