import torch
from .network import *
import torch.optim as optim
import torch.nn as nn
from .replay_memory import ReplayMemory, Transition
import random
import math
import numpy as np


class DoubleDeepQNetworks:
    def __init__(self, env, params, solved_mean_rw) -> None:
        self.env = env
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        n_actions = env.action_space.n
        n_observations = env.observation_space.shape[0]
        lr = params.learning_rate
        self.tau = params.target_update_rate
        self.gamma = params.gamma
        self.batch_size = params.batch_size
        self.episodes = params.episodes
        self.epsilon_start = params.epsilon
        self.epsilon = params.epsilon
        self.epsilon_min = params.epsilon_min
        self.epsilon_dec = params.epsilon_dec
        self.max_num_steps = params.max_num_steps
        self.solved_mean_rw = solved_mean_rw

        policy_net = DQN(n_observations, n_actions).to(device)
        target_net = DQN(n_observations, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        self.policy_net = policy_net
        self.target_net = target_net

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory(params.memory_size)
        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()

        eps_threshold = self.epsilon_min + (
            self.epsilon_start - self.epsilon_min
        ) * math.exp(-1.0 * self.steps_done / self.epsilon_dec)
        self.epsilon = eps_threshold
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor(
                [[self.env.action_space.sample()]], device=self.device, dtype=torch.long
            )

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            best_policy_actions = self.policy_net(non_final_next_states).max(1).indices
            target_values = self.target_net(non_final_next_states)
            next_state_values[non_final_mask] = target_values.gather(
                1, best_policy_actions.unsqueeze(1)
            ).squeeze(1)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self):
        rewards = []
        for i_episode in range(self.episodes):
            if len(rewards) > 100:
                mean_reward = np.mean(rewards[-100:])
                if mean_reward > self.solved_mean_rw:
                    print("Solved!")
                    break
            else:
                mean_reward = np.mean(rewards)
            print(f"Episode {i_episode} - mean reward: {mean_reward} - epsilon: {self.epsilon}")
            state, _ = self.env.reset()
            done = False
            steps = 0
            reward_sum = 0
            state = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            while not done and (steps < self.max_num_steps):
                steps += 1
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(
                    action.item()
                )
                reward_sum += reward

                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated
                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(
                        observation, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)

                self.memory.push(state, action, next_state, reward)
                state = next_state

                self.optimize_model()
                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[
                        key
                    ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    rewards.append(reward_sum)
                    break
        return rewards

    def __repr__(self) -> str:
        return "ddqn"

    def save_model(self, path):
        torch.save(self.policy_net, path)