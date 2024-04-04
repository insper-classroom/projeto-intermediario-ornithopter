import random
import numpy as np
import torch
from torch.optim import Adam
from .network import create_network
import math


MAX_TIMESTEPS = 1000


class DeepQLearning:
    """
    Implementacao do algoritmo proposto em
    Playing Atari with Deep Reinforcement Learning, Mnih et al., 2013
    https://arxiv.org/abs/1312.5602
    """

    def __init__(self, env, hyperparameters, memory):
        self.env = env
        self.gamma = hyperparameters.gamma

        self.epsilon = hyperparameters.epsilon
        self.epsilon_start = hyperparameters.epsilon
        self.epsilon_min = hyperparameters.epsilon_min
        self.epsilon_dec = hyperparameters.epsilon_dec

        self.episodes = hyperparameters.episodes
        self.batch_size = hyperparameters.batch_size
        self.memory = memory
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_network(env).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=hyperparameters.learning_rate)
        self.steps_done = 0

    def select_action(self, state):
        eps_threshold = self.epsilon_min + (
            self.epsilon_start - self.epsilon_min
        ) * math.exp(-1.0 * self.steps_done / self.epsilon_dec)
        self.steps_done += 1
        if np.random.rand() < eps_threshold:
            return random.randrange(self.env.action_space.n)
        # We dont need gradient to make predictions, grab the action with the highest Q-value
        with torch.no_grad():
            action = self.model(state)
            action = torch.argmax(action).item()
        return action

    # cria uma memoria longa de experiencias
    def experience(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))

    def experience_replay(self):
        # soh acontece o treinamento depois da memoria ser maior que o batch_size informado
        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)

            actions = torch.tensor([i[1] for i in batch], dtype=torch.int8).to(
                self.device
            )
            states = torch.cat([i[0].unsqueeze(0) for i in batch]).to(self.device)
            rewards = torch.tensor([i[2] for i in batch]).to(self.device)
            next_states = torch.cat([i[3].unsqueeze(0) for i in batch]).to(self.device)
            terminals = torch.tensor([i[4] for i in batch], dtype=torch.int8).to(
                self.device
            )

            # usando o modelo para selecionar as melhores acoes
            next_max = self.model(next_states).max(1).values

            targets = rewards + self.gamma * (next_max) * (1 - terminals)
            targets_full = self.model(states)
            indexes = torch.tensor(
                [i for i in range(self.batch_size)], dtype=torch.int16
            ).to(self.device)

            # usando os q-valores para atualizar os pesos da rede
            targets_full[indexes.long(), actions.long()] = targets.float()
            self.fit_model(states, targets_full)

    def fit_model(self, state, target):
        self.optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(self.model(state), target)
        loss.backward()
        self.optimizer.step()

    def transform_state(self, state):
        return torch.tensor(state, dtype=torch.float32).to(self.device)

    def train(self):
        rewards = []
        for episode in range(self.episodes + 1):
            (state, _) = self.env.reset()
            state = self.transform_state(state)
            score = 0
            steps = 0
            done = False
            while not done and steps < MAX_TIMESTEPS:
                steps += 1
                action = self.select_action(state)
                next_state, reward, terminal, truncated, _ = self.env.step(action)
                if terminal or truncated:
                    done = True
                score += reward
                next_state = self.transform_state(next_state)
                self.experience(state, action, reward, next_state, terminal)

                state = next_state
                self.experience_replay()

                if done:
                    print(f"Episódio: {episode}/{self.episodes}. Score: {score}")
                    break
            rewards.append(score)

            if len(rewards) > 100:
                mean_rewards = np.mean(rewards[-100:])
                print(f"Last 100 mean rewards: {mean_rewards}")
                if mean_rewards > 200:
                    print("Solved!")
                    break

        torch.save(self.model, f"models/{self}.pt")
        return rewards

    def __repr__(self) -> str:
        return "dql"
