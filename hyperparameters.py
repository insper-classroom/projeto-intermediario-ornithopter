from dataclasses import dataclass


@dataclass
class Hyperparameters:
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_dec: float = 0.99
    episodes: int = 200
    batch_size: int = 64
    memory_size: int = 10000
    # This is used in the DQN model only.
    target_update_rate: float = 0.005
