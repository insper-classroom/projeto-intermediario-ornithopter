import argparse

SUPPORTED_ENVS = ["CartPole-v0", "LunarLander-v2", "FlappyBird-v0"]
SUPPORTED_MODELS = ["dql", "dqn", "ddqn"]


def parse_args():
    parser = argparse.ArgumentParser(description="Intermediate project RL")
    parser.add_argument(
        "--env-name", type=str, default=SUPPORTED_ENVS[0], choices=SUPPORTED_ENVS
    )
    parser.add_argument(
        "--model",
        type=str,
        default=SUPPORTED_MODELS[-1],
        choices=SUPPORTED_MODELS,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-times", type=int, default=1)
    return parser.parse_args()


def parse_args_trained():
    parser = argparse.ArgumentParser(description="Intermediate project RL")
    parser.add_argument(
        "--env-name", type=str, default=SUPPORTED_ENVS[0], choices=SUPPORTED_ENVS
    )
    parser.add_argument("--steps", type=int, default=2000)
    return parser.parse_args()
