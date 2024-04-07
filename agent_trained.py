import flappy_bird_gymnasium
import gymnasium as gym
import torch
from pygame_recorder import PygameRecord
from cli_interface import parse_args_trained
import os


args = parse_args_trained()
env_name = args.env_name
model_name = args.model
print(f"Using model {model_name} for environment {env_name}")

# Print data folder
print("All checkpoints in models folder:")
all_models = os.listdir(f"models/{env_name}")
if not all_models:
    print("No models found")
    exit(0)

if len(all_models) == 1:
    model_name = all_models[0]
else:
    for model in all_models:
        print(model)
    model_name = input("Enter the model to use: ")

model_path = f"models/{env_name}/{model_name}"
if not model_path.endswith(".pt"):
    model_path = f"{model_path}.pt"

model = torch.load(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def predict(state):
    with torch.no_grad():
        return torch.argmax(model(state)).item()


done = False
truncated = False
rewards = 0
steps = 0
if env_name == "FlappyBird-v0":
    env = gym.make(env_name, render_mode="human", pipe_gap=150, use_lidar=False, normalize_obs=False, score_limit=500)
else:
    env = gym.make(env_name, render_mode="human")
(state, _) = env.reset()

RENDER_FPS = 50
# Remove extension and split by underscore
model_name = model_name.removesuffix(".pt")
model_name = model_name.split("_")[0]
with PygameRecord(
    f"results/{env_name}_{model_name}.gif", RENDER_FPS
) as recorder:
    while (not done) and (not truncated) and (steps < args.steps):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = predict(state)
        state, reward, done, truncated, info = env.step(action)
        rewards += reward
        env.render()
        steps += 1
        recorder.add_frame()
    print(f"Steps = {steps}")
    env.close()

print(f"Score = {rewards}")
input("press a key...")
