import gymnasium as gym
import torch
from pygame_recorder import PygameRecord
import os

# Print data folder
print("All models in models folder:")
for model in os.listdir("models"):
    print(model)

model_name = input("Enter the model to use: ")

if not model_name.endswith(".pt"):
    model_name = f"{model_name}.pt"

model = torch.load(f"models/{model_name}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def predict(state):
    with torch.no_grad():
        return torch.argmax(model(state)).item()


done = False
truncated = False
rewards = 0
steps = 0
env = gym.make("LunarLander-v2", render_mode="human").env
(state, _) = env.reset()

LUNAR_LANDER_FPS = 50
MAX_STEPS = 1000
with PygameRecord(
    f"results/lander_trained_{model_name}.gif", LUNAR_LANDER_FPS
) as recorder:
    while (not done) and (not truncated) and (steps < MAX_STEPS):
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
