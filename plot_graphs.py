import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

model_name = input("What is the model name? (dqn/dql)")
# Load the data
df = pd.read_csv(f"results/_{model_name}_results.csv")

window_size = 50
df["rewards_mean"] = df["Reward"].rolling(window=window_size).mean()
# Plot the data
sns.set_theme(style="darkgrid")
sns.lineplot(data=df, x="Episode", y="rewards_mean")
plt.xlabel("Episodes")
plt.ylabel(f"Mean of Rewards (Window Size={window_size})")
plt.title(f"Rewards over time for {model_name} (lr=1e-4, gamma=0.99, epsilon=0.9)")
plt.savefig(f"results/ll_{model_name}_results.jpg", dpi=300)
