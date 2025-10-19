import time
import numpy as np
import os
import matplotlib.pyplot as plt

from lib.agent import Snake, Snake_Associative, Snake_FourArmedBandit
from lib.term import *


NUM_EPISODES = 50000
TICK_RATE = 0.1



clear_screen()



agent_random = Snake()
agent_fourarmed = Snake_FourArmedBandit()
agent_ass = Snake_Associative()

for episode in range(NUM_EPISODES):
    print(f"\n Training... Episode {episode+1}/{NUM_EPISODES}")
    # agent_random.train()
    # agent_fourarmed.train()
    agent_ass.train()


print(agent_ass.max_score)

# Plot training rewards and save to working directory
try:

    episodes = np.arange(NUM_EPISODES)
    plt.figure(figsize=(10, 6))
    # Calculate cumulative moving average
    cumulative_avg = np.cumsum(agent_ass.rewards) / (episodes + 1)
    cum_scores = np.cumsum(agent_ass.scores) / (episodes + 1)
    # cumulative_avg_1 = np.cumsum(agent_random.rewards) / (episodes + 1)
    
    plt.plot(episodes, agent_ass.rewards, label="Associative (Episode Reward)", alpha=0.2)
    plt.plot(episodes, cumulative_avg, label="Associative (Cumulative Average)", linewidth=2)
    plt.plot(episodes, cum_scores, label="Cummulative Scores", linewidth=2)
    # plt.plot(episodes, cumulative_avg_1, label="Random Walk(Cumulative Average)", linewidth=2)
    
    plt.xlabel("Episode")
    plt.ylabel("Average Total Reward")
    plt.title("Training Rewards Comparison")
    plt.grid(True)
    plt.legend()
    out_path = os.path.join(os.getcwd(), "training_rewards.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved training plot to {out_path}")
except Exception as e:
    print(f"Failed to create/save training plot: {e}")