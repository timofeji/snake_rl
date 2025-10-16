import time
import numpy as np
import os
import matplotlib.pyplot as plt

from lib.agent import Snake_FourArmedBandit, Snake_RandomWalk
from lib.env import SnakeEnv
from lib.term import *


WIDTH = 10
HEIGHT = 10
TICK_RATE = 0.1
NUM_EPISODES = 5000

agent_random = Snake_RandomWalk()
agent_fourarmed = Snake_FourArmedBandit()
clear_screen()

steps = np.arange(NUM_EPISODES)
rewards_random = np.zeros(NUM_EPISODES)
rewards_fourarmed = np.zeros(NUM_EPISODES)


# ## 4-Armed Bandit Training Loop
# for episode in range(NUM_EPISODES):
#     env = SnakeEnv(width=WIDTH, height=HEIGHT)

#     while not env.game_over:
#         reward = agent_random.forward(env)
#         rewards_random[episode] += reward
#         draw_frame(env)

#     print(f"\n Training... Episode {episode+1}/{NUM_EPISODES}")


## 4-Armed Bandit Training Loop
for episode in range(NUM_EPISODES):
    env = SnakeEnv(width=WIDTH, height=HEIGHT)

    while not env.game_over:
        reward = agent_fourarmed.forward(env)
        rewards_fourarmed[episode] += reward
        draw_frame(env)

    print(f"\n Training... Episode {episode+1}/{NUM_EPISODES}")


# Demo loop
agent_fourarmed.isTraining = False
env = SnakeEnv(width=WIDTH, height=HEIGHT)
while not env.game_over:
    action = agent_fourarmed.forward(env)
    draw_frame(env)
    time.sleep(TICK_RATE)


for v in agent_fourarmed.q_values:
    print(f"State: {dir} => Value: {v:.2f} \n")

# Plot training rewards and save to working directory
try:
    plt.figure(figsize=(10, 6))
    # Calculate cumulative moving average
    cumulative_avg = np.cumsum(rewards_fourarmed) / (steps + 1)
    
    plt.plot(steps, rewards_fourarmed, label="Four-Armed Bandit (Episode Reward)", alpha=0.5)
    plt.plot(steps, cumulative_avg, label="Four-Armed Bandit (Cumulative Average)", linewidth=2)
    
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards Comparison")
    plt.grid(True)
    plt.legend()
    out_path = os.path.join(os.getcwd(), "training_rewards.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved training plot to {out_path}")
except Exception as e:
    print(f"Failed to create/save training plot: {e}")