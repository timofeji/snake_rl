import sys
import time
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

from lib.agent import Snake, Snake_Associative, Snake_DQN, Snake_FourArmedBandit, Snake_Associative_Body, Snake_MonteCarlo
from lib.term import clear_screen, draw_frame


parser = argparse.ArgumentParser(description="Train Snake agents (with faster options)")
parser.add_argument("--episodes", type=int, default=15750, help="Number of training episodes")
parser.add_argument("--tick", type=float, default=0.02, help="Tick rate for demo playback")
parser.add_argument("--no-plot", action="store_true", help="Skip plotting/saving training plot")
parser.add_argument("--no-draw", action="store_true", help="Disable drawing during training/demo to speed up")
parser.add_argument("--width", type=int, default=20, help="Environment width (smaller speeds training)")
parser.add_argument("--height", type=int, default=20, help="Environment height (smaller speeds training)")
parser.add_argument("--seed", type=int, default=None, help="Random seed for deterministic runs")
parser.add_argument("--step-limit", type=int, default=1000, help="Maximum steps per episode (halts long games)")
args = parser.parse_args()

NUM_EPISODES = args.episodes
TICK_RATE = args.tick

clear_screen()


agent_random = Snake()
agent_fourarmed = Snake_FourArmedBandit()
agent_ass = Snake_MonteCarlo()
agent_dqn = Snake_DQN()
# agent_ass.load()

for episode in range(NUM_EPISODES):
    agent_dqn.train()
    if episode % 100 == 0:
        out = f"Training... \nEpisode [{episode+1}/{NUM_EPISODES}] Top Score[{agent_dqn.max_score}]" 
        sys.stdout.write("\033[?25l") 
        sys.stdout.write("\033[H")  
        sys.stdout.write(out)
        sys.stdout.flush()
        sys.stdout.write("\033[?25h")
        agent_dqn.scores.append(agent_dqn.env.score)


# # Optionally plot results
# if not args.no_plot:
#     try:
#         episodes = np.arange(NUM_EPISODES)
#         plt.figure(figsize=(10, 6))

#         # rewards = np.array(agent_ass.rewards)
#         scores = np.array(agent_ass.scores)
#         n = np.arange(len(scores))

#         window = 100  # number of episodes to average over
#         if len(scores) >= window:
#             smoothed_scores = np.convolve(scores, np.ones(window)/window, mode='valid')
#             plt.plot(np.arange(len(smoothed_scores)), smoothed_scores, label=f"Score (window={window})", linewidth=2)


#         # plt.plot(episodes[:n], rewards, label="Associative (Episode Reward)", alpha=0.2)
#         # plt.plot(x_moving, moving_rewards, label=f"Associative (Moving Avg, window={MOVING_AVG_WINDOW})", linewidth=2)
#         # plt.plot(n, scores, label=f"Avg Scores ", linewidth=2)
#         plt.xlabel("Episode")
#         plt.ylabel("Average Total Reward")
#         plt.title("Training Rewards Comparison")
#         plt.grid(True)
#         plt.legend()
#         out_path = os.path.join(os.getcwd(), "training_rewards.png")
#         plt.savefig(out_path, bbox_inches="tight")
#         plt.close()
#         print(f"Saved training plot to {out_path}")
#     except Exception as e:
#         print(f"Failed to create/save training plot: {e}")

# agent_ass.load()
# agent_ass.load("model_checkpoint.pkl")
# Demo playback (greedy) — only run if drawing is enabled
if not args.no_draw:
    while True:
        agent_ass.env.reset()
        while not agent_ass.env.game_over:
            s = agent_ass.env.state
            a = agent_ass.policy[s]
            agent_ass.env.step(a)

            draw_frame(agent_ass.env)
            time.sleep(TICK_RATE)
