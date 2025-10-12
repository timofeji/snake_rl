import time

from snake.env import SnakeEnv
from snake.term import *


TICK_RATE = 0.2

env = SnakeEnv()

while not env.game_over:
    action = process_input()
    observation, reward, terminated, _, _ = env.step(action)
    draw_frame(env)
    time.sleep(TICK_RATE)