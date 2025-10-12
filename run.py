import random
import time

from snake.env import SnakeEnv
from snake.term import *


WIDTH = 32
HEIGHT = 32
TICK_RATE = 0.2

env = SnakeEnv(width=WIDTH, height=HEIGHT)
clear_screen()

while not env.game_over:
    action = random.randint(0, 3)
    observation, reward, terminated, _, _ = env.step(action)
    draw_frame(env)
    time.sleep(TICK_RATE)