# --- Bot Algorithm Base Class ---
import math
import os
import random

import numpy as np

from lib.env import ACTION_SPACE

np.random.seed(42)


class Agent():
    """Abstract base class for snake bot algorithms."""
    def forward(self, snake, food_pos, current_direction, width, height) -> int:
        """
        Calculate the next direction for the snake.
        
        Args:
            snake: List of (x, y) tuples representing snake positions
            food_pos: (x, y) tuple for food location
            current_direction: Current (dx, dy) direction tuple
            width: Game board width
            height: Game board height
            
        Returns:
            Action for the next direction
        """
        pass


class Snake_RandomWalk(Agent):
    def forward(self, env) -> int:
        action = random.randint(0, 3)   
        reward = env.step(action)
        return np.average(reward)


# Simple Value Estimator
# 3 ingredients:
# 1. Policy (π): A strategy that the agent employs to determine the next action based on the current state.
###  Probability of selecting action a in state s under policy π: π(a|s)

# 2. Reward: Immediate reward received after taking an action.
##   Defines the ultimate goal of the agent, over the long run,
##   given to to the agent at each time step t after taking action a_t in state S_t
##   I.E. Pleasure Pain stimulus.

# 3. Value Function: Total amount of reward that an agent can expect to accumulate over the future,
##   starting from state S_t and following policy π.
##   Different from reward which is immediate, value function is long-term.

# Updates values using Temporal Difference Learning:
# V(S_t) =  V(S_t) + a(V(S_t+1) - V(S_t))

# q*(a) = E[R_t | a_t = a]  # Expected return after taking action a

class Snake_FourArmedBandit(Agent):
    """ 
    Simple k-armed Bandit  
    initialize for all a in ACTION_SPACE: 
        Q(a) <- 0
        N(a) <- 0 
    on each step:
        with probability ε select a random action a
        otherwise select a = argmax_a Q(a)
        R <- env(a) reward after taking action a
        N(a) <- N(a) + 1
        Q(a) <- Q(a) + (1/N(a)) * (R - Q(a))  # Incremental update
        
    # this follows the form of Temporal Difference Learning:    
        V(S_t) =  V(S_t) + a(V(S_t+1) - V(S_t)) 

    Values stored in a dictionary with keys as (snake_tuple, food_pos)
    """
    def __init__(self, width=32, height=32, learning_rate=0.1, discount_factor=0.9):
        super().__init__()
        self.q_values = [0 for _ in range(len(ACTION_SPACE))]  # State-Value dictionary
        self.num_action_taken  = [0 for _ in range(len(ACTION_SPACE))]  # Nt(a)
        self.learning_rate = learning_rate  # α
        self.isTraining = True
        self.epsilon = .1
        self.degree_of_exploration = 2
        self.time = 1

    def forward(self, env) -> int:
        action = random.randint(0, 3)

        # 10% exploration rate
        explore = (random.random() < .01) and self.isTraining
        # Choose action greedily
        if not explore:
            action = np.argmax(self.q_values)

        reward = env.step(action)
        self.num_action_taken[action] += 1
        self.time += 1

        self.q_values[action] = self.q_values[action] + self.learning_rate * (
            reward  - self.q_values[action]
        )

        return np.average(reward)


# class GreedyBot(SnakeBot):
#     """Simple bot that always moves toward the food."""

#     def get_next_direction(self, snake, food_pos, current_direction, width, height):
#         head_x, head_y = snake[0]
#         food_x, food_y = food_pos

#         # Calculate differences
#         dx_to_food = food_x - head_x
#         dy_to_food = food_y - head_y

#         # Possible directions in priority order
#         possible_directions = []

#         # Prioritize horizontal or vertical based on distance
#         if abs(dx_to_food) > abs(dy_to_food):
#             if dx_to_food > 0:
#                 possible_directions.append((1, 0))   # Right
#             elif dx_to_food < 0:
#                 possible_directions.append((-1, 0))  # Left

#             if dy_to_food > 0:
#                 possible_directions.append((0, 1))   # Down
#             elif dy_to_food < 0:
#                 possible_directions.append((0, -1))  # Up
#         else:
#             if dy_to_food > 0:
#                 possible_directions.append((0, 1))   # Down
#             elif dy_to_food < 0:
#                 possible_directions.append((0, -1))  # Up

#             if dx_to_food > 0:
#                 possible_directions.append((1, 0))   # Right
#             elif dx_to_food < 0:
#                 possible_directions.append((-1, 0))  # Left

#         # Add perpendicular directions as fallback
#         if current_direction[0] == 0:  # Currently moving vertically
#             possible_directions.extend([(1, 0), (-1, 0)])
#         else:  # Currently moving horizontally
#             possible_directions.extend([(0, 1), (0, -1)])

#         # Try each direction and pick the first valid one
#         for new_dir in possible_directions:
#             if self._is_opposite_direction(new_dir, current_direction):
#                 continue

#             new_head = (head_x + new_dir[0], head_y + new_dir[1])

#             # Check if this move is safe
#             if self._is_valid_move(new_head, snake, width, height):
#                 return new_dir

#         # If no safe move found, continue current direction
#         return current_direction

#     def _is_opposite_direction(self, new_dir, current_dir):
#         """Check if new direction is opposite to current direction."""
#         return (new_dir[0] == -current_dir[0] and new_dir[1] == -current_dir[1])

#     def _is_valid_move(self, pos, snake, width, height):
#         """Check if a position is valid (not wall or snake body)."""
#         x, y = pos
#         if not (0 <= x < width and 0 <= y < height):
#             return False
#         if pos in snake[:-1]:  # Exclude tail as it will move
#             return False
#         return True

# class PathfindingBot(SnakeBot):
#     """Bots that uses BFS to find the hortest path to food."""

#     def get_next_direction(self, snake, food_pos, current_direction, width, height):
#         path = self._bfs_to_food(snake, food_pos, width, height)

#         if path and len(path) > 1:
#             # Get the first step in the path
#             next_pos = path[1]
#             head_x, head_y = snake[0]
#             new_dir = (next_pos[0] - head_x, next_pos[1] - head_y)

#             # Make sure it's not opposite direction
#             if not self._is_opposite_direction(new_dir, current_direction):
#                 return new_dir

#         # Fallback to greedy approach if no path found
#         greedy = GreedyBot()
#         return greedy.get_next_direction(snake, food_pos, current_direction, width, height)

#     def _bfs_to_food(self, snake, food_pos, width, height):
#         """Use BFS to find shortest path to food."""
#         head = snake[0]
#         queue = deque([(head, [head])])
#         visited = {head}

#         while queue:
#             current_pos, path = queue.popleft()

#             if current_pos == food_pos:
#                 return path

#             # Explore neighbors
#             for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
#                 next_pos = (current_pos[0] + dx, current_pos[1] + dy)

#                 if next_pos in visited:
#                     continue

#                 # Check if position is valid
#                 if not (0 <= next_pos[0] < width and 0 <= next_pos[1] < height):
#                     continue

#                 # Allow moving to tail position as it will move away
#                 if next_pos in snake[:-1]:
#                     continue

#                 visited.add(next_pos)
#                 queue.append((next_pos, path + [next_pos]))

#         return None  # No path found

#     def _is_opposite_direction(self, new_dir, current_dir):
#         """Check if new direction is opposite to current direction."""
#         return (new_dir[0] == -current_dir[0] and new_dir[1] == -current_dir[1])
