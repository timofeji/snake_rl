# --- Game Configuration ---
import random
import torch
import numpy as np

ACTION_TO_DIR = {
    0: (1, 0),   # Right
    1: (-1, 0),  # Left
    2: (0, 1),   # Down
    3: (0, -1)   # Up
}

DIR_TO_ACTION = {i: s for s, i in ACTION_TO_DIR.items()}

class SnakeEnv():
    def __init__(self, width=20, height=20):
        # --- Environment Initialization ---
        self.WIDTH = width
        self.HEIGHT = height
        # Use an instance-local RNG so behavior can be made deterministic per-environment
        # when a seed is provided.
        self.reset()

    def reset(self):
        # --- Game State ---
        # The snake starts with 3 segments, moving right
        start_x = self.WIDTH // 2
        start_y = self.HEIGHT // 2
        self.clip_length = 1 // self.WIDTH 

        self.snake = [
            (start_x, start_y),
            (start_x - 1, start_y),
            (start_x - 2, start_y),
        ]

        self.direction = (1, 0)  # (dx, dy)
        self.last_action = 0
        self.score = 0
        self.food_pos = (2, 2)
        self.time_alive = 0
        self.game_over = False
        self.updateState()

        pass

    def place_food(self):
        """Places food in a random location not occupied by the snake."""
        while True:
            # Use the environment's RNG (deterministic when seed is set) instead of the
            # global random module.
            food_pos = (random.randint(0, self.WIDTH - 1), random.randint(0, self.HEIGHT - 1))
            if food_pos not in self.snake:
                self.food_pos = food_pos
                break

    def update_direction(self, action):
        """Updates the snake's direction based on the action."""
        new_direction = ACTION_TO_DIR.get(action, self.direction)
        self.last_action = action
        # Prevent the snake from reversing
        # A reversal would have both components be the negation of the current direction.
        # Only allow the new direction if it's not the exact opposite.
        if not (new_direction[0] == -self.direction[0] and new_direction[1] == -self.direction[1]):
            self.direction = new_direction

    def isColliding(self, pos):
        x, y = pos
        # Combined boundary and body check
        return not (0 <= x < self.WIDTH and 0 <= y < self.HEIGHT) or pos in self.snake[:-1]

    def updateState(self):
        if(self.game_over):
            return

        self.state = [0] * (self.WIDTH * self.HEIGHT)

        food_x, food_y = self.food_pos
        self.state[food_y * self.WIDTH + food_x] = 1

        for s in self.snake:
            self.state[s[1] * self.WIDTH + s[0]] = 2

    # def updateState(self):
    #     head_x, head_y = self.snake[0]
    #     food_dx = head_x - self.food_pos[0] + 19
    #     food_dy = head_y - self.food_pos[1] + 19

    #     dx, dy = self.direction

    #     # Pre-compute collision positions
    #     collision_positions = (
    #         (head_x + dx, head_y + dy),
    #         (head_x + 4*dx, head_y + 4*dy),
    #         (head_x + dx + dy, head_y + dy - dx),
    #         (head_x + 4*dx + 4*dy, head_y + 4*dy - 4*dx),
    #         (head_x + dy, head_y - dx),
    #         (head_x + 4*dy, head_y - 4*dx),
    #         (head_x + dx - dy, head_y + dy + dx),
    #         (head_x + 4*dx - 4*dy, head_y + 4*dy + 4*dx)
    #     )

    #     # Convert snake body to set once (if not already cached)
    #     body_set = set(self.snake[:-1])

    #     # Check collisions using bitwise operations
    #     collision_bits = sum(
    #         1 << i for i, pos in enumerate(collision_positions)
    #         if not (0 <= pos[0] < self.WIDTH and 0 <= pos[1] < self.HEIGHT) or pos in body_set
    #     )

    #     self.state = ((DIR_TO_ACTION[self.direction] * 39 + food_dx) * 39 + food_dy) * 256 + collision_bits

    # self.state = (self.last_action,food_dx, food_dy, forward, left, right)
    # self.state = tuple((
    #     DIR_TO_ACTION[self.direction],
    #     #
    #     food_dx,
    #     food_dy,
    #     forward,
    #     left,
    #     right,
    # ))

    def step(self, action):
        """Moves the snake, checks for collisions and food."""

        self.update_direction(action)

        # Calculate new head position
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        old_dx = self.food_pos[0] - head_x
        old_dy = self.food_pos[1] - head_y
        new_dx = self.food_pos[0] - new_head[0]
        new_dy = self.food_pos[1] - new_head[1]

        old_dist_sq = old_dx**2 + old_dy**2
        new_dist_sq = new_dx**2 + new_dy**2

        reward = 1 if new_dist_sq < old_dist_sq else -.1

        if new_head == self.food_pos:
            self.score += 1
            self.place_food()
            reward = 10
            self.time_alive = 0
        else:
            self.snake.pop()

        if self.isColliding(new_head):
            self.game_over = True
            reward = -15

        self.snake.insert(0, new_head)
        self.updateState()

        return reward
