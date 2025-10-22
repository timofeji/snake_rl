# --- Game Configuration ---
import random

import numpy as np

ACTION_SPACE = {
    0: (1, 0),   # Right
    1: (-1, 0),  # Left
    2: (0, 1),   # Down
    3: (0, -1)   # Up
}


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
        new_direction = ACTION_SPACE.get(action, self.direction)
        # Prevent the snake from reversing
        # A reversal would have both components be the negation of the current direction.
        # Only allow the new direction if it's not the exact opposite.
        if not (new_direction[0] == -self.direction[0] and new_direction[1] == -self.direction[1]):
            self.direction = new_direction
            self.last_action = action

    def distance(self, pos1, pos2):
        """Calculates the Euclidean distance between two points."""
        return abs(pos1[0] - pos2[0])  + abs(pos1[1] - pos2[1])

    def isDangerous(self, pos):
        x, y = pos
        # check boundaries
        if not (0 <= x < self.WIDTH and 0 <= y < self.HEIGHT):
            return True
        # check collision with body (skip tail)
        for segment in self.snake[:-1]:
            if pos == segment:
                return True
        return False

    def updateState(self):
        head_pos = self.snake[0]
        food_dx = self.snake[0][0] - self.food_pos[0]
        food_dy = self.snake[0][1] - self.food_pos[1]

        dx = head_pos[0] + self.direction[0]
        dy = head_pos[1] + self.direction[1]
        forward = self.isDangerous((dx, dy))
        left = self.isDangerous((-dy, dx))
        right = self.isDangerous((dy, -dx))

        length = len(self.snake)
        

        # self.state = (self.last_action,food_dx, food_dy, forward, left, right)
        self.state = (
            self.last_action,
            int(np.clip(3 * length * self.clip_length , 0, 2)),
            food_dx,
            food_dy,
            forward,
            left,
            right,
        )

    def step(self, action):
        """Moves the snake, checks for collisions and food."""

        self.update_direction(action)

        # Calculate new head position
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        # old_dist = self.distance(self.snake[0], self.food_pos)
        # new_dist = self.distance(new_head, self.food_pos)

        # reward = -.5 if new_dist >= old_dist else 0.05

        old_dx = self.food_pos[0] - head_x
        old_dy = self.food_pos[1] - head_y
        new_dx = self.food_pos[0] - new_head[0]
        new_dy = self.food_pos[1] - new_head[1]

        old_dist_sq = old_dx**2 + old_dy**2
        new_dist_sq = new_dx**2 + new_dy**2

        reward = 0.05 if new_dist_sq < old_dist_sq else -0.5

        if new_head == self.food_pos:
            self.score += 1
            self.place_food()
            reward = 100
            self.time_alive = 0
        else:
            self.snake.pop()

        if self.isDangerous(new_head):
            self.game_over = True
            reward = -155 

        reward -= .0005*self.time_alive

        self.time_alive += 1

        self.snake.insert(0, new_head)
        self.updateState()

        return reward
