# --- Game Configuration ---
import random
import math


ACTION_SPACE = {
    0: (1, 0),   # Right
    1: (-1, 0),  # Left
    2: (0, 1),   # Down
    3: (0, -1)   # Up
}


class SnakeEnv():
    def __init__(self, width=32, height=32):
        # --- Environment Initialization ---
        self.WIDTH = width
        self.HEIGHT = height

        # --- Game State ---
        # The snake starts with 3 segments, moving right
        start_x = self.WIDTH // 2
        start_y = self.HEIGHT // 2

        self.snake = [(start_x, start_y), (start_x - 1, start_y), (start_x - 2, start_y)]
        self.food_pos = (0, 0)
        self.direction = (1, 0)  # (dx, dy)
        self.score = 0
        self.game_over = False

        self.place_food()

    def place_food(self):
        """Places food in a random location not occupied by the snake."""
        while True:
            food_pos = (random.randint(0, self.WIDTH - 1), random.randint(0, self.HEIGHT - 1))
            if food_pos not in self.snake:
                # self.food_pos = food_pos
                break

    def update_direction(self, action):
        """Updates the snake's direction based on the action."""
        new_direction = ACTION_SPACE.get(action, self.direction)
        # Prevent the snake from reversing
        if (new_direction[0] != -self.direction[0] or new_direction[1] != -self.direction[1]):
            self.direction = new_direction

    def distance(self, pos1, pos2):
        """Calculates the Euclidean distance between two points."""
        return abs(pos1[0] - pos2[0])  + abs(pos1[1] - pos2[1])

    def step(self, action):
        """Moves the snake, checks for collisions and food."""

        self.update_direction(action)

        # Calculate new head position
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)


        old_dist = self.distance(self.snake[0], self.food_pos)
        new_dist = self.distance(new_head, self.food_pos)

        reward = -1 if new_dist >= old_dist else 1

        if not (0 <= new_head[0] < self.WIDTH and 0 <= new_head[1] < self.HEIGHT):
            self.game_over = True
            return 0

        if new_head in self.snake[:-1]:
            self.game_over = True
            return 0

        self.snake.insert(0, new_head)

        # Check for food
        if new_head == self.food_pos:
            self.score += 10
            # self.place_food()
            self.game_over = True
            # reward = 0
        else:
            self.snake.pop()

        return reward
