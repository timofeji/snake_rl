# --- Bot Algorithm Base Class ---
import math
import os
import random
import numpy as np
import pickle


from lib.term import draw_frame
from lib.env import ACTION_TO_DIR, SnakeEnv


WIDTH = 20
HEIGHT = 20

class Snake():
    def __init__(self):
        self.env = SnakeEnv(width=WIDTH, height=HEIGHT)
        self.rewards = []
        self.scores = []

    def train(self):
        self.env.reset()

        train_rewards = []
        train_steps = 0
        while not self.env.game_over:
            step_reward = self.step()
            train_rewards.append(step_reward)
            train_steps += 1

            draw_frame(self.env)

        self.rewards.append(sum(train_rewards)/train_steps)

    """Abstract base class for snake bot algorithms."""
    def step(self) -> int:
        """
        Calculate the next action for the snake and advance on step
        Returns:
            Reward from taking an action in the environment
        """
        action = random.randint(0, 3)   
        reward = self.env.step(action)
        return reward


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

class Snake_FourArmedBandit(Snake):
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
    def __init__(self, learning_rate=0.1, exploration_rate=0.01):
        super().__init__()
        self.q_values = [0 for _ in range(len(ACTION_TO_DIR))]  # State-Value dictionary
        self.num_action_taken  = [0 for _ in range(len(ACTION_TO_DIR))]  # Nt(a)
        self.learning_rate = learning_rate  # α
        self.epsilon = exploration_rate

    def step(self) -> int:
        # eplore if random float below epsilon, else choose greedy action
        action = random.randint(0, 3) if random.random() < self.epsilon else np.argmax(self.q_values)

        reward = self.env.step(action)
        self.num_action_taken[action] += 1

        self.q_values[action] = self.q_values[action] + self.learning_rate * (
            reward  - self.q_values[action]
        )

        return reward


### Associative Learning
# So far, we only learn the average value of the actions, which isn't enough to even steer the snake to the food reward.
# We want Q(a,s), which is defined in terms of v(s)
# q*(s,a) = max_π q_π(s,a)
# q*(s,a) = E[R_t+1 + γv*(S_t+1) | S_t = s, A_t = a]

# Compared to Multi-Armed bandit, we now take into account the actual state
# and thus value-function(Total amount of reward that an agent can expect to accumulate over the future)
# starting from state S_t and following policy π.
# v*(S_t+1) = E[R|S_t+1]

class Snake_Associative(Snake):
    """
    ☆*: .｡. o(≧▽≦)o .｡.:*☆
    Tries to learn q_values for state-action pairs instead of just actions

    π(a) - probability of a certain action [0,1]
    q(a,s) - quality of the action taken(expected return from performing an action)
    v(s) - expected return of being in a state
    forms a dependency chain:
        π -> q -> v

    so, to get π, we need to calculate v, and q, then update our probability distribution
        
    Q-Learning Algorithm:

    self.q_values[state] -> 4 q_values for each action
    
    1. Initialize Q(s,a) = 0 for all states s and actions a
    2. Set number of episodes E (about 10,000)
    3. Set maximum steps per episode T (From beginning to death)
    4. For each episode e = 1, 2, ..., E:
        a. t ← 1 
        b. Select random initial state s₁
        c. While goal state not reached and t ≤ T:
            i.   Select action aₖ (ε-greedy)
            ii.  Record resulting state sₜ₊₁ and reward rₜ
            iii. Q(sₜ,aₜ) ← rₜ + γ * max Q(sₜ₊₁,a)
            iv.  t ← t + 1
        d. End while
    5. End for
    """
    def __init__(self, width=20, height=20, learning_rate=.1, discount_factor=.97, exploration_rate = 0.1):
        super().__init__()
        self.max_score = 0
        self.learning_rate = learning_rate  # α
        self.discount_factor = discount_factor
        self.epsilon = exploration_rate 

        # initialize q_value table with all possible state(food_pos, head_pos) values,
        # which returns an array of q_values for each action at that state
        # q(s) -> {q_value(a | state)} for all a

        num_states =  self.env.WIDTH * self.env.WIDTH *self.env.HEIGHT *self.env.HEIGHT
        self.q_values = np.full((num_states, len(ACTION_TO_DIR)), -.1)

    def train(self):
        self.env.reset()
        train_rewards = []
        train_steps = 0

        while not self.env.game_over:
            step_reward = self.step()
            train_rewards.append(step_reward)
            train_steps += 1
            # draw_frame(self.env)

        self.rewards.append(sum(train_rewards)/train_steps)
        self.scores.append(self.env.score)
        self.max_score = max(self.env.score, self.max_score)

    def step(self) -> int:
        grid_size = self.env.WIDTH * self.env.HEIGHT

        food_index = self.env.food_pos[1] * self.env.WIDTH + self.env.food_pos[0]
        head_index = self.env.snake[0][1] *   self.env.WIDTH + self.env.snake[0][0]
        state_index = food_index * grid_size + head_index

        # eplore if random float below epsilon, else choose greedy action
        action = random.randint(0, 3) if random.random() < self.epsilon else np.argmax(self.q_values[state_index])
        reward = self.env.step(action)

        # Updated State at t+1
        food_index = self.env.food_pos[1] * self.env.WIDTH + self.env.food_pos[0]
        head_index = self.env.snake[0][1] *   self.env.WIDTH + self.env.snake[0][0]
        state_prime_index  = food_index * grid_size + head_index

        if self.env.game_over:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_values[state_prime_index])


        self.q_values[state_index][action] += self.learning_rate * (
            target - self.q_values[state_index][action]
        )

        return reward

    def save(self, filepath="model_checkpoint.pkl"):
        """
        Save Q-values table and training metrics to a pickle file.
        
        Args:
            filepath: Path where the model will be saved (default: "model_checkpoint.pkl")
        """
        save_data = {
            'q_values': self.q_values,
            'rewards': self.rewards,
            'scores': self.scores,
            'max_score': self.max_score,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon
        }

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Failed to save model: {e}")

    def load(self, filepath="model_checkpoint.pkl"):
        """
        Load Q-values table and training metrics from a pickle file.
        
        Args:
            filepath: Path to the saved model file (default: "model_checkpoint.pkl")
        
        Returns:
            True if load was successful, False otherwise
        """
        if not os.path.exists(filepath):
            print(f"No saved model found at {filepath}")
            return False

        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)

            self.q_values = save_data['q_values']
            self.rewards = save_data['rewards']
            self.scores = save_data['scores']
            self.max_score = save_data['max_score']
            self.learning_rate = save_data['learning_rate']
            self.discount_factor = save_data['discount_factor']
            self.epsilon = save_data['epsilon']

            print(f"Model loaded from {filepath}")
            print(f"  Episodes trained: {len(self.rewards)}")
            print(f"  Max score: {self.max_score}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False


### Associative Learning
# We're learning to get the food now, but we're not storing enough state information for the algorhitm to keep track of the snake's body
# and so we're plataueing in the maxing amount of reward we can get as the snake collides with itself.
class Snake_Associative_Body(Snake):
    """
    """
    def __init__(self, width=20, height=20, learning_rate=0.025, discount_factor=.75, exploration_rate = 0.005):
        super().__init__()
        self.max_score = 0
        self.learning_rate = learning_rate  # α
        self.discount_factor = discount_factor
        self.epsilon = exploration_rate 
        self.q_values = {}

    def train(self):
        self.env.reset()
        while not self.env.game_over:
            step_reward = self.step()
            # draw_frame(self.env)

        self.max_score = max(self.env.score, self.max_score)

    def step(self) -> int:
        state = self.env.state

        if state not in self.q_values:
            self.q_values[state] = [0] * len(ACTION_TO_DIR)

        # eplore if random float below epsilon, else choose greedy action
        action = (
            random.randint(0, 3)
            if random.random() < self.epsilon
            else np.argmax(self.q_values[state])
        )
        reward = self.env.step(action)

        # Updated State at t+1
        state_prime  = self.env.state

        # Initialize state_prime if not seen before
        if state_prime not in self.q_values:
            self.q_values[state_prime] = [0] * len(ACTION_TO_DIR)

        # If the environment reached a terminal state (game over) after taking the action,
        # the target should be just the immediate reward (no bootstrapped future value).
        if self.env.game_over:
            target = reward
        else:
            target = reward + self.discount_factor * max(self.q_values[state_prime])

        # Standard TD/Q-learning incremental update
        self.q_values[state][action] = (1 - self.learning_rate) * self.q_values[state][
            action
        ] + self.learning_rate * (target - self.q_values[state][action])

        return reward

    def save(self, filepath="model_checkpoint.pkl"):
        """
        Save Q-values table and training metrics to a pickle file.
        
        Args:
            filepath: Path where the model will be saved (default: "model_checkpoint.pkl")
        """
        save_data = {
            'q_values': self.q_values,
        }

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Failed to save model: {e}")

    def load(self, filepath="model_checkpoint.pkl"):
        """
        Load Q-values table and training metrics from a pickle file.
        
        Args:
            filepath: Path to the saved model file (default: "model_checkpoint.pkl")
        
        Returns:
            True if load was successful, False otherwise
        """
        if not os.path.exists(filepath):
            print(f"No saved model found at {filepath}")
            return False

        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)

            self.q_values = save_data['q_values']

            print(f"Model loaded from {filepath}")
            print(f"  Episodes trained: {len(self.rewards)}")
            print(f"  Max score: {self.max_score}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False


class Snake_MonteCarlo(Snake):
    """
    """
    def __init__(self, width=20, height=20, learning_rate=0.025, discount_factor=.95, exploration_rate = 0.1):
        super().__init__()
        self.max_score = 0
        self.learning_rate = learning_rate  # α
        self.discount_factor = discount_factor
        self.epsilon = exploration_rate 
        # self.q_values = {}
        # self.policy = {}
        # self.returns = {}

        self.num_actions = len(ACTION_TO_DIR)

        grid_size = (width * height) - 1
        num_states = 4 * grid_size * grid_size * 256
        self.q_values = np.zeros((num_states, len(ACTION_TO_DIR)), dtype=np.float32)
        self.policy = np.zeros(num_states, dtype=np.int8)
        self.rng = np.random.default_rng()

    def train(self):
        self.env.reset()

        episode = []
        while not self.env.game_over:
            s = self.env.state
            a = self.rng.integers(self.num_actions) if self.rng.random() < self.epsilon else  self.policy[s]
            r = self.env.step(a)
            episode.append((s,a,r))

        states, actions, rewards = zip(*episode)
        rewards = np.array(rewards, dtype=np.float32)

        # discounted cumulative sum from the end
        discounts = self.discount_factor ** np.arange(len(rewards))
        returns = np.flip(np.cumsum(np.flip(rewards * discounts))) / discounts

        visited = set()
        for s, a, G in zip(states, actions, returns):
            key = (s, a)
            if key in visited:  # only first occurrence from end
                continue
            visited.add(key)
            current_q = self.q_values[s, a]
            self.q_values[s, a] += self.learning_rate * (G - current_q)
            self.policy[s] = np.argmax(self.q_values[s])

        self.max_score = max(self.env.score, self.max_score)

    def save(self, filepath="montecarlo.pkl"):
        """
        Save Q-values table and training metrics to a pickle file.
        
        Args:
            filepath: Path where the model will be saved (default: "model_checkpoint.pkl")
        """
        save_data = {
            'q_values': self.q_values,
            'policy' :self.policy
        }

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Failed to save model: {e}")

    def load(self, filepath="montecarlo.pkl"):
        """
        Load Q-values table and training metrics from a pickle file.
        
        Args:
            filepath: Path to the saved model file (default: "model_checkpoint.pkl")
        
        Returns:
            True if load was successful, False otherwise
        """
        if not os.path.exists(filepath):
            print(f"No saved model found at {filepath}")
            return False

        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)

            self.q_values = save_data['q_values']
            self.policy = save_data['policy']

            print(f"Model loaded from {filepath}")
            print(f"  Episodes trained: {len(self.rewards)}")
            print(f"  Max score: {self.max_score}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
