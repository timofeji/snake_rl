# --- Bot Algorithm Base Class ---
import math
import os
import random
import numpy as np


from lib.term import draw_frame
from lib.env import ACTION_SPACE, SnakeEnv

random.seed(42)
np.random.seed(42)


WIDTH = 10
HEIGHT = 10

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
            step_reward = self.forward()
            train_rewards.append(step_reward)
            train_steps += 1

            draw_frame(self.env)

        self.rewards.append(sum(train_rewards)/train_steps)

    """Abstract base class for snake bot algorithms."""
    def forward(self) -> int:
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
    def __init__(self, learning_rate=0.1, discount_factor=0.9):
        super().__init__()
        self.q_values = [0 for _ in range(len(ACTION_SPACE))]  # State-Value dictionary
        self.num_action_taken  = [0 for _ in range(len(ACTION_SPACE))]  # Nt(a)
        self.learning_rate = learning_rate  # α
        self.epsilon = .1

    def forward(self) -> int:
        # eplore if random float below epsilon, else choose greedy action
        action = random.randint(0, 3) if random.random() < .01 else np.argmax(self.q_values)

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
            i.   Select action aₖ (ε-greedy or random)
            ii.  Record resulting state sₜ₊₁ and reward rₜ
            iii. Q(sₜ,aₜ) ← rₜ + γ * max Q(sₜ₊₁,a)
            iv.  t ← t + 1
        d. End while
    5. End for
    """
    def __init__(self, width=10, height=10, learning_rate=0.1, discount_factor=1):
        super().__init__()
        self.max_score = 0
        self.learning_rate = learning_rate  # α
        self.discount = discount_factor
        self.q_values = {}


        #initialize q_value table with all possible state(food_pos, head_pos) values, 
        # which returns an array of q_values for each action at that state
        # q(s) -> {q_value(a | state)} for all a
        for x in range(self.env.WIDTH):
            for y in range(self.env.HEIGHT):
                for i in range(self.env.WIDTH):
                    for j in range(self.env.HEIGHT):
                        self.q_values[((x, y), (i, j))] = [0 for _ in range(len(ACTION_SPACE))]


    def train(self):
        self.env.reset()
        train_rewards = []
        train_steps = 0

        while not self.env.game_over:
            step_reward = self.forward()
            train_rewards.append(step_reward)
            train_steps += 1
            # draw_frame(self.env)

        self.rewards.append(sum(train_rewards)/train_steps)
        self.scores.append(self.env.score)
        self.max_score = max(self.env.score, self.max_score)

    def forward(self) -> int:
        state = (self.env.food_pos, self.env.snake[0])

        # eplore if random float below epsilon, else choose greedy action
        action = random.randint(0, 3) if random.random() < .01 else np.argmax(self.q_values[state])
        reward = self.env.step(action)

        # Updated State at t+1
        state_prime = (self.env.food_pos, self.env.snake[0])

        self.q_values[state][action] = self.q_values[state][action] + self.learning_rate * (
            reward  - self.q_values[state_prime][action]
        )

        return reward
