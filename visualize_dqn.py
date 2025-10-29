"""
Visualization and analysis tools for DQN agent

Helps understand what the network has learned.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from lib.agent import Snake_DQN
from lib.env import ACTION_TO_DIR

ACTION_NAMES = {0: "RIGHT", 1: "LEFT", 2: "DOWN", 3: "UP"}

def visualize_q_values(agent, state):
    """
    Visualize Q-values for a given state
    
    Args:
        agent: Trained Snake_DQN agent
        state: Game state (2D grid)
    """
    with torch.no_grad():
        state_tensor = agent.state_to_tensor(state)
        q_values = agent.q_net(state_tensor).squeeze().numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot the game state
    state_np = np.array(state)
    ax1.imshow(state_np, cmap='viridis', interpolation='nearest')
    ax1.set_title('Game State\n(0=empty, 1=food, 2=snake)')
    ax1.grid(False)
    
    # Add gridlines
    ax1.set_xticks(np.arange(-.5, state_np.shape[1], 1), minor=True)
    ax1.set_yticks(np.arange(-.5, state_np.shape[0], 1), minor=True)
    ax1.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    
    # Plot Q-values
    colors = ['blue' if q < 0 else 'green' for q in q_values]
    bars = ax2.bar(range(4), q_values, color=colors, alpha=0.7)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels([ACTION_NAMES[i] for i in range(4)])
    ax2.set_ylabel('Q-Value')
    ax2.set_title('Q-Values for Each Action')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Highlight best action
    best_action = np.argmax(q_values)
    bars[best_action].set_edgecolor('red')
    bars[best_action].set_linewidth(3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, q_values)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom' if height > 0 else 'top')
    
    plt.tight_layout()
    return fig

def analyze_policy(agent, num_states=100, save_path='policy_analysis.png'):
    """
    Analyze the learned policy by sampling random states
    
    Args:
        agent: Trained Snake_DQN agent
        num_states: Number of random states to sample
    """
    action_counts = {i: 0 for i in range(4)}
    avg_q_values = {i: [] for i in range(4)}
    
    for _ in range(num_states):
        # Generate random state
        agent.env.reset()
        # Take some random steps to get varied states
        for _ in range(np.random.randint(0, 20)):
            if agent.env.game_over:
                agent.env.reset()
            agent.env.step(np.random.randint(0, 4))
        
        if not agent.env.game_over:
            state = agent.env.state
            with torch.no_grad():
                state_tensor = agent.state_to_tensor(state)
                q_values = agent.q_net(state_tensor).squeeze().numpy()
            
            best_action = np.argmax(q_values)
            action_counts[best_action] += 1
            
            for i, q in enumerate(q_values):
                avg_q_values[i].append(q)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Action preference distribution
    actions = [ACTION_NAMES[i] for i in range(4)]
    counts = [action_counts[i] for i in range(4)]
    ax1.bar(actions, counts, color='steelblue', alpha=0.7)
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Preferred Actions Across {num_states} States')
    ax1.grid(True, alpha=0.3)
    
    # Average Q-values per action
    avg_qs = [np.mean(avg_q_values[i]) for i in range(4)]
    std_qs = [np.std(avg_q_values[i]) for i in range(4)]
    ax2.bar(actions, avg_qs, yerr=std_qs, color='coral', alpha=0.7, capsize=5)
    ax2.set_ylabel('Average Q-Value')
    ax2.set_title('Average Q-Values by Action')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Policy analysis saved to {save_path}")
    plt.close()
    
    return action_counts, avg_q_values

def watch_trained_agent(checkpoint_path='dqn_checkpoint.pth', num_games=5):
    """
    Watch the trained agent play and see Q-values in real-time
    
    Args:
        checkpoint_path: Path to saved model
        num_games: Number of games to watch
    """
    agent = Snake_DQN()
    if not agent.load(checkpoint_path):
        print("Failed to load model")
        return
    
    agent.epsilon = 0.0  # Pure exploitation
    
    print(f"\nWatching trained agent play {num_games} games...")
    print("=" * 50)
    
    for game in range(num_games):
        agent.env.reset()
        steps = 0
        total_reward = 0
        
        print(f"\nGame {game + 1}:")
        
        while not agent.env.game_over and steps < 1000:
            state = agent.env.state
            
            # Get Q-values
            with torch.no_grad():
                state_tensor = agent.state_to_tensor(state)
                q_values = agent.q_net(state_tensor).squeeze().numpy()
            
            action = np.argmax(q_values)
            reward = agent.env.step(action)
            
            total_reward += reward
            steps += 1
            
            # Print every 10 steps
            if steps % 10 == 0:
                print(f"  Step {steps}: Score={agent.env.score}, "
                      f"Action={ACTION_NAMES[action]}, "
                      f"Q-values=[{', '.join([f'{q:.2f}' for q in q_values])}]")
        
        print(f"  Final: Score={agent.env.score}, Steps={steps}, Total Reward={total_reward:.2f}")
        print("-" * 50)

def compare_random_vs_trained(checkpoint_path='dqn_checkpoint.pth', num_games=20):
    """
    Compare random agent vs trained DQN agent
    
    Args:
        checkpoint_path: Path to saved model
        num_games: Number of games for each agent
    """
    # Random agent
    print("Testing random agent...")
    random_scores = []
    random_steps = []
    
    for _ in range(num_games):
        from lib.env import SnakeEnv
        env = SnakeEnv(width=20, height=20)
        env.reset()
        steps = 0
        
        while not env.game_over and steps < 1000:
            action = np.random.randint(0, 4)
            env.step(action)
            steps += 1
        
        random_scores.append(env.score)
        random_steps.append(steps)
    
    # Trained agent
    print("Testing trained agent...")
    agent = Snake_DQN()
    if not agent.load(checkpoint_path):
        print("Failed to load model")
        return
    
    agent.epsilon = 0.0
    trained_scores = []
    trained_steps = []
    
    for _ in range(num_games):
        agent.env.reset()
        steps = 0
        
        while not agent.env.game_over and steps < 1000:
            action = agent.select_action(agent.env.state)
            agent.env.step(action)
            steps += 1
        
        trained_scores.append(agent.env.score)
        trained_steps.append(steps)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scores comparison
    axes[0].boxplot([random_scores, trained_scores], 
                    labels=['Random', 'DQN'],
                    showmeans=True)
    axes[0].set_ylabel('Score')
    axes[0].set_title('Score Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # Steps comparison
    axes[1].boxplot([random_steps, trained_steps],
                    labels=['Random', 'DQN'],
                    showmeans=True)
    axes[1].set_ylabel('Steps Survived')
    axes[1].set_title('Survival Time Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('random_vs_trained.png')
    print("\nComparison saved to random_vs_trained.png")
    plt.close()
    
    # Print statistics
    print("\n" + "=" * 50)
    print("COMPARISON RESULTS")
    print("=" * 50)
    print(f"Random Agent:")
    print(f"  Avg Score: {np.mean(random_scores):.2f} ± {np.std(random_scores):.2f}")
    print(f"  Max Score: {np.max(random_scores)}")
    print(f"  Avg Steps: {np.mean(random_steps):.2f} ± {np.std(random_steps):.2f}")
    print(f"\nTrained DQN Agent:")
    print(f"  Avg Score: {np.mean(trained_scores):.2f} ± {np.std(trained_scores):.2f}")
    print(f"  Max Score: {np.max(trained_scores)}")
    print(f"  Avg Steps: {np.mean(trained_steps):.2f} ± {np.std(trained_steps):.2f}")
    print(f"\nImprovement:")
    print(f"  Score: {((np.mean(trained_scores) / np.mean(random_scores) - 1) * 100):.1f}%")
    print(f"  Survival: {((np.mean(trained_steps) / np.mean(random_steps) - 1) * 100):.1f}%")
    print("=" * 50)

if __name__ == "__main__":
    import sys
    
    checkpoint = 'dqn_checkpoint.pth'
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "watch":
            watch_trained_agent(checkpoint, num_games=5)
        
        elif command == "analyze":
            agent = Snake_DQN()
            if agent.load(checkpoint):
                analyze_policy(agent, num_states=100)
        
        elif command == "compare":
            compare_random_vs_trained(checkpoint, num_games=20)
        
        elif command == "qvalues":
            agent = Snake_DQN()
            if agent.load(checkpoint):
                agent.env.reset()
                # Take a few steps
                for _ in range(5):
                    action = agent.select_action(agent.env.state)
                    agent.env.step(action)
                
                fig = visualize_q_values(agent, agent.env.state)
                plt.savefig('q_values_visualization.png')
                print("Q-values visualization saved to q_values_visualization.png")
                plt.show()
        
        else:
            print(f"Unknown command: {command}")
            print("Available commands: watch, analyze, compare, qvalues")
    else:
        print("Usage:")
        print("  python visualize_dqn.py watch     - Watch agent play")
        print("  python visualize_dqn.py analyze   - Analyze learned policy")
        print("  python visualize_dqn.py compare   - Compare random vs trained")
        print("  python visualize_dqn.py qvalues   - Visualize Q-values for a state")
