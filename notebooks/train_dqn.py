"""
Training script for Snake DQN agent

This script trains a Deep Q-Network to play Snake.
It includes training loop, metrics tracking, and model checkpointing.
"""

import torch
import matplotlib.pyplot as plt
from lib.agent import Snake_DQN
from lib.term import draw_frame

def plot_training_progress(agent, save_path='training_progress.png'):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Scores over time
    axes[0, 0].plot(agent.scores)
    axes[0, 0].set_title('Score per Episode')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].grid(True)
    
    # Average rewards over time
    axes[0, 1].plot(agent.rewards)
    axes[0, 1].set_title('Average Reward per Episode')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Avg Reward')
    axes[0, 1].grid(True)
    
    # Moving average of scores (window=100)
    if len(agent.scores) >= 100:
        moving_avg = [sum(agent.scores[max(0, i-99):i+1]) / min(i+1, 100) 
                     for i in range(len(agent.scores))]
        axes[1, 0].plot(moving_avg)
        axes[1, 0].set_title('Moving Average Score (window=100)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Avg Score')
        axes[1, 0].grid(True)
    
    # Max score over time
    max_scores = [max(agent.scores[:i+1]) for i in range(len(agent.scores))]
    axes[1, 1].plot(max_scores)
    axes[1, 1].set_title('Max Score Over Time')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Max Score')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training progress plot saved to {save_path}")
    plt.close()

def train_dqn(num_episodes=1000, save_interval=100, visualize=False, 
              checkpoint_path='dqn_checkpoint.pth', resume=False):
    """
    Train the DQN agent
    
    Args:
        num_episodes: Number of training episodes
        save_interval: Save model every N episodes
        visualize: Whether to visualize the game during training
        checkpoint_path: Path to save/load checkpoints
        resume: Whether to resume from checkpoint
    """
    
    # Initialize agent
    agent = Snake_DQN(
        width=20, 
        height=20,
        learning_rate=0.001,
        discount_factor=0.95,
        exploration_rate=1.0,  # Start with high exploration
        epsilon_decay=0.995,
        epsilon_min=0.01,
        batch_size=64,
        target_update_freq=10
    )
    
    # Resume from checkpoint if requested
    start_episode = 0
    if resume:
        if agent.load(checkpoint_path):
            start_episode = len(agent.scores)
            print(f"Resuming training from episode {start_episode}")
        else:
            print("Starting fresh training")
    
    print(f"\nTraining DQN for {num_episodes} episodes")
    print(f"Starting epsilon: {agent.epsilon:.4f}")
    print("-" * 50)
    
    try:
        for episode in range(start_episode, start_episode + num_episodes):
            # Train one episode
            metrics = agent.train()
            
            # Print progress
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1:4d} | "
                      f"Score: {metrics['score']:2d} | "
                      f"Max: {agent.max_score:2d} | "
                      f"Steps: {metrics['steps']:3d} | "
                      f"Loss: {metrics['avg_loss']:.4f} | "
                      f"ε: {metrics['epsilon']:.4f}")
            
            # Save checkpoint
            if (episode + 1) % save_interval == 0:
                agent.save(checkpoint_path)
                plot_training_progress(agent)
                print(f"\n>>> Checkpoint saved at episode {episode + 1}")
                print(f">>> Replay buffer size: {len(agent.replay_buffer)}\n")
        
        # Final save
        agent.save(checkpoint_path)
        plot_training_progress(agent)
        
        print("\n" + "=" * 50)
        print(f"Training completed!")
        print(f"Max score achieved: {agent.max_score}")
        print(f"Final epsilon: {agent.epsilon:.4f}")
        print(f"Total episodes: {len(agent.scores)}")
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving current progress...")
        agent.save(checkpoint_path)
        plot_training_progress(agent)
        print("Progress saved!")
    
    return agent

def evaluate_agent(checkpoint_path='dqn_checkpoint.pth', num_games=10, visualize=True):
    """
    Evaluate a trained DQN agent
    
    Args:
        checkpoint_path: Path to the saved model
        num_games: Number of games to play
        visualize: Whether to visualize the games
    """
    agent = Snake_DQN()
    
    if not agent.load(checkpoint_path):
        print("Failed to load model. Exiting.")
        return
    
    # Set epsilon to 0 for pure exploitation
    agent.epsilon = 0.0
    
    scores = []
    print(f"\nEvaluating agent for {num_games} games...")
    print("-" * 50)
    
    for game in range(num_games):
        agent.env.reset()
        steps = 0
        
        while not agent.env.game_over:
            state = agent.env.state
            action = agent.select_action(state)
            agent.env.step(action)
            steps += 1
            
            if visualize:
                draw_frame(agent.env)
        
        scores.append(agent.env.score)
        print(f"Game {game + 1}: Score = {agent.env.score}, Steps = {steps}")
    
    print("-" * 50)
    print(f"Average Score: {sum(scores) / len(scores):.2f}")
    print(f"Max Score: {max(scores)}")
    print(f"Min Score: {min(scores)}")
    print("=" * 50)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        # Evaluation mode
        evaluate_agent(
            checkpoint_path='dqn_checkpoint.pth',
            num_games=10,
            visualize=True
        )
    else:
        # Training mode
        agent = train_dqn(
            num_episodes=1000,
            save_interval=100,
            visualize=False,
            checkpoint_path='dqn_checkpoint.pth',
            resume=False  # Set to True to continue training from checkpoint
        )
