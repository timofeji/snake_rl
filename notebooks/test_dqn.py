"""
Quick test script for DQN implementation

Tests basic functionality of the DQN agent before full training.
"""

import torch
import numpy as np
from lib.agent import Snake_DQN

def test_dqn_basic():
    """Test basic DQN functionality"""
    print("Testing DQN Basic Functionality")
    print("=" * 50)
    
    # Create agent
    agent = Snake_DQN(width=10, height=10)
    
    # Test 1: Network initialization
    print("\n1. Network Architecture:")
    print(f"   Q-Network: {agent.q_net}")
    print(f"   State size: {agent.state_size}")
    print(f"   Number of actions: {agent.num_actions}")
    
    # Test 2: State conversion
    print("\n2. State Conversion:")
    agent.env.reset()
    state = agent.env.state
    print(f"   Original state shape: {len(state)}x{len(state[0])}")
    state_tensor = agent.state_to_tensor(state)
    print(f"   Tensor shape: {state_tensor.shape}")
    
    # Test 3: Action selection
    print("\n3. Action Selection:")
    action = agent.select_action(state)
    print(f"   Selected action: {action}")
    print(f"   Current epsilon: {agent.epsilon}")
    
    # Test 4: Q-value prediction
    print("\n4. Q-Value Prediction:")
    with torch.no_grad():
        q_values = agent.q_net(state_tensor)
        print(f"   Q-values: {q_values}")
        print(f"   Best action: {q_values.argmax().item()}")
    
    # Test 5: Store and sample transitions
    print("\n5. Replay Buffer:")
    for i in range(100):
        state = agent.env.state
        action = agent.select_action(state)
        reward = agent.env.step(action)
        next_state = agent.env.state
        done = agent.env.game_over
        agent.store_transition(state, action, reward, next_state, done)
        if done:
            agent.env.reset()
    
    print(f"   Buffer size: {len(agent.replay_buffer)}")
    batch = agent.sample_batch()
    if batch:
        states, actions, rewards, next_states, dones = batch
        print(f"   Batch states shape: {states.shape}")
        print(f"   Batch actions shape: {actions.shape}")
        print(f"   Batch rewards shape: {rewards.shape}")
    
    # Test 6: Learning step
    print("\n6. Learning Step:")
    loss = agent.learn()
    print(f"   Loss: {loss:.4f if loss else 'N/A (not enough samples)'}")
    
    # Test 7: Full episode
    print("\n7. Full Episode Test:")
    agent.env.reset()
    metrics = agent.train()
    print(f"   Score: {metrics['score']}")
    print(f"   Steps: {metrics['steps']}")
    print(f"   Avg Reward: {metrics['avg_reward']:.4f}")
    print(f"   Avg Loss: {metrics['avg_loss']:.4f}")
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)

def test_dqn_training():
    """Test short training run"""
    print("\nTesting Short Training Run (10 episodes)")
    print("=" * 50)
    
    agent = Snake_DQN(
        width=10,
        height=10,
        learning_rate=0.001,
        exploration_rate=1.0,
        batch_size=32
    )
    
    for episode in range(10):
        metrics = agent.train()
        print(f"Episode {episode + 1}: "
              f"Score={metrics['score']:2d}, "
              f"Steps={metrics['steps']:3d}, "
              f"Loss={metrics['avg_loss']:.4f}, "
              f"ε={metrics['epsilon']:.4f}")
    
    print("\n" + "=" * 50)
    print(f"Training completed!")
    print(f"Max score: {agent.max_score}")
    print(f"Buffer size: {len(agent.replay_buffer)}")
    print("=" * 50)

def test_save_load():
    """Test model saving and loading"""
    print("\nTesting Save/Load Functionality")
    print("=" * 50)
    
    # Train for a few episodes
    print("\n1. Training initial agent...")
    agent1 = Snake_DQN(width=10, height=10)
    for _ in range(5):
        agent1.train()
    
    print(f"   Scores: {agent1.scores}")
    print(f"   Max score: {agent1.max_score}")
    
    # Save
    print("\n2. Saving model...")
    agent1.save("test_dqn.pth")
    
    # Load into new agent
    print("\n3. Loading into new agent...")
    agent2 = Snake_DQN(width=10, height=10)
    agent2.load("test_dqn.pth")
    
    print(f"   Loaded scores: {agent2.scores}")
    print(f"   Loaded max score: {agent2.max_score}")
    
    # Verify they match
    assert agent1.scores == agent2.scores, "Scores don't match!"
    assert agent1.max_score == agent2.max_score, "Max scores don't match!"
    
    print("\n" + "=" * 50)
    print("Save/Load test passed! ✓")
    print("=" * 50)

if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 50)
    print("DQN TESTING SUITE")
    print("=" * 50)
    
    try:
        test_dqn_basic()
        test_dqn_training()
        test_save_load()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED! 🎉")
        print("=" * 50)
        print("\nYou're ready to start full training!")
        print("Run: python train_dqn.py")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
