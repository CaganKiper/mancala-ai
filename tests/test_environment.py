"""Test script for the Mancala environment.

This script tests the basic functionality of the BaseEnvironment class.
"""

from mancala_ai.environments import BaseEnvironment


def main():
    """Run a simple test of the BaseEnvironment class."""
    # Initialize the environment
    env = BaseEnvironment(num_pits=6, num_stones=4)
    
    # Reset the environment and render the initial state
    env.reset()
    print("Initial state:")
    env.render()
    
    # Make a few moves
    for _ in range(5):
        # Get valid actions
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            print("No valid actions available. Game over.")
            break
        
        # Choose the first valid action
        action = valid_actions[0]
        print(f"\nPlayer {env.current_player + 1} selects pit {action + 1}")
        
        # Take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Render the new state
        env.render()
        
        print(f"Reward: {reward}")
        
        if terminated:
            print("Game over!")
            winner = env.get_winner()
            if winner is not None:
                print(f"Player {winner + 1} wins!")
            else:
                print("It's a draw!")
            break
    
    # Print final state if not already done
    if not env.done:
        print("\nFinal state after 5 moves:")
        env.render()


if __name__ == "__main__":
    main()