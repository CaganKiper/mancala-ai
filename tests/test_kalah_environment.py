"""Test script for the Kalah environment.

This script tests the functionality of the KalahEnvironment class,
including the Kalah-specific rules such as capturing and extra turns.
"""

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mancala_ai.environments import KalahEnvironment


def main():
    """Run a test of the KalahEnvironment class."""
    # Initialize the environment
    env = KalahEnvironment(num_pits=6, num_stones=4)

    # Reset the environment and render the initial state
    env.reset()
    print("Initial state:")
    env.render()

    # Make a series of moves to demonstrate Kalah-specific rules
    print("\n--- Testing Kalah-specific rules ---")

    # Make moves until the game is over or we've made 10 moves
    move_count = 0
    while not env.done and move_count < 10:
        move_count += 1

        # Get valid actions
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            print("No valid actions available. Game over.")
            break

        # Choose the first valid action
        action = valid_actions[0]
        print(f"\nMove {move_count}: Player {env.current_player + 1} selects pit {action + 1}")

        # Remember the current player to check for extra turns
        current_player = env.current_player

        # Take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)

        # Render the new state
        env.render()

        print(f"Reward: {reward}")

        # Check if the player got an extra turn
        if not terminated and env.current_player == current_player:
            print("Player got an extra turn!")

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
        print("\nFinal state after 10 moves:")
        env.render()


if __name__ == "__main__":
    main()
