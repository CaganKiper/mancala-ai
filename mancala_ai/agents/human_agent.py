"""Human agent class for Mancala games.

This module provides the human agent class for Mancala games.
"""

from typing import List

from mancala_ai.agents.base_agent import Agent
from mancala_ai.environments.base_environment import BaseEnvironment


class HumanAgent(Agent):
    """Human agent that gets actions from user input."""

    DISPLAY_NAME = "Human Player"  # Custom display name for this agent

    def __init__(self, name: str):
        """Initialize a human agent.

        Args:
            name: The name of the agent.
        """
        super().__init__(name)
        self.is_interactive = True  # Human agents are interactive

    def get_action(self, env: BaseEnvironment) -> int:
        """Get the next action from user input.

        Args:
            env: The game environment.

        Returns:
            The selected action.
        """
        valid_actions = env.get_valid_actions()

        if not valid_actions:
            print("No valid actions available.")
            return -1

        # Display valid actions (1-indexed for user-friendliness)
        print(f"Valid moves: {[action + 1 for action in valid_actions]}")

        while True:
            try:
                # Get input (1-indexed for user-friendliness)
                action_input = input(f"{self.name}, enter your move (1-{env.num_pits}): ")

                # Check if the user wants to quit
                if action_input.lower() in ['q', 'quit', 'exit']:
                    return -1

                # Convert to 0-indexed action
                action = int(action_input) - 1

                if action in valid_actions:
                    return action
                else:
                    print(f"Invalid move. Please choose from {[action + 1 for action in valid_actions]}")
            except ValueError:
                print("Please enter a valid number.")
