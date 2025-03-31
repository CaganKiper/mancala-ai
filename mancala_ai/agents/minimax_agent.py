"""Minimax agent class for Mancala games.

This module provides the minimax agent class for Mancala games, which uses the
Minimax algorithm to select actions.
"""

import copy
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

from mancala_ai.agents.base_agent import Agent
from mancala_ai.environments.base_environment import BaseEnvironment


class MinimaxAgent(Agent):
    """AI agent that uses the Minimax algorithm to select actions."""

    DISPLAY_NAME = "Minimax Agent"  # Custom display name for this agent

    def __init__(self, name: str, depth: int = 3):
        """Initialize a minimax agent.

        Args:
            name: The name of the agent.
            depth: The maximum depth to search in the game tree.
        """
        super().__init__(name)
        self.depth = depth

    def get_settings(self) -> Dict[str, Tuple[Any, str, str]]:
        """Get the agent's configurable settings.

        Returns:
            A dictionary mapping setting names to tuples of 
            (current_value, description, type).
        """
        return {
            "depth": (self.depth, "Maximum depth to search in the game tree", "int")
        }

    def set_setting(self, setting_name: str, value: Any) -> bool:
        """Set a specific setting to a new value.

        Args:
            setting_name: The name of the setting to change.
            value: The new value for the setting.

        Returns:
            True if the setting was successfully updated, False otherwise.
        """
        if setting_name == "depth":
            try:
                depth = int(value)
                if depth > 0:
                    self.depth = depth
                    return True
                else:
                    print("Depth must be a positive integer.")
                    return False
            except ValueError:
                print("Depth must be a valid integer.")
                return False
        return False

    def get_action(self, env: BaseEnvironment) -> int:
        """Get the best action according to the Minimax algorithm.

        Args:
            env: The game environment.

        Returns:
            The selected action.
        """
        valid_actions = env.get_valid_actions()

        if not valid_actions:
            return -1

        # Create a deep copy of the environment to avoid modifying the original
        env_copy = copy.deepcopy(env)

        # Find the best action using minimax
        best_action, _ = self._minimax(env_copy, self.depth, True)
        return best_action

    def _minimax(self, env: BaseEnvironment, depth: int, maximizing_player: bool) -> Tuple[Optional[int], float]:
        """Recursive minimax algorithm implementation.

        Args:
            env: The game environment.
            depth: The current depth in the search tree.
            maximizing_player: Whether the current player is maximizing or minimizing.

        Returns:
            A tuple containing:
                - The best action (or None if no valid actions or terminal state).
                - The value of the best action.
        """
        # Check if we've reached a terminal state or maximum depth
        if depth == 0 or env.done:
            return None, self._evaluate(env)

        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return None, self._evaluate(env)

        # Initialize best action and value
        best_action = None
        if maximizing_player:
            best_value = float('-inf')
        else:
            best_value = float('inf')

        # Try each valid action
        for action in valid_actions:
            # Create a deep copy of the environment
            env_copy = copy.deepcopy(env)

            # Take the action
            original_player = env_copy.current_player
            _, reward, terminated, _, _ = env_copy.step(action)

            # Check if the player gets an extra turn (current player didn't change)
            extra_turn = not terminated and env_copy.current_player == original_player

            # If the player gets an extra turn, it's still their turn in the minimax tree
            next_maximizing = maximizing_player if extra_turn else not maximizing_player

            # Recursively evaluate the resulting state
            _, value = self._minimax(env_copy, depth - 1, next_maximizing)

            # Update best action and value
            if maximizing_player:
                if value > best_value:
                    best_value = value
                    best_action = action
            else:
                if value < best_value:
                    best_value = value
                    best_action = action

        return best_action, best_value

    def _evaluate(self, env: BaseEnvironment) -> float:
        """Evaluate the current state of the environment.

        Args:
            env: The game environment.

        Returns:
            A value representing how good the state is for the current player.
        """
        # If the game is over, return a large positive or negative value
        if env.done:
            winner = env.get_winner()
            if winner is None:  # Draw
                return 0.0
            elif winner == 0:  # Player 0 wins
                return 1000.0
            else:  # Player 1 wins
                return -1000.0

        # Otherwise, use the difference in stones in the stores as the evaluation
        player0_store = env.board[env.num_pits]
        player1_store = env.board[2 * env.num_pits + 1]

        # Return the difference from the perspective of the current player
        if env.current_player == 0:
            return player0_store - player1_store
        else:
            return player1_store - player0_store
