"""Base agent class for Mancala games.

This module provides the base agent class for Mancala games.
"""

from typing import List

from mancala_ai.environments.base_environment import BaseEnvironment


class Agent:
    """Base class for agents (human or AI)."""

    DISPLAY_NAME = "Agent"  # Default display name, should be overridden by subclasses

    def __init__(self, name: str):
        """Initialize an agent.

        Args:
            name: The name of the agent.
        """
        self.name = name
        self.is_interactive = False  # Default to non-interactive

    def get_action(self, env: BaseEnvironment) -> int:
        """Get the next action for the agent.

        Args:
            env: The game environment.

        Returns:
            The selected action.
        """
        raise NotImplementedError("Subclasses must implement get_action")
