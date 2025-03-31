"""Base agent class for Mancala games.

This module provides the base agent class for Mancala games.
"""

from typing import Dict, List, Any, Tuple

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

    def get_settings(self) -> Dict[str, Tuple[Any, str, str]]:
        """Get the agent's configurable settings.

        Returns:
            A dictionary mapping setting names to tuples of 
            (current_value, description, type).
            The type should be one of: 'int', 'float', 'bool', 'str'.
        """
        return {}

    def set_setting(self, setting_name: str, value: Any) -> bool:
        """Set a specific setting to a new value.

        Args:
            setting_name: The name of the setting to change.
            value: The new value for the setting.

        Returns:
            True if the setting was successfully updated, False otherwise.
        """
        settings = self.get_settings()
        if setting_name not in settings:
            return False

        # This base implementation doesn't actually change any settings
        # Subclasses should override this method to update their settings
        return False

    def has_settings(self) -> bool:
        """Check if the agent has any configurable settings.

        Returns:
            True if the agent has settings, False otherwise.
        """
        return bool(self.get_settings())
