"""Random agent class for Mancala games.

This module provides the random agent class for Mancala games.
"""

import random
from typing import List

from mancala_ai.agents.base_agent import Agent
from mancala_ai.environments.base_environment import BaseEnvironment


class RandomAgent(Agent):
    """AI agent that selects random valid actions."""

    DISPLAY_NAME = "Random Player"  # Custom display name for this agent

    def get_action(self, env: BaseEnvironment) -> int:
        """Get a random valid action.

        Args:
            env: The game environment.

        Returns:
            The selected action.
        """
        valid_actions = env.get_valid_actions()

        if not valid_actions:
            return -1

        return random.choice(valid_actions)
