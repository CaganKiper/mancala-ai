"""Pytest configuration file for Mancala AI tests.

This file contains fixtures and configuration for pytest.
"""

import pytest
import numpy as np

from mancala_ai.environments.base_environment import BaseEnvironment
from mancala_ai.environments.kalah_environment import KalahEnvironment
from mancala_ai.agents.random_agent import RandomAgent
from mancala_ai.agents.human_agent import HumanAgent


@pytest.fixture
def base_environment():
    """Fixture for a BaseEnvironment instance."""
    env = BaseEnvironment(num_pits=6, num_stones=4)
    env.reset()
    return env


@pytest.fixture
def kalah_environment():
    """Fixture for a KalahEnvironment instance."""
    env = KalahEnvironment(num_pits=6, num_stones=4)
    env.reset()
    # Print the board state for debugging
    print(f"Board state after reset: {env.board}")
    return env


@pytest.fixture
def random_agent():
    """Fixture for a RandomAgent instance."""
    return RandomAgent(name="Test Random Agent")


@pytest.fixture
def custom_board_state():
    """Fixture for a custom board state.

    This creates a board state that can be used to test specific scenarios.
    """
    # Create a board with a specific state
    # [4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0]
    board = np.array([4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0], dtype=np.int32)
    return board
