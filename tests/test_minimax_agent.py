"""Tests for the MinimaxAgent class.

This module contains tests for the MinimaxAgent class, which implements the Minimax algorithm.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from mancala_ai.agents.minimax_agent import MinimaxAgent
from mancala_ai.environments.base_environment import BaseEnvironment
from mancala_ai.environments.kalah_environment import KalahEnvironment


class TestMinimaxAgent:
    """Tests for the MinimaxAgent class."""

    def test_initialization(self):
        """Test that the minimax agent initializes correctly."""
        agent = MinimaxAgent(name="Test Minimax Agent")
        assert agent.name == "Test Minimax Agent"
        assert not agent.is_interactive
        assert agent.depth == 3  # Default depth

    def test_initialization_with_custom_depth(self):
        """Test that the minimax agent initializes correctly with a custom depth."""
        agent = MinimaxAgent(name="Test Minimax Agent", depth=5)
        assert agent.name == "Test Minimax Agent"
        assert not agent.is_interactive
        assert agent.depth == 5

    def test_get_action_with_valid_actions(self, base_environment):
        """Test that get_action returns a valid action."""
        # Create a minimax agent
        agent = MinimaxAgent(name="Test Minimax Agent", depth=2)
        
        # Get an action
        action = agent.get_action(base_environment)
        
        # The action should be valid
        assert action in base_environment.get_valid_actions()

    def test_get_action_with_no_valid_actions(self, base_environment):
        """Test that get_action returns -1 when there are no valid actions."""
        # Create a minimax agent
        agent = MinimaxAgent(name="Test Minimax Agent")
        
        # Modify the environment to have no valid actions
        base_environment.board = np.zeros_like(base_environment.board)
        
        # Get an action
        action = agent.get_action(base_environment)
        
        # The action should be -1 (no valid actions)
        assert action == -1

    def test_minimax_chooses_winning_move(self):
        """Test that the minimax algorithm chooses a winning move when available."""
        # Create a custom environment with a winning move
        env = KalahEnvironment(num_pits=6, num_stones=4)
        
        # Set up a board state where player 0 can win in one move
        # Player 0's pits: [0, 0, 0, 0, 0, 1]
        # Player 0's store: 24
        # Player 1's pits: [0, 0, 0, 0, 0, 0]
        # Player 1's store: 23
        env.board = np.zeros(14, dtype=np.int32)
        env.board[5] = 1  # Player 0's last pit has 1 stone
        env.board[6] = 24  # Player 0's store has 24 stones
        env.board[13] = 23  # Player 1's store has 23 stones
        env.current_player = 0
        
        # Create a minimax agent
        agent = MinimaxAgent(name="Test Minimax Agent", depth=2)
        
        # Get an action
        action = agent.get_action(env)
        
        # The action should be 5 (the winning move)
        assert action == 5

    def test_minimax_blocks_opponent_winning_move(self):
        """Test that the minimax algorithm blocks an opponent's winning move."""
        # Create a custom environment where player 1 can win in one move
        env = KalahEnvironment(num_pits=6, num_stones=4)
        
        # Set up a board state where player 1 can win in one move, but player 0 can block
        # Player 0's pits: [0, 0, 0, 0, 1, 1]
        # Player 0's store: 23
        # Player 1's pits: [0, 0, 0, 0, 0, 1]
        # Player 1's store: 22
        env.board = np.zeros(14, dtype=np.int32)
        env.board[4] = 1  # Player 0's second-to-last pit has 1 stone
        env.board[5] = 1  # Player 0's last pit has 1 stone
        env.board[6] = 23  # Player 0's store has 23 stones
        env.board[12] = 1  # Player 1's last pit has 1 stone
        env.board[13] = 22  # Player 1's store has 22 stones
        env.current_player = 0
        
        # Create a minimax agent
        agent = MinimaxAgent(name="Test Minimax Agent", depth=3)
        
        # Get an action
        action = agent.get_action(env)
        
        # The action should be 4 or 5 (blocking moves)
        assert action in [4, 5]

    def test_minimax_prefers_extra_turn(self):
        """Test that the minimax algorithm prefers moves that give an extra turn."""
        # Create a custom environment where an extra turn is possible
        env = KalahEnvironment(num_pits=6, num_stones=4)
        
        # Set up a board state where player 0 can get an extra turn
        # Player 0's pits: [0, 0, 0, 0, 0, 1]
        # Player 0's store: 0
        # Player 1's pits: [1, 1, 1, 1, 1, 1]
        # Player 1's store: 0
        env.board = np.zeros(14, dtype=np.int32)
        env.board[5] = 1  # Player 0's last pit has 1 stone
        for i in range(7, 13):
            env.board[i] = 1  # Player 1's pits each have 1 stone
        env.current_player = 0
        
        # Create a minimax agent
        agent = MinimaxAgent(name="Test Minimax Agent", depth=2)
        
        # Get an action
        action = agent.get_action(env)
        
        # The action should be 5 (gives an extra turn)
        assert action == 5