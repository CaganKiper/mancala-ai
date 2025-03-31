"""Tests for the BaseEnvironment class.

This module contains tests for the BaseEnvironment class, which implements
the core mechanics of a Mancala game.
"""

import pytest
import numpy as np

from mancala_ai.environments.base_environment import BaseEnvironment


class TestBaseEnvironment:
    """Tests for the BaseEnvironment class."""

    def test_initialization(self, base_environment):
        """Test that the environment initializes correctly."""
        # Check that the board has the correct shape
        assert len(base_environment.board) == 2 * (base_environment.num_pits + 1)

        # Check that the pits have the correct number of stones
        for i in range(base_environment.num_pits):
            assert base_environment.board[i] == base_environment.num_stones

        for i in range(base_environment.num_pits + 1, 2 * base_environment.num_pits + 1):
            assert base_environment.board[i] == base_environment.num_stones

        # Check that the stores are empty
        assert base_environment.board[base_environment.num_pits] == 0
        assert base_environment.board[2 * base_environment.num_pits + 1] == 0

        # Check that player 0 starts
        assert base_environment.current_player == 0

        # Check that the game is not done
        assert not base_environment.done

    def test_reset(self):
        """Test that reset returns the environment to its initial state."""
        # Create an environment and modify its state
        env = BaseEnvironment(num_pits=6, num_stones=4)
        env.board[0] = 0  # Modify the board
        env.current_player = 1  # Change the player
        env.done = True  # Mark the game as done

        # Reset the environment
        observation = env.reset()

        # Check that the board has been reset
        assert env.board[0] == env.num_stones

        # Check that the player has been reset
        assert env.current_player == 0

        # Check that the game is not done
        assert not env.done

        # Check that the observation is correct
        assert np.array_equal(observation, env.board)

    def test_valid_actions(self, base_environment):
        """Test that get_valid_actions returns the correct actions."""
        # In the initial state, all pits for the current player should be valid
        valid_actions = base_environment.get_valid_actions()
        assert valid_actions == list(range(base_environment.num_pits))

        # If a pit is empty, it should not be a valid action
        base_environment.board[0] = 0
        valid_actions = base_environment.get_valid_actions()
        assert 0 not in valid_actions
        assert valid_actions == list(range(1, base_environment.num_pits))

    def test_step_invalid_action(self, base_environment):
        """Test that taking an invalid action returns an error."""
        # Try to take an action that's out of range
        observation, reward, terminated, truncated, info = base_environment.step(-1)
        assert reward == -1.0
        assert not terminated
        assert not truncated
        assert "error" in info

        # Empty a pit and try to select it
        base_environment.board[0] = 0
        observation, reward, terminated, truncated, info = base_environment.step(0)
        assert reward == -1.0
        assert not terminated
        assert not truncated
        assert "error" in info

    def test_step_valid_action(self, base_environment):
        """Test that taking a valid action updates the state correctly."""
        # Take a valid action
        action = 0
        stones_in_pit = base_environment.board[action]
        observation, reward, terminated, truncated, info = base_environment.step(action)

        # Check that the pit is now empty
        assert base_environment.board[action] == 0

        # Check that the stones have been distributed
        # (This is a simple check; more detailed checks would depend on the specific rules)
        assert np.sum(base_environment.board) == base_environment.num_pits * 2 * base_environment.num_stones

        # Check that the player has changed
        assert base_environment.current_player == 1

    def test_game_end(self):
        """Test that the game ends correctly when all pits of a player are empty."""
        # Create an environment with a specific state where player 0's pits are empty
        env = BaseEnvironment(num_pits=6, num_stones=4)
        env.reset()

        # Empty player 0's pits
        for i in range(env.num_pits):
            env.board[i] = 0

        # Put some stones in player 0's store
        env.board[env.num_pits] = 10

        # Make sure player 1's pits have the correct number of stones
        for i in range(env.num_pits + 1, 2 * env.num_pits + 1):
            env.board[i] = env.num_stones

        # Check game end
        env._check_game_end()

        # The game should be done
        assert env.done

        # Player 1's stones should have been moved to their store
        assert env.board[2 * env.num_pits + 1] == env.num_pits * env.num_stones

        # Player 1's pits should be empty
        for i in range(env.num_pits + 1, 2 * env.num_pits + 1):
            assert env.board[i] == 0

    def test_get_winner(self):
        """Test that get_winner returns the correct winner."""
        # Create an environment with a specific state
        env = BaseEnvironment(num_pits=6, num_stones=4)
        env.reset()

        # Game is not done, so there should be no winner
        assert env.get_winner() is None

        # End the game with player 0 having more stones
        env.done = True
        env.board[env.num_pits] = 25  # Player 0's store
        env.board[2 * env.num_pits + 1] = 23  # Player 1's store

        # Player 0 should be the winner
        assert env.get_winner() == 0

        # End the game with player 1 having more stones
        env.board[env.num_pits] = 23  # Player 0's store
        env.board[2 * env.num_pits + 1] = 25  # Player 1's store

        # Player 1 should be the winner
        assert env.get_winner() == 1

        # End the game with a draw
        env.board[env.num_pits] = 24  # Player 0's store
        env.board[2 * env.num_pits + 1] = 24  # Player 1's store

        # It should be a draw
        assert env.get_winner() is None
