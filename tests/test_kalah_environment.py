"""Tests for the KalahEnvironment class.

This module contains tests for the KalahEnvironment class, which implements
the Kalah variant of Mancala with specific rules such as capturing and extra turns.
"""

import pytest
import numpy as np

from mancala_ai.environments.kalah_environment import KalahEnvironment


class TestKalahEnvironment:
    """Tests for the KalahEnvironment class."""

    def test_initialization(self, kalah_environment):
        """Test that the Kalah environment initializes correctly."""
        # Check that the board has the correct shape
        assert len(kalah_environment.board) == 2 * (kalah_environment.num_pits + 1)

        # Check that player 0's pits have the correct number of stones
        for i in range(kalah_environment.num_pits):
            assert kalah_environment.board[i] == kalah_environment.num_stones

        # Check that player 1's pits have the correct number of stones
        for i in range(kalah_environment.num_pits + 1, 2 * kalah_environment.num_pits + 1):
            assert kalah_environment.board[i] == kalah_environment.num_stones

        # Check that the stores are empty
        assert kalah_environment.board[kalah_environment.num_pits] == 0
        assert kalah_environment.board[2 * kalah_environment.num_pits + 1] == 0

        # Check that player 0 starts
        assert kalah_environment.current_player == 0

        # Check that the game is not done
        assert not kalah_environment.done

    def test_extra_turn(self):
        """Test that a player gets an extra turn when the last stone lands in their store."""
        # Create a specific board state where the next move will result in an extra turn
        env = KalahEnvironment(num_pits=6, num_stones=4)
        env.reset()

        # Set up a state where player 0 has 1 stone in pit 5 (index 5)
        # This will land in their store and give them an extra turn
        env.board = np.zeros(2 * (env.num_pits + 1), dtype=np.int32)
        # Add stones to player 0's pits to prevent the game from ending
        for i in range(env.num_pits):
            if i != 5:  # Skip pit 5, we'll set it separately
                env.board[i] = 1
        env.board[5] = 1  # Player 0's last pit has 1 stone
        # Add some stones to player 1's pits to prevent the game from ending
        for i in range(env.num_pits + 1, 2 * env.num_pits + 1):
            env.board[i] = 1
        env.current_player = 0

        # Take the action
        observation, reward, terminated, truncated, info = env.step(5)

        # Check that the stone is now in player 0's store
        assert env.board[env.num_pits] == 1
        assert env.board[5] == 0

        # Check that player 0 still has the turn (extra turn)
        assert env.current_player == 0

        # Check that the game is not done
        assert not terminated

    def test_capture(self):
        """Test that stones are captured correctly when the last stone lands in an empty pit."""
        # Create a specific board state where the next move will result in a capture
        env = KalahEnvironment(num_pits=6, num_stones=4)
        env.reset()

        # Set up a state where player 0 has 1 stone in pit 2 (index 2)
        # and player 1 has 4 stones in the opposite pit (index 9)
        env.board = np.zeros(2 * (env.num_pits + 1), dtype=np.int32)
        env.board[2] = 1  # Player 0's pit has 1 stone
        env.board[9] = 4  # Player 1's opposite pit has 4 stones
        env.current_player = 0

        # Take the action
        observation, reward, terminated, truncated, info = env.step(2)

        # Check that both pits are now empty
        assert env.board[2] == 0
        assert env.board[9] == 0

        # Check that player 0's store has 5 stones (1 + 4)
        assert env.board[env.num_pits] == 5

        # Check that player 1's turn is next
        assert env.current_player == 1

    def test_no_capture_on_non_empty_pit(self):
        """Test that stones are not captured when the last stone lands in a non-empty pit."""
        # Create a specific board state where the last stone lands in a non-empty pit
        env = KalahEnvironment(num_pits=6, num_stones=4)
        env.reset()

        # Set up a state where player 0 has 2 stones in pit 2 (index 2)
        # and player 1 has 4 stones in the opposite pit (index 9)
        env.board = np.zeros(2 * (env.num_pits + 1), dtype=np.int32)
        env.board[2] = 2  # Player 0's pit has 2 stones
        env.board[9] = 4  # Player 1's opposite pit has 4 stones
        # Add a stone to pit 4 to make it non-empty (this is where the last stone will land)
        env.board[4] = 1
        env.current_player = 0

        # Take the action
        observation, reward, terminated, truncated, info = env.step(2)

        # Check that player 0's original pit is now empty
        assert env.board[2] == 0

        # Check that the last stone landed in pit 4, which now has 2 stones (1 + 1)
        assert env.board[4] == 2

        # Check that player 1's opposite pit still has 4 stones (no capture)
        assert env.board[9] == 4

        # Check that player 0's store has 0 stones (no stones were distributed to the store)
        assert env.board[env.num_pits] == 0

        # Check that player 1's turn is next
        assert env.current_player == 1

    def test_skip_opponent_store(self):
        """Test that stones skip the opponent's store during distribution."""
        # Create a specific board state where stones would pass through the opponent's store
        env = KalahEnvironment(num_pits=6, num_stones=4)
        env.reset()

        # Set up a state where player 0 has 8 stones in pit 5 (index 5)
        # This will distribute stones past player 1's store
        env.board = np.zeros(2 * (env.num_pits + 1), dtype=np.int32)
        env.board[5] = 8  # Player 0's last pit has 8 stones
        env.current_player = 0

        # Take the action
        observation, reward, terminated, truncated, info = env.step(5)

        # Check that player 0's store has 3 stones
        assert env.board[env.num_pits] == 3

        # Check that player 1's store has 5 stones
        assert env.board[2 * env.num_pits + 1] == 5

        # Check that the stones were distributed correctly
        # (7 stones should be distributed to player 1's pits and back to player 0's pits)
        assert env.board[5] == 0  # Original pit is empty
        assert np.sum(env.board) == 8  # Total stones is preserved

        # Check that player 1's turn is next
        assert env.current_player == 1

    def test_game_end_and_scoring(self):
        """Test that the game ends correctly and final scoring is applied."""
        # Create a specific board state where player 0's pits are almost empty
        env = KalahEnvironment(num_pits=6, num_stones=4)
        env.reset()

        # Set up a state where player 0 has 1 stone in pit 0 and the rest are empty
        # Player 1 has stones in all pits
        env.board = np.zeros(2 * (env.num_pits + 1), dtype=np.int32)
        env.board[0] = 1  # Player 0's first pit has 1 stone
        for i in range(env.num_pits + 1, 2 * env.num_pits + 1):
            env.board[i] = 4  # Player 1's pits have 4 stones each

        env.current_player = 0

        # Take the action that will empty player 0's pits
        observation, reward, terminated, truncated, info = env.step(0)

        # Check that the game is done
        assert terminated

        # Check that all of player 1's stones have been moved to their store
        assert env.board[2 * env.num_pits + 1] == 20  # 5 pits * 4 stones = 20 stones

        # Check that all pits are empty
        for i in range(env.num_pits):
            assert env.board[i] == 0
        for i in range(env.num_pits + 1, 2 * env.num_pits + 1):
            assert env.board[i] == 0

        # Check that player 1 is the winner (they have more stones)
        assert env.get_winner() == 1
