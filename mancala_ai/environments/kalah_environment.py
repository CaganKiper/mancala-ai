"""Kalah environment for Mancala game.

This module provides a class for implementing the Kalah variant of Mancala game.
It extends the BaseEnvironment class and overrides methods to implement
Kalah-specific rules.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from mancala_ai.environments.base_environment import BaseEnvironment


class KalahEnvironment(BaseEnvironment):
    """Class for Kalah variant of Mancala game.

    This class extends the BaseEnvironment class and implements the Kalah-specific
    rules, such as capturing and extra turns.

    Attributes:
        num_pits (int): Number of pits per player (excluding stores).
        num_stones (int): Initial number of stones per pit.
        board (np.ndarray): Game board representation.
        current_player (int): Current player (0 or 1).
        done (bool): Whether the game is finished.
    """

    DISPLAY_NAME = "Kalah"  # Custom display name for this environment

    def _execute_move(self, action: int) -> float:
        """Execute a move in the Kalah game.

        Args:
            action: The pit index to select (0 to num_pits-1).

        Returns:
            The reward for the move.
        """
        # Calculate the actual index in the board array based on current player
        board_idx = action if self.current_player == 0 else self.num_pits + 1 + action

        # Get the initial state of the player's store for reward calculation
        player_store = self.num_pits if self.current_player == 0 else 2 * self.num_pits + 1
        initial_store_count = self.board[player_store]

        # Pick up stones
        stones = self.board[board_idx]
        self.board[board_idx] = 0

        # Distribute stones
        current_idx = board_idx
        last_idx = None
        while stones > 0:
            current_idx = (current_idx + 1) % len(self.board)

            # Skip opponent's store
            opponent_store = 2 * self.num_pits + 1 if self.current_player == 0 else self.num_pits
            if current_idx == opponent_store:
                continue

            # Place a stone
            self.board[current_idx] += 1
            stones -= 1

            # Remember the last index where a stone was placed
            last_idx = current_idx

        # Check for capture: if the last stone was placed in an empty pit on the player's side
        player_side_start = 0 if self.current_player == 0 else self.num_pits + 1
        player_side_end = self.num_pits - 1 if self.current_player == 0 else 2 * self.num_pits
        player_store = self.num_pits if self.current_player == 0 else 2 * self.num_pits + 1

        # Track capture information for reward calculation
        captured_stones = 0
        capture_occurred = False

        if (last_idx >= player_side_start and last_idx <= player_side_end and 
            self.board[last_idx] == 1):
            # Calculate the opposite pit index
            opposite_idx = 2 * self.num_pits - last_idx

            # Capture stones from the opposite pit and the last stone
            if self.board[opposite_idx] > 0:
                captured_stones = self.board[opposite_idx] + self.board[last_idx]
                self.board[player_store] += captured_stones
                self.board[opposite_idx] = 0
                self.board[last_idx] = 0
                capture_occurred = True

        # Track if an extra turn occurs
        extra_turn = False

        # Check for extra turn: if the last stone was placed in the player's store
        if last_idx == player_store:
            # Player gets an extra turn, so don't switch players
            extra_turn = True
        else:
            # Switch player
            self.current_player = 1 - self.current_player

        # Calculate reward components
        reward = 0.0

        # 1. Reward for stones added to the player's store
        stones_added = self.board[player_store] - initial_store_count
        reward += stones_added * 1.0  # Base reward for each stone added

        # 2. Bonus reward for capturing stones
        if capture_occurred:
            reward += captured_stones * 1.5  # Higher reward for capturing (1.5 per stone)

        # 3. Bonus reward for getting an extra turn
        if extra_turn:
            reward += 3.0  # Significant bonus for extra turn

        # 4. Reward for distributing many stones (encourages choosing pits with more stones)
        original_stones = self.board[board_idx] + stones_added  # Reconstruct original stone count
        reward += min(original_stones, 5) * 0.2  # Small bonus based on number of stones moved, capped at 5

        # 5. Reward for strategic positioning (having stones in pits close to the store)
        original_player = 0 if player_store == self.num_pits else 1
        player_side_start = 0 if original_player == 0 else self.num_pits + 1
        player_side_end = self.num_pits - 1 if original_player == 0 else 2 * self.num_pits

        position_reward = 0.0
        for i in range(player_side_start, player_side_end + 1):
            # Weight positions closer to the store more heavily
            position_weight = (i - player_side_start + 1) / (player_side_end - player_side_start + 1)
            position_reward += self.board[i] * position_weight * 0.1

        reward += position_reward

        # 6. Reward for emptying pits on player's side (approaching game end in a good position)
        empty_pits = sum(1 for i in range(player_side_start, player_side_end + 1) if self.board[i] == 0)
        reward += empty_pits * 0.3  # Small bonus for each empty pit

        return reward
