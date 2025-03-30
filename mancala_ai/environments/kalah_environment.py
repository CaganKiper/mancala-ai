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

    def _execute_move(self, action: int) -> float:
        """Execute a move in the Kalah game.

        Args:
            action: The pit index to select (0 to num_pits-1).

        Returns:
            The reward for the move.
        """
        # Calculate the actual index in the board array based on current player
        board_idx = action if self.current_player == 0 else self.num_pits + 1 + action
        
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
        
        if (last_idx >= player_side_start and last_idx <= player_side_end and 
            self.board[last_idx] == 1):
            # Calculate the opposite pit index
            opposite_idx = 2 * self.num_pits - last_idx
            
            # Capture stones from the opposite pit and the last stone
            if self.board[opposite_idx] > 0:
                self.board[player_store] += self.board[opposite_idx] + self.board[last_idx]
                self.board[opposite_idx] = 0
                self.board[last_idx] = 0
        
        # Check for extra turn: if the last stone was placed in the player's store
        if last_idx == player_store:
            # Player gets an extra turn, so don't switch players
            pass
        else:
            # Switch player
            self.current_player = 1 - self.current_player
        
        # Basic reward: number of stones in player's store
        return float(self.board[player_store])