"""Base environment for Mancala game.

This module provides a base class for implementing Mancala game environments.
It follows the OpenAI Gym interface pattern and can be extended to implement
specific variants of Mancala, such as Kalah.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class BaseEnvironment:
    """Base class for Mancala game environments.

    This class implements the core mechanics of a Mancala game, following
    the OpenAI Gym interface pattern. It provides methods for resetting the
    environment, taking steps (making moves), and rendering the current state.

    Attributes:
        num_pits (int): Number of pits per player (excluding stores).
        num_stones (int): Initial number of stones per pit.
        board (np.ndarray): Game board representation.
        current_player (int): Current player (0 or 1).
        done (bool): Whether the game is finished.
    """

    DISPLAY_NAME = "Mancala Game"  # Default display name, should be overridden by subclasses

    def __init__(self, num_pits: int = 6, num_stones: int = 4):
        """Initialize the Mancala environment.

        Args:
            num_pits: Number of pits per player (excluding stores).
            num_stones: Initial number of stones per pit.
        """
        self.num_pits = num_pits
        self.num_stones = num_stones

        # Board representation:
        # [p0_pit0, p0_pit1, ..., p0_store, p1_pit0, p1_pit1, ..., p1_store]
        self.board = None
        self.current_player = None
        self.done = None

        # Initialize the game
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the environment to the initial state.

        Returns:
            The initial observation of the environment.
        """
        # Initialize the board with num_stones in each pit and 0 in stores
        self.board = np.zeros(2 * (self.num_pits + 1), dtype=np.int32)

        # Fill pits with initial stones (excluding stores)
        for i in range(2 * (self.num_pits + 1)):
            if i != self.num_pits and i != 2 * self.num_pits + 1:
                self.board[i] = self.num_stones

        # Player 0 starts
        self.current_player = 0
        self.done = False

        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment by making a move.

        Args:
            action: The pit index to select (0 to num_pits-1).

        Returns:
            A tuple containing:
                - observation: The current state of the environment.
                - reward: The reward for taking the action.
                - terminated: Whether the episode is terminated.
                - truncated: Whether the episode is truncated.
                - info: Additional information.
        """
        if self.done:
            return self._get_observation(), 0.0, True, False, {"error": "Game already finished"}

        # Validate action
        if not self._is_valid_action(action):
            return self._get_observation(), -5.0, False, False, {"error": "Invalid action"}

        # Execute the move
        reward = self._execute_move(action)

        # Check if the game is finished
        self._check_game_end()

        # Add terminal rewards if the game is finished
        if self.done:
            winner = self.get_winner()
            if winner is not None:
                # Positive reward for winning, negative for losing
                terminal_reward = 100.0 if winner == self.current_player else -100.0
                reward += terminal_reward
            else:
                # Small positive reward for a draw
                reward += 10.0

        # Add a small step penalty to encourage faster solutions
        reward -= 0.1

        # Return normalized reward
        return self._get_observation(), reward, self.done, False, {}

    def render(self) -> None:
        """Render the current state of the environment."""
        # Player 1's pits (reversed)
        p1_pits = self.board[self.num_pits + 1:2 * self.num_pits + 1]
        p1_store = self.board[2 * self.num_pits + 1]

        # Player 0's pits
        p0_pits = self.board[:self.num_pits]
        p0_store = self.board[self.num_pits]

        # Print the board
        print("\nCurrent board state:")
        print("    " + " ".join(f"{pit:2d}" for pit in reversed(p1_pits)))
        print(f"{p1_store:2d}" + " " * (4 * self.num_pits) + f" {p0_store:2d}")
        print("    " + " ".join(f"{pit:2d}" for pit in p0_pits))
        print(f"\nPlayer {self.current_player + 1}'s turn")

    def _get_observation(self) -> np.ndarray:
        """Get the current observation of the environment.

        Returns:
            The current state of the board.
        """
        return self.board.copy()

    def _is_valid_action(self, action: int) -> bool:
        """Check if an action is valid.

        Args:
            action: The pit index to select (0 to num_pits-1).

        Returns:
            Whether the action is valid.
        """
        # Action must be within range and the selected pit must have stones
        if action < 0 or action >= self.num_pits:
            return False

        # Calculate the actual index in the board array based on current player
        board_idx = action if self.current_player == 0 else self.num_pits + 1 + action

        # The pit must have stones
        return self.board[board_idx] > 0

    def _execute_move(self, action: int) -> float:
        """Execute a move in the game.

        Args:
            action: The pit index to select (0 to num_pits-1).

        Returns:
            The reward for the move.
        """
        # This is a base implementation that will be overridden by specific game variants
        # In the base implementation, we just distribute stones counter-clockwise

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

        # Switch player (in base implementation, always switch)
        # Specific variants might have rules for extra turns
        original_player = self.current_player
        self.current_player = 1 - self.current_player

        # Calculate reward components
        reward = 0.0

        # 1. Reward for stones added to the player's store
        stones_added = self.board[player_store] - initial_store_count
        reward += stones_added * 1.0  # Base reward for each stone added

        # 2. Reward for ending in the player's store (would give an extra turn in some variants)
        if last_idx == player_store:
            reward += 2.0  # Bonus for ending in store

        # 3. Reward for distributing many stones (encourages choosing pits with more stones)
        reward += min(stones, 5) * 0.2  # Small bonus based on number of stones moved, capped at 5

        # 4. Reward for strategic positioning (having stones in pits close to the store)
        player_side_start = 0 if original_player == 0 else self.num_pits + 1
        player_side_end = self.num_pits - 1 if original_player == 0 else 2 * self.num_pits

        position_reward = 0.0
        for i in range(player_side_start, player_side_end + 1):
            # Weight positions closer to the store more heavily
            position_weight = (i - player_side_start + 1) / (player_side_end - player_side_start + 1)
            position_reward += self.board[i] * position_weight * 0.1

        reward += position_reward

        return reward

    def _check_game_end(self) -> None:
        """Check if the game has ended and update the done flag."""
        # Check if all pits of either player are empty
        p0_pits_empty = all(self.board[i] == 0 for i in range(self.num_pits))
        p1_pits_empty = all(self.board[i] == 0 for i in range(self.num_pits + 1, 2 * self.num_pits + 1))

        if p0_pits_empty or p1_pits_empty:
            self.done = True

            # In most Mancala variants, remaining stones go to the respective player's store
            if p0_pits_empty:
                # Move all stones from player 1's pits to their store
                for i in range(self.num_pits + 1, 2 * self.num_pits + 1):
                    self.board[2 * self.num_pits + 1] += self.board[i]
                    self.board[i] = 0
            else:  # p1_pits_empty
                # Move all stones from player 0's pits to their store
                for i in range(self.num_pits):
                    self.board[self.num_pits] += self.board[i]
                    self.board[i] = 0

    def get_valid_actions(self) -> List[int]:
        """Get a list of valid actions for the current player.

        Returns:
            A list of valid pit indices.
        """
        valid_actions = []
        for action in range(self.num_pits):
            if self._is_valid_action(action):
                valid_actions.append(action)
        return valid_actions

    def get_winner(self) -> Optional[int]:
        """Get the winner of the game.

        Returns:
            The player index (0 or 1) of the winner, or None if the game is not finished
            or it's a draw.
        """
        if not self.done:
            return None

        p0_score = self.board[self.num_pits]
        p1_score = self.board[2 * self.num_pits + 1]

        if p0_score > p1_score:
            return 0
        elif p1_score > p0_score:
            return 1
        else:
            return None  # Draw
