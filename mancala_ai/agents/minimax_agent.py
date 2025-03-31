"""Minimax agent class for Mancala games.

This module provides the minimax agent class for Mancala games, which uses the
Minimax algorithm with Alpha-Beta pruning to select actions.
"""

import copy
import hashlib
from typing import List, Tuple, Optional, Dict, Any, Hashable

import numpy as np

from mancala_ai.agents.base_agent import Agent
from mancala_ai.environments.base_environment import BaseEnvironment


class MinimaxAgent(Agent):
    """AI agent that uses the Minimax algorithm with several optimizations to select actions.

    The implementation includes the following optimizations:

    1. Alpha-Beta pruning: Reduces the number of nodes evaluated in the search tree by
       stopping evaluation of a move when it is determined to be worse than a previously
       examined move.

    2. Move ordering: Examines the most promising moves first to increase the likelihood
       of early cutoffs in alpha-beta pruning. The heuristic prioritizes moves that lead
       to extra turns, captures, and moves with more stones.

    3. Transposition table: Stores previously evaluated positions to avoid re-computing
       the same positions multiple times, significantly reducing computation time for
       positions that can be reached through different move sequences.
    """

    DISPLAY_NAME = "Minimax Agent"  # Custom display name for this agent

    def __init__(self, name: str, depth: int = 3):
        """Initialize a minimax agent.

        Args:
            name: The name of the agent.
            depth: The maximum depth to search in the game tree.
        """
        super().__init__(name)
        self.depth = depth
        # Transposition table to store evaluated positions
        self.transposition_table = {}

    def get_settings(self) -> Dict[str, Tuple[Any, str, str]]:
        """Get the agent's configurable settings.

        Returns:
            A dictionary mapping setting names to tuples of 
            (current_value, description, type).
        """
        return {
            "depth": (self.depth, "Maximum depth to search in the game tree", "int")
        }

    def set_setting(self, setting_name: str, value: Any) -> bool:
        """Set a specific setting to a new value.

        Args:
            setting_name: The name of the setting to change.
            value: The new value for the setting.

        Returns:
            True if the setting was successfully updated, False otherwise.
        """
        if setting_name == "depth":
            try:
                depth = int(value)
                if depth > 0:
                    self.depth = depth
                    return True
                else:
                    print("Depth must be a positive integer.")
                    return False
            except ValueError:
                print("Depth must be a valid integer.")
                return False
        return False

    def _get_board_hash(self, env: BaseEnvironment) -> str:
        """Generate a hash key for the current board state.

        Args:
            env: The game environment.

        Returns:
            A string hash representing the board state.
        """
        # Create a string representation of the board and current player
        board_str = np.array2string(env.board) + str(env.current_player)

        # Generate a hash of the string
        return hashlib.md5(board_str.encode()).hexdigest()

    def get_action(self, env: BaseEnvironment) -> int:
        """Get the best action according to the Minimax algorithm.

        Args:
            env: The game environment.

        Returns:
            The selected action.
        """
        # Clear the transposition table for a new search
        self.transposition_table = {}

        valid_actions = env.get_valid_actions()

        if not valid_actions:
            return -1

        # Create a deep copy of the environment to avoid modifying the original
        env_copy = copy.deepcopy(env)

        # Find the best action using minimax with alpha-beta pruning
        best_action, _ = self._minimax(env_copy, self.depth, True, float('-inf'), float('inf'))
        return best_action

    def _order_moves(self, env: BaseEnvironment, valid_actions: List[int]) -> List[int]:
        """Order moves based on a simple heuristic for better alpha-beta pruning efficiency.

        This method orders moves to increase the likelihood of early cutoffs in alpha-beta pruning.
        The heuristic prioritizes:
        1. Moves that lead to an extra turn
        2. Moves that capture opponent's stones
        3. Moves with more stones (which generally have more impact)

        Args:
            env: The game environment.
            valid_actions: List of valid actions to order.

        Returns:
            Ordered list of actions.
        """
        player = env.current_player
        player_store = env.num_pits if player == 0 else 2 * env.num_pits + 1

        # Create a list of (action, score) tuples for sorting
        action_scores = []

        for action in valid_actions:
            # Calculate the actual index in the board array
            board_idx = action if player == 0 else env.num_pits + 1 + action
            stones = env.board[board_idx]

            # Base score is the number of stones (more stones generally have more impact)
            score = stones

            # Check if this move leads to an extra turn (landing in the player's store)
            landing_idx = (board_idx + stones) % len(env.board)
            if landing_idx == player_store:
                # Heavily prioritize moves that give an extra turn
                score += 100

            # Check if this move leads to a capture
            player_side_start = 0 if player == 0 else env.num_pits + 1
            player_side_end = env.num_pits - 1 if player == 0 else 2 * env.num_pits

            if (landing_idx >= player_side_start and 
                landing_idx <= player_side_end and 
                env.board[landing_idx] == 0):

                # Calculate the opposite pit index
                opposite_idx = 2 * env.num_pits - landing_idx

                # If the opposite pit has stones, it's a potential capture
                if env.board[opposite_idx] > 0:
                    # Prioritize captures based on the number of stones that would be captured
                    score += 50 + env.board[opposite_idx]

            action_scores.append((action, score))

        # Sort actions by score in descending order
        action_scores.sort(key=lambda x: x[1], reverse=True)

        # Return just the ordered actions
        return [action for action, _ in action_scores]

    def _minimax(self, env: BaseEnvironment, depth: int, maximizing_player: bool, 
              alpha: float = float('-inf'), beta: float = float('inf')) -> Tuple[Optional[int], float]:
        """Recursive minimax algorithm implementation with alpha-beta pruning.

        Args:
            env: The game environment.
            depth: The current depth in the search tree.
            maximizing_player: Whether the current player is maximizing or minimizing.
            alpha: The best value that the maximizing player currently can guarantee.
            beta: The best value that the minimizing player currently can guarantee.

        Returns:
            A tuple containing:
                - The best action (or None if no valid actions or terminal state).
                - The value of the best action.
        """
        # Check if we've reached a terminal state or maximum depth
        if depth == 0 or env.done:
            return None, self._evaluate(env)

        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return None, self._evaluate(env)

        # Generate a hash key for the current board state
        board_hash = self._get_board_hash(env)

        # Check if this position has already been evaluated
        tt_entry = self.transposition_table.get((board_hash, depth, maximizing_player))
        if tt_entry is not None:
            # If we have a stored value for this exact position, depth, and player
            stored_action, stored_value, stored_alpha, stored_beta = tt_entry

            # If the stored bounds are valid for the current search window
            if stored_alpha <= alpha and stored_beta >= beta:
                return stored_action, stored_value

        # Initialize best action and value
        best_action = None
        if maximizing_player:
            best_value = float('-inf')
        else:
            best_value = float('inf')

        # Order moves for better alpha-beta pruning efficiency
        ordered_actions = self._order_moves(env, valid_actions)

        # Try each valid action
        for action in ordered_actions:
            # Create a deep copy of the environment
            env_copy = copy.deepcopy(env)

            # Take the action
            original_player = env_copy.current_player
            _, reward, terminated, _, _ = env_copy.step(action)

            # Check if the player gets an extra turn (current player didn't change)
            extra_turn = not terminated and env_copy.current_player == original_player

            # If the player gets an extra turn, it's still their turn in the minimax tree
            next_maximizing = maximizing_player if extra_turn else not maximizing_player

            # Recursively evaluate the resulting state
            _, value = self._minimax(env_copy, depth - 1, next_maximizing, alpha, beta)

            # Update best action and value
            if maximizing_player:
                if value > best_value:
                    best_value = value
                    best_action = action
                alpha = max(alpha, best_value)
                # Beta cutoff
                if alpha >= beta:
                    break
            else:
                if value < best_value:
                    best_value = value
                    best_action = action
                beta = min(beta, best_value)
                # Alpha cutoff
                if beta <= alpha:
                    break

        # Store the result in the transposition table
        self.transposition_table[(board_hash, depth, maximizing_player)] = (best_action, best_value, alpha, beta)

        return best_action, best_value

    def _evaluate(self, env: BaseEnvironment) -> float:
        """Evaluate the current state of the environment.

        Args:
            env: The game environment.

        Returns:
            A value representing how good the state is for the current player.
        """
        # If the game is over, return a large positive or negative value
        if env.done:
            winner = env.get_winner()
            if winner is None:  # Draw
                return 0.0
            elif winner == 0:  # Player 0 wins
                return 1000.0
            else:  # Player 1 wins
                return -1000.0

        # Get the basic evaluation based on stones in stores
        player0_store = env.board[env.num_pits]
        player1_store = env.board[2 * env.num_pits + 1]

        # Calculate the basic score difference
        if env.current_player == 0:
            score_diff = player0_store - player1_store
        else:
            score_diff = player1_store - player0_store

        # Initialize the evaluation with the score difference
        evaluation = score_diff

        # Add evaluation for potential captures
        evaluation += self._evaluate_potential_captures(env)

        # Add evaluation for potential extra turns
        evaluation += self._evaluate_potential_extra_turns(env)

        # Add evaluation for stone distribution
        evaluation += self._evaluate_stone_distribution(env)

        # Add evaluation for empty pits
        evaluation += self._evaluate_empty_pits(env)

        return evaluation

    def _evaluate_potential_captures(self, env: BaseEnvironment) -> float:
        """Evaluate the potential for capturing opponent's stones.

        Args:
            env: The game environment.

        Returns:
            A value representing the potential for capturing.
        """
        # Initialize the evaluation
        evaluation = 0.0

        # Get the player's side range
        player = env.current_player
        player_side_start = 0 if player == 0 else env.num_pits + 1
        player_side_end = env.num_pits - 1 if player == 0 else 2 * env.num_pits

        # Get the opponent's side range
        opponent_side_start = env.num_pits + 1 if player == 0 else 0
        opponent_side_end = 2 * env.num_pits if player == 0 else env.num_pits - 1

        # Check each pit on the player's side
        for i in range(player_side_start, player_side_end + 1):
            # If the pit is empty, it's a potential landing spot for a capture
            if env.board[i] == 0:
                # Calculate the opposite pit index
                opposite_idx = 2 * env.num_pits - i

                # If the opposite pit has stones, it's a potential capture
                if env.board[opposite_idx] > 0:
                    # Add a negative evaluation to avoid moves that create capture opportunities for the opponent
                    evaluation -= env.board[opposite_idx] * 1.5

            # Check if this pit can lead to a capture
            stones = env.board[i]
            if stones > 0:
                # Calculate where the last stone would land
                landing_idx = (i + stones) % len(env.board)

                # Skip if it would land in the opponent's store
                opponent_store = 2 * env.num_pits + 1 if player == 0 else env.num_pits
                if landing_idx == opponent_store:
                    continue

                # If it would land in an empty pit on the player's side
                if (landing_idx >= player_side_start and 
                    landing_idx <= player_side_end and 
                    env.board[landing_idx] == 0):

                    # Calculate the opposite pit index
                    opposite_idx = 2 * env.num_pits - landing_idx

                    # If the opposite pit has stones, it's a potential capture
                    if env.board[opposite_idx] > 0:
                        # Add a positive evaluation for potential captures
                        evaluation += (env.board[opposite_idx] + 1) * 1.5

        return evaluation

    def _evaluate_potential_extra_turns(self, env: BaseEnvironment) -> float:
        """Evaluate the potential for getting an extra turn.

        Args:
            env: The game environment.

        Returns:
            A value representing the potential for getting an extra turn.
        """
        # Initialize the evaluation
        evaluation = 0.0

        # Get the player's store index
        player = env.current_player
        player_store = env.num_pits if player == 0 else 2 * env.num_pits + 1

        # Get the player's side range
        player_side_start = 0 if player == 0 else env.num_pits + 1
        player_side_end = env.num_pits - 1 if player == 0 else 2 * env.num_pits

        # Check each pit on the player's side
        for i in range(player_side_start, player_side_end + 1):
            # Check if this pit can lead to an extra turn
            stones = env.board[i]
            if stones > 0:
                # Calculate where the last stone would land
                landing_idx = (i + stones) % len(env.board)

                # If it would land in the player's store, it's an extra turn
                if landing_idx == player_store:
                    # Add a positive evaluation for potential extra turns
                    evaluation += 3.0

        return evaluation

    def _evaluate_stone_distribution(self, env: BaseEnvironment) -> float:
        """Evaluate the distribution of stones on the board.

        Args:
            env: The game environment.

        Returns:
            A value representing the quality of stone distribution.
        """
        # Initialize the evaluation
        evaluation = 0.0

        # Get the player's side range
        player = env.current_player
        player_side_start = 0 if player == 0 else env.num_pits + 1
        player_side_end = env.num_pits - 1 if player == 0 else 2 * env.num_pits

        # Evaluate the distribution of stones on the player's side
        for i in range(player_side_start, player_side_end + 1):
            # Weight positions closer to the store more heavily
            position_weight = (i - player_side_start + 1) / (player_side_end - player_side_start + 1)
            evaluation += env.board[i] * position_weight * 0.1

        return evaluation

    def _evaluate_empty_pits(self, env: BaseEnvironment) -> float:
        """Evaluate the number of empty pits.

        Args:
            env: The game environment.

        Returns:
            A value representing the quality of empty pits.
        """
        # Initialize the evaluation
        evaluation = 0.0

        # Get the player's side range
        player = env.current_player
        player_side_start = 0 if player == 0 else env.num_pits + 1
        player_side_end = env.num_pits - 1 if player == 0 else 2 * env.num_pits

        # Get the opponent's side range
        opponent_side_start = env.num_pits + 1 if player == 0 else 0
        opponent_side_end = 2 * env.num_pits if player == 0 else env.num_pits - 1

        # Count empty pits on the player's side
        player_empty_pits = sum(1 for i in range(player_side_start, player_side_end + 1) if env.board[i] == 0)

        # Count empty pits on the opponent's side
        opponent_empty_pits = sum(1 for i in range(opponent_side_start, opponent_side_end + 1) if env.board[i] == 0)

        # Add a positive evaluation for empty pits on the player's side (approaching game end in a good position)
        evaluation += player_empty_pits * 0.3

        # Add a negative evaluation for empty pits on the opponent's side
        evaluation -= opponent_empty_pits * 0.3

        return evaluation
