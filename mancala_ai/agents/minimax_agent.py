"""Enhanced Minimax agent with specialized Mancala strategies.

This version incorporates sophisticated Mancala-specific knowledge with
significantly improved endgame handling and chain-move detection.
"""

import copy
import time
from typing import List, Tuple, Optional, Dict, Any, Set

import numpy as np

from mancala_ai.agents.base_agent import Agent
from mancala_ai.environments.base_environment import BaseEnvironment


class MinimaxAgent(Agent):
    """AI agent that uses an optimized Minimax algorithm with Mancala-specific enhancements.

    Strategic improvements:
    1. Chain Move Analysis: Identifies multi-move sequences that lead to multiple consecutive turns
    2. Endgame Database: Special handling for near-empty board states
    3. Pattern Recognition: Evaluates common Mancala patterns and traps
    4. Advanced Capture Evaluation: Prioritizes key captures and protects against opponent captures
    5. Dynamic Depth Management: Searches critical positions deeper
    """

    DISPLAY_NAME = "Minimax Agent"  # Keep the original display name

    # Phase definitions with more precise boundaries
    PHASE_OPENING = 0    # Many stones distributed relatively evenly
    PHASE_MIDGAME = 1    # Some empty pits but still good distribution
    PHASE_ENDGAME = 2    # Several empty pits, focus on captures and clearing
    PHASE_FINAL = 3      # One or both sides nearly empty, requires precise calculation

    # Stone count thresholds for phases
    ENDGAME_THRESHOLD = 0.4  # When 40% of pits are empty
    FINAL_THRESHOLD = 0.7    # When 70% of pits are empty

    def __init__(self, name: str, depth: int = 7, time_limit: float = 5.0,
                 tt_size: int = 500000):
        """Initialize a minimax agent.

        Args:
            name: The name of the agent.
            depth: The default maximum depth to search in the game tree.
            time_limit: Maximum time in seconds to spend on a move.
            tt_size: Maximum size of transposition table.
        """
        super().__init__(name)
        self.depth = depth  # Keep original parameter name for compatibility
        self.max_depth = depth  # Rename internally for clarity
        self.time_limit = time_limit
        self.tt_size = tt_size

        # Transposition table with size limit
        self.transposition_table = {}
        self.tt_keys_queue = []  # For LRU replacement strategy

        # Zobrist hashing tables
        self.zobrist_table = None
        self.zobrist_turn = None

        # Search enhancement data structures
        self.killer_moves = [[None, None] for _ in range(self.max_depth + 1)]
        self.history_table = {}  # Tracks effectiveness of moves

        # Pattern database for common Mancala patterns
        self.pattern_database = self._init_pattern_database()

        # Chain move sequences for extended planning
        self.chain_sequences = {}  # Stores promising chain move sequences

        # Performance metrics
        self.nodes_searched = 0
        self.tt_hits = 0
        self.ab_cutoffs = 0
        self.q_searches = 0

        # Time management
        self.start_time = 0
        self.time_up = False

        # Debug info
        self.debug_mode = False
        self.eval_components = {}

    def _init_pattern_database(self) -> Dict:
        """Initialize a database of common beneficial Mancala patterns.

        Returns:
            A dictionary of pattern templates and their values.
        """
        # Simple pattern database with some common patterns
        # Format: {pattern_name: (pattern_template, value)}
        # Pattern templates are functions that check for the pattern
        return {
            "store_sweep": self._pattern_store_sweep,
            "chain_setup": self._pattern_chain_setup,
            "trap_setup": self._pattern_trap_setup,
            "safe_distribution": self._pattern_safe_distribution,
        }

    def _pattern_store_sweep(self, env: BaseEnvironment, player: int) -> float:
        """Detect and evaluate the store sweep pattern.

        This pattern occurs when a player has stones distributed in a way
        that allows capturing multiple opponent stones in one sequence.

        Args:
            env: The game environment.
            player: The player to check for (0 or 1).

        Returns:
            A score value for the detected pattern.
        """
        player_side_start = 0 if player == 0 else env.num_pits + 1
        player_side_end = env.num_pits - 1 if player == 0 else 2 * env.num_pits
        player_store = env.num_pits if player == 0 else 2 * env.num_pits + 1

        # Check for seeds that can land in the store
        sweep_potential = 0
        for i in range(player_side_start, player_side_end + 1):
            stones = env.board[i]
            if stones > 0:
                # Check if these stones can land in the store
                landing_idx = (i + stones) % len(env.board)
                if landing_idx == player_store:
                    # Extra bonus if this is part of a chain
                    sweep_potential += 3.0

        return sweep_potential

    def _pattern_chain_setup(self, env: BaseEnvironment, player: int) -> float:
        """Detect and evaluate chain move setups.

        This pattern checks for consecutive moves that can be chained together.

        Args:
            env: The game environment.
            player: The player to check for (0 or 1).

        Returns:
            A score value for the detected pattern.
        """
        player_side_start = 0 if player == 0 else env.num_pits + 1
        player_side_end = env.num_pits - 1 if player == 0 else 2 * env.num_pits
        player_store = env.num_pits if player == 0 else 2 * env.num_pits + 1

        chain_value = 0

        # Check for pit distances to the store
        for i in range(player_side_start, player_side_end + 1):
            stones = env.board[i]
            if stones > 0:
                # Distance to store
                distance_to_store = (player_store - i) if player_store > i else (len(env.board) - i + player_store)

                # Check if stones match distance exactly (direct store landing)
                if stones == distance_to_store:
                    chain_value += 4.0

                    # Check if there's a follow-up move possible
                    # This checks if another pit would land in this pit after the move
                    for j in range(player_side_start, player_side_end + 1):
                        if j != i:
                            j_stones = env.board[j]
                            if j_stones > 0:
                                j_distance = (i - j) if i > j else (len(env.board) - j + i)
                                if j_stones == j_distance:
                                    # Found a potential chain
                                    chain_value += 6.0

        return chain_value

    def _pattern_trap_setup(self, env: BaseEnvironment, player: int) -> float:
        """Detect and evaluate trap setups.

        This pattern checks for moves that force the opponent into unfavorable positions.

        Args:
            env: The game environment.
            player: The player to check for (0 or 1).

        Returns:
            A score value for the detected pattern.
        """
        player_side_start = 0 if player == 0 else env.num_pits + 1
        player_side_end = env.num_pits - 1 if player == 0 else 2 * env.num_pits

        opponent = 1 - player
        opponent_side_start = 0 if opponent == 0 else env.num_pits + 1
        opponent_side_end = env.num_pits - 1 if opponent == 0 else 2 * env.num_pits

        trap_value = 0

        # Check for empty pits on player's side adjacent to opponent pits with stones
        for i in range(player_side_start, player_side_end + 1):
            if env.board[i] == 0:
                # This is an empty pit - perfect for setting a trap
                opposite_idx = 2 * env.num_pits - i

                # Check if opposite pit has stones (potential capture setup)
                if opposite_idx >= opponent_side_start and opposite_idx <= opponent_side_end:
                    if env.board[opposite_idx] > 0:
                        trap_value += env.board[opposite_idx] * 1.5

                        # Extra bonus if nearby pits can land in this empty pit
                        for j in range(player_side_start, player_side_end + 1):
                            if j != i and env.board[j] > 0:
                                j_distance = (i - j) if i > j else (len(env.board) - j + i)
                                if env.board[j] == j_distance:
                                    trap_value += 2.0

        return trap_value

    def _pattern_safe_distribution(self, env: BaseEnvironment, player: int) -> float:
        """Evaluate the safety of the stone distribution.

        This pattern checks if stones are distributed in a way that minimizes
        risk of opponent captures.

        Args:
            env: The game environment.
            player: The player to check for (0 or 1).

        Returns:
            A score value for the pattern.
        """
        player_side_start = 0 if player == 0 else env.num_pits + 1
        player_side_end = env.num_pits - 1 if player == 0 else 2 * env.num_pits

        opponent = 1 - player
        opponent_side_start = 0 if opponent == 0 else env.num_pits + 1
        opponent_side_end = env.num_pits - 1 if opponent == 0 else 2 * env.num_pits

        safety_value = 0

        # Count empty pits - they can't be captured
        empty_pits = sum(1 for i in range(player_side_start, player_side_end + 1)
                         if env.board[i] == 0)

        # Check for vulnerable configurations
        for i in range(player_side_start, player_side_end + 1):
            if env.board[i] == 1:
                # Single stones are vulnerable - check if opponent can capture
                for j in range(opponent_side_start, opponent_side_end + 1):
                    j_stones = env.board[j]
                    if j_stones > 0:
                        j_distance = (i - j) if i > j else (len(env.board) - j + i)
                        if j_stones == j_distance:
                            # Opponent can capture this stone
                            safety_value -= 2.0

            elif env.board[i] > 1:
                # Multiple stones - less vulnerable
                safety_value += 0.5

        # Bonus for evenly distributed stones
        stones = [env.board[i] for i in range(player_side_start, player_side_end + 1) if env.board[i] > 0]
        if stones:
            mean = sum(stones) / len(stones)
            variance = sum((s - mean) ** 2 for s in stones) / len(stones)
            # Lower variance means more even distribution, which is safer
            safety_value += 3.0 / (1.0 + variance)

        return safety_value

    def get_settings(self) -> Dict[str, Tuple[Any, str, str]]:
        """Get the agent's configurable settings."""
        return {
            "depth": (self.depth, "Maximum depth to search in the game tree", "int"),
            "time_limit": (self.time_limit, "Maximum time in seconds to spend on a move", "float"),
            "tt_size": (self.tt_size, "Maximum size of transposition table", "int")
        }

    def set_setting(self, setting_name: str, value: Any) -> bool:
        """Set a specific setting to a new value."""
        if setting_name == "depth":
            try:
                depth = int(value)
                if depth > 0:
                    self.depth = depth
                    self.max_depth = depth
                    # Resize killer moves table
                    self.killer_moves = [[None, None] for _ in range(depth + 1)]
                    return True
                else:
                    print("Depth must be a positive integer.")
                    return False
            except ValueError:
                print("Depth must be a valid integer.")
                return False
        elif setting_name == "time_limit":
            try:
                time_limit = float(value)
                if time_limit > 0:
                    self.time_limit = time_limit
                    return True
                else:
                    print("Time limit must be a positive number.")
                    return False
            except ValueError:
                print("Time limit must be a valid number.")
                return False
        elif setting_name == "tt_size":
            try:
                tt_size = int(value)
                if tt_size > 0:
                    # Resize transposition table (clear it if needed)
                    if tt_size < len(self.transposition_table):
                        self.transposition_table = {}
                        self.tt_keys_queue = []
                    self.tt_size = tt_size
                    return True
                else:
                    print("Transposition table size must be a positive integer.")
                    return False
            except ValueError:
                print("Transposition table size must be a valid integer.")
                return False
        return False

    def setup_zobrist(self, env: BaseEnvironment) -> None:
        """Initialize Zobrist hashing tables for the given environment."""
        if self.zobrist_table is not None:
            return  # Already initialized

        # Determine maximum stones per pit (estimate conservatively)
        total_stones = np.sum(env.board)
        max_stones = total_stones  # Theoretical max if all stones in one pit

        # Create random bitstrings for each possible state
        board_size = len(env.board)  # Total pits including stores
        np.random.seed(42)  # Fixed seed for reproducibility

        # Initialize tables with random 64-bit integers
        self.zobrist_table = np.random.randint(
            1, 2**64 - 1, size=(board_size, max_stones + 1), dtype=np.uint64
        )

        # Random number for player turn
        self.zobrist_turn = np.random.randint(1, 2**64 - 1, dtype=np.uint64)

    def compute_zobrist_hash(self, env: BaseEnvironment) -> int:
        """Compute Zobrist hash for the current board state."""
        # Initialize Zobrist tables if needed
        if self.zobrist_table is None:
            self.setup_zobrist(env)

        # Start with 0
        hash_value = 0

        # XOR in the value for each pit's stone count
        for i, stones in enumerate(env.board):
            hash_value ^= self.zobrist_table[i, int(stones)]  # Ensure stones is int

        # XOR in the value for player turn
        if env.current_player == 1:  # Only XOR for player 1
            hash_value ^= self.zobrist_turn

        return hash_value

    def add_to_transposition_table(self, hash_key: int, depth: int,
                                   value: float, flag: str, best_move: Optional[int]) -> None:
        """Add an entry to the transposition table with LRU replacement."""
        # Check if we need to remove an entry due to size limit
        if len(self.transposition_table) >= self.tt_size:
            # Remove oldest entry
            if self.tt_keys_queue:
                oldest_key = self.tt_keys_queue.pop(0)
                if oldest_key in self.transposition_table:
                    del self.transposition_table[oldest_key]

        # Store the new entry
        entry = (depth, value, flag, best_move)
        self.transposition_table[hash_key] = entry

        # Add to queue for LRU tracking
        self.tt_keys_queue.append(hash_key)

        # Keep queue size manageable by removing duplicates
        if len(self.tt_keys_queue) > self.tt_size * 1.2:
            # Remove duplicates while preserving order (only keep the last occurrence)
            seen = set()
            new_queue = []
            for x in reversed(self.tt_keys_queue):
                if x not in seen:
                    seen.add(x)
                    new_queue.append(x)
            self.tt_keys_queue = list(reversed(new_queue))

    def get_action(self, env: BaseEnvironment) -> int:
        """Get the best action using iterative deepening and optimizations."""
        # Reset metrics
        self.nodes_searched = 0
        self.tt_hits = 0
        self.ab_cutoffs = 0
        self.q_searches = 0
        self.time_up = False
        self.eval_components = {}

        # Set start time for time management
        self.start_time = time.time()

        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return -1

        # Initialize best action
        best_action = valid_actions[0]
        best_value = float('-inf')

        # Create a deep copy of the environment
        env_copy = copy.deepcopy(env)

        # Get the Zobrist hash for this position
        position_hash = self.compute_zobrist_hash(env_copy)

        # Check game phase for dynamic depth adjustment
        game_phase = self._determine_game_phase(env_copy)

        # Adjust depth based on phase (search deeper in endgame)
        max_depth = self.max_depth
        if game_phase == self.PHASE_ENDGAME:
            max_depth += 2  # Search 2 levels deeper in endgame
        elif game_phase == self.PHASE_FINAL:
            max_depth += 4  # Search 4 levels deeper in final phase

        # Iterative deepening
        for current_depth in range(1, max_depth + 1):
            # Reset killer moves for this iteration
            self.killer_moves = [[None, None] for _ in range(current_depth + 1)]

            # For storing evaluations of all moves
            move_values = {}

            # Search each move at this depth
            for action in valid_actions:
                # Create a copy for this move
                move_env = copy.deepcopy(env_copy)

                # Take the action
                original_player = move_env.current_player
                _, reward, terminated, _, _ = move_env.step(action)

                # Check if the player gets an extra turn
                extra_turn = not terminated and move_env.current_player == original_player

                # Calculate new position hash
                move_hash = self._update_hash_after_move(env_copy, move_env, position_hash, action)

                # Color is always 1.0 for the current player, -1.0 for opponent
                next_color = 1.0 if extra_turn else -1.0

                # Evaluate this move
                if terminated:
                    # Game over after this move
                    value = self._evaluate_terminal(move_env) * (1.0 if move_env.get_winner() == env_copy.current_player else -1.0)
                else:
                    # Search deeper
                    value, _ = self._negamax(
                        move_env, current_depth - 1, float('-inf'), float('inf'),
                        next_color, move_hash, extra_turn
                    )
                    if not extra_turn:
                        value = -value  # Negate if opponent's turn next

                # Store the value
                move_values[action] = value

                # Check if we're out of time
                if self.time_up:
                    break

            # If we completed the search for this depth, update best action
            if not self.time_up:
                if move_values:
                    best_action = max(move_values.items(), key=lambda x: x[1])[0]
                    best_value = move_values[best_action]

            # Break if we're out of time
            if self.time_up:
                break

            # If we've found a winning move, no need to search deeper
            if best_value > 900:
                break

            # If we're almost out of time, don't start a new iteration
            elapsed_time = time.time() - self.start_time
            if elapsed_time > self.time_limit * 0.8:
                break

        if self.debug_mode:
            # Print debug info
            for action in valid_actions:
                if action in move_values:
                    print(f"Move {action}: {move_values[action]:.2f}")

            # Print evaluation components
            if self.eval_components:
                print("Evaluation components:")
                for k, v in self.eval_components.items():
                    print(f"  {k}: {v:.2f}")

        return best_action

    def _negamax(self, env: BaseEnvironment, depth: int, alpha: float, beta: float,
                 color: float, position_hash: int, is_extra_turn: bool = False) -> Tuple[float, Optional[int]]:
        """Negamax search with alpha-beta pruning and optimizations."""
        # Increment nodes searched counter
        self.nodes_searched += 1

        # Check if we're out of time
        if self.nodes_searched % 1000 == 0:
            if time.time() - self.start_time > self.time_limit:
                self.time_up = True
                return 0.0, None

        # Original alpha value for TT flag determination
        alpha_orig = alpha

        # Check transposition table
        if position_hash in self.transposition_table:
            tt_depth, tt_value, tt_flag, tt_move = self.transposition_table[position_hash]

            # Only use the stored value if it was searched deep enough
            if tt_depth >= depth:
                self.tt_hits += 1

                # Use stored value based on its flag
                if tt_flag == "EXACT":
                    return tt_value, tt_move
                elif tt_flag == "LOWER_BOUND" and tt_value > alpha:
                    alpha = tt_value
                elif tt_flag == "UPPER_BOUND" and tt_value < beta:
                    beta = tt_value

                # Return if we have a cutoff
                if alpha >= beta:
                    return tt_value, tt_move
        else:
            tt_move = None

        # Check for terminal state
        if env.done:
            score = self._evaluate_terminal(env) * color
            return score, None

        # Dynamic depth extension for important positions
        should_extend_depth = False

        # Extend search depth in critical positions
        if self._is_critical_position(env):
            should_extend_depth = True

        # If we're at max depth and not extending, use quiescence search
        if depth <= 0 and not should_extend_depth:
            return self._quiescence_search(env, alpha, beta, color, is_extra_turn), None

        valid_actions = env.get_valid_actions()
        if not valid_actions:
            score = self._evaluate(env, is_extra_turn) * color
            return score, None

        # Get ordered actions for more efficient search
        ordered_actions = self._order_moves(env, valid_actions, depth, tt_move)

        best_value = float('-inf')
        best_action = None

        # Try each valid action
        for i, action in enumerate(ordered_actions):
            # Create a deep copy of the environment
            env_copy = copy.deepcopy(env)

            # Take the action
            original_player = env_copy.current_player
            _, reward, terminated, _, _ = env_copy.step(action)

            # Check if the player gets an extra turn
            extra_turn = not terminated and env_copy.current_player == original_player

            # Calculate new position hash by updating incrementally
            new_hash = self._update_hash_after_move(env, env_copy, position_hash, action)

            # Determine next color based on extra turn
            next_color = color if extra_turn else -color

            # Apply Late Move Reduction for moves searched later
            # But don't reduce if this is a capture move or gives an extra turn
            reduced_depth = depth
            if (i >= 3 and depth >= 3 and not extra_turn and
                    not self._is_capture_move(env, action) and
                    not should_extend_depth):
                reduced_depth = depth - 1  # Reduce depth for less promising moves

            # Add depth for critical positions
            if should_extend_depth:
                reduced_depth += 1

            # Recursively evaluate the resulting state
            child_value, _ = self._negamax(
                env_copy, reduced_depth - 1, -beta, -alpha, next_color, new_hash, extra_turn
            )
            value = -child_value

            # If reduced depth search indicates a good move, re-search at full depth
            if reduced_depth < depth and value > alpha:
                child_value, _ = self._negamax(
                    env_copy, depth - 1, -beta, -alpha, next_color, new_hash, extra_turn
                )
                value = -child_value

            # Update best value and action
            if value > best_value:
                best_value = value
                best_action = action

            # Update alpha
            alpha = max(alpha, value)

            # Beta cutoff
            if alpha >= beta:
                self.ab_cutoffs += 1

                # Store as killer move if non-capturing
                if not self._is_capture_move(env, action):
                    if self.killer_moves[depth][0] != action:
                        self.killer_moves[depth][1] = self.killer_moves[depth][0]
                        self.killer_moves[depth][0] = action

                # Update history table
                if action not in self.history_table:
                    self.history_table[action] = 0
                self.history_table[action] += 2 ** depth  # Weight by depth

                break

            # Check if we're out of time after each move
            if self.time_up:
                return best_value, best_action

        # Determine the type of value for transposition table
        if best_value <= alpha_orig:
            tt_flag = "UPPER_BOUND"
        elif best_value >= beta:
            tt_flag = "LOWER_BOUND"
        else:
            tt_flag = "EXACT"

        # Store in transposition table
        self.add_to_transposition_table(position_hash, depth, best_value, tt_flag, best_action)

        return best_value, best_action

    def _is_critical_position(self, env: BaseEnvironment) -> bool:
        """Determine if a position is critical and warrants deeper search.

        Critical positions include:
        1. Near-empty board states
        2. Positions with many potential captures
        3. Positions with chain move potential

        Args:
            env: The game environment.

        Returns:
            True if the position is critical, False otherwise.
        """
        # Get game phase
        game_phase = self._determine_game_phase(env)

        # Final phase is always critical
        if game_phase == self.PHASE_FINAL:
            return True

        # Check for potential sweep (one side close to empty)
        player = env.current_player
        player_side_start = 0 if player == 0 else env.num_pits + 1
        player_side_end = env.num_pits - 1 if player == 0 else 2 * env.num_pits

        player_stones = sum(env.board[player_side_start:player_side_end+1])

        # If player's side is almost empty, this is critical
        if player_stones < env.num_pits * 2:
            return True

        # Check for multiple potential captures
        capture_potential = 0
        for action in env.get_valid_actions():
            if self._is_capture_move(env, action):
                capture_potential += 1

        # If there are multiple potential captures, this is critical
        if capture_potential >= 2:
            return True

        # Check for chain move potential
        chain_potential = self._pattern_chain_setup(env, player)
        if chain_potential >= 6.0:  # High chain potential
            return True

        return False

    def _update_hash_after_move(self, old_env: BaseEnvironment, new_env: BaseEnvironment,
                                old_hash: int, action: int) -> int:
        """Update Zobrist hash incrementally after a move."""
        # Start with the old hash
        new_hash = old_hash

        # XOR out old board state and XOR in new board state for changed pits
        for i in range(len(old_env.board)):
            if old_env.board[i] != new_env.board[i]:
                # XOR out old value
                new_hash ^= self.zobrist_table[i, int(old_env.board[i])]
                # XOR in new value
                new_hash ^= self.zobrist_table[i, int(new_env.board[i])]

        # Update player turn if it changed
        if old_env.current_player != new_env.current_player:
            new_hash ^= self.zobrist_turn

        return new_hash

    def _quiescence_search(self, env: BaseEnvironment, alpha: float, beta: float,
                           color: float, is_extra_turn: bool, depth: int = 3) -> float:
        """Quiescence search to evaluate volatile positions more accurately."""
        self.q_searches += 1

        # Base case: terminal state or max depth reached
        if env.done or depth <= 0:
            return self._evaluate(env, is_extra_turn) * color

        # First, get a static evaluation
        stand_pat = self._evaluate(env, is_extra_turn) * color

        # Check if we can prune immediately
        if stand_pat >= beta:
            return stand_pat

        # Update alpha if the static evaluation is better
        alpha = max(alpha, stand_pat)

        # Find all capture moves and extra turn moves
        valid_actions = env.get_valid_actions()

        # If no valid moves, return static evaluation
        if not valid_actions:
            return stand_pat

        # Find all tactically important moves (captures and extra turns)
        tactical_moves = []
        for action in valid_actions:
            # Check if this is a capture move
            if self._is_capture_move(env, action):
                tactical_moves.append(action)
                continue

            # Check if this gives an extra turn
            pit_idx = action if env.current_player == 0 else env.num_pits + 1 + action
            stones = env.board[pit_idx]
            if stones > 0:
                landing_idx = (pit_idx + stones) % len(env.board)
                player_store = env.num_pits if env.current_player == 0 else 2 * env.num_pits + 1
                if landing_idx == player_store:
                    tactical_moves.append(action)

        # If no tactical moves, return static evaluation
        if not tactical_moves:
            return stand_pat

        # Check each tactical move
        for action in tactical_moves:
            # Create a copy of the environment
            env_copy = copy.deepcopy(env)

            # Take the action
            original_player = env_copy.current_player
            _, reward, terminated, _, _ = env_copy.step(action)

            # Check if the player gets an extra turn
            extra_turn = not terminated and env_copy.current_player == original_player

            # Determine next color based on extra turn
            next_color = color if extra_turn else -color

            # Recursively evaluate
            value = -self._quiescence_search(env_copy, -beta, -alpha, next_color, extra_turn, depth - 1)

            # Update alpha
            alpha = max(alpha, value)

            # Beta cutoff
            if alpha >= beta:
                break

        return alpha

    def _is_capture_move(self, env: BaseEnvironment, action: int) -> bool:
        """Check if a move will result in a capture."""
        player = env.current_player
        player_side_start = 0 if player == 0 else env.num_pits + 1
        player_side_end = env.num_pits - 1 if player == 0 else 2 * env.num_pits

        # Calculate the pit index
        pit_idx = action if player == 0 else env.num_pits + 1 + action
        stones = env.board[pit_idx]

        # Calculate where the last stone would land
        landing_idx = (pit_idx + stones) % len(env.board)

        # If landing in player's store, not a capture
        player_store = env.num_pits if player == 0 else 2 * env.num_pits + 1
        if landing_idx == player_store:
            return False

        # Check if landing in an empty pit on the player's side
        if (landing_idx >= player_side_start and
                landing_idx <= player_side_end and
                env.board[landing_idx] == 0):

            # Calculate the opposite pit index
            opposite_idx = 2 * env.num_pits - landing_idx

            # If the opposite pit has stones, it's a capture
            if env.board[opposite_idx] > 0:
                return True

        return False

    def _order_moves(self, env: BaseEnvironment, valid_actions: List[int],
                     depth: int, tt_move: Optional[int]) -> List[int]:
        """Order moves for better alpha-beta pruning efficiency."""
        player = env.current_player
        player_store = env.num_pits if player == 0 else 2 * env.num_pits + 1

        # Create a list of (action, score) tuples for sorting
        action_scores = []

        for action in valid_actions:
            # Start with a base score of 0
            score = 0

            # Prioritize transposition table move
            if action == tt_move:
                score += 20000  # Higher priority for TT moves

            # Prioritize killer moves
            if action == self.killer_moves[depth][0]:
                score += 10000
            elif action == self.killer_moves[depth][1]:
                score += 9000

            # History heuristic
            if action in self.history_table:
                score += min(self.history_table[action], 8000)  # Cap to prevent dominating other heuristics

            # Calculate the actual index in the board array
            board_idx = action if player == 0 else env.num_pits + 1 + action
            stones = env.board[board_idx]

            # Base score includes the number of stones
            score += stones * 10

            # Check if this move leads to an extra turn (much higher priority)
            landing_idx = (board_idx + stones) % len(env.board)
            if landing_idx == player_store:
                score += 15000  # Very high priority for extra turns

            # Check if this move leads to a capture (high priority)
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
                    score += 12000 + env.board[opposite_idx] * 100

            # Detect if this move can set up a future extra turn
            # (landing stone count would match exactly what's needed to land in store)
            if landing_idx >= player_side_start and landing_idx <= player_side_end:
                # Distance from landing spot to store
                distance_to_store = (player_store - landing_idx) if player_store > landing_idx else (len(env.board) - landing_idx + player_store)
                if stones == 1:  # Would land exactly 1 stone
                    # Check if there are exactly 'distance_to_store - 1' stones in the target pit
                    # (After our move it would have 'distance_to_store' stones, perfect for an extra turn)
                    if env.board[landing_idx] == distance_to_store - 1:
                        score += 5000  # Good setup move

            # Check for sweep potential - moves that can clear our side
            game_phase = self._determine_game_phase(env)
            if game_phase >= self.PHASE_ENDGAME:
                # How many pits would be empty after this move
                empty_after = sum(1 for i in range(player_side_start, player_side_end + 1)
                                  if i != board_idx and env.board[i] == 0)
                # If this move would empty the current pit
                if stones == 1:
                    empty_after += 1

                # Higher priority for moves that help clear our side in endgame
                score += empty_after * 500

            action_scores.append((action, score))

        # Sort actions by score in descending order
        action_scores.sort(key=lambda x: x[1], reverse=True)

        # Return just the ordered actions
        return [action for action, _ in action_scores]

    def _evaluate_terminal(self, env: BaseEnvironment) -> float:
        """Evaluate terminal game states with more nuance."""
        winner = env.get_winner()

        # Get the stone count difference
        player0_store = env.board[env.num_pits]
        player1_store = env.board[2 * env.num_pits + 1]
        score_diff = player0_store - player1_store

        # Normalize to current player's perspective
        if env.current_player == 1:
            score_diff = -score_diff

        if winner is None:  # Draw
            return 0.0
        elif winner == env.current_player:  # Current player wins
            # Scale win value by margin of victory for more nuanced evaluation
            margin = abs(score_diff)
            return 1000.0 + margin
        else:  # Current player loses
            # Scale loss value by margin of defeat
            margin = abs(score_diff)
            return -1000.0 - margin

    def _determine_game_phase(self, env: BaseEnvironment) -> int:
        """Determine the current game phase based on the board state."""
        # Count total stones on each side
        player0_stones = sum(env.board[0:env.num_pits])
        player1_stones = sum(env.board[env.num_pits+1:2*env.num_pits+1])

        # Count empty pits on each side
        player0_empty = sum(1 for i in range(env.num_pits) if env.board[i] == 0)
        player1_empty = sum(1 for i in range(env.num_pits+1, 2*env.num_pits+1)
                            if env.board[i] == 0)

        # Calculate empty pit ratio
        total_empty = player0_empty + player1_empty
        total_pits = 2 * env.num_pits
        empty_ratio = total_empty / total_pits

        # Check for near-empty sides (final phase)
        if player0_stones <= 3 or player1_stones <= 3:
            return self.PHASE_FINAL

        # Otherwise use empty pit ratio
        if empty_ratio < self.ENDGAME_THRESHOLD:
            if empty_ratio < 0.25:
                return self.PHASE_OPENING
            else:
                return self.PHASE_MIDGAME
        elif empty_ratio < self.FINAL_THRESHOLD:
            return self.PHASE_ENDGAME
        else:
            return self.PHASE_FINAL

    def _evaluate(self, env: BaseEnvironment, is_extra_turn: bool = False) -> float:
        """Enhanced evaluation function with phase-based weights and pattern recognition."""
        # If the game is over, use terminal evaluation
        if env.done:
            return self._evaluate_terminal(env)

        # Determine game phase
        game_phase = self._determine_game_phase(env)

        # Get the basic evaluation based on stones in stores
        player0_store = env.board[env.num_pits]
        player1_store = env.board[2 * env.num_pits + 1]

        # Calculate the score difference
        if env.current_player == 0:
            score_diff = player0_store - player1_store
        else:
            score_diff = player1_store - player0_store

        # Get the current player
        player = env.current_player
        opponent = 1 - player

        # Initialize component values
        store_value = score_diff
        capture_value = self._evaluate_potential_captures(env)
        extra_turn_value = self._evaluate_potential_extra_turns(env)
        distribution_value = self._evaluate_stone_distribution(env)
        empty_pits_value = self._evaluate_empty_pits(env)
        tempo_value = self._evaluate_tempo(env)
        mobility_value = self._evaluate_mobility(env)
        chain_value = self._evaluate_chain_potential(env)

        # Pattern recognition values
        pattern_values = {}
        for pattern_name, pattern_func in self.pattern_database.items():
            pattern_values[pattern_name] = pattern_func(env, player)

        # Phase-specific weight adjustments
        if game_phase == self.PHASE_OPENING:
            # Opening: prioritize stone distribution and development
            weights = {
                'store': 1.0,
                'capture': 1.0,
                'extra_turn': 2.5,
                'distribution': 1.0,
                'empty_pits': 0.2,
                'tempo': 0.8,
                'mobility': 1.0,
                'chain': 1.5,
                'store_sweep': 0.5,
                'chain_setup': 1.5,
                'trap_setup': 0.5,
                'safe_distribution': 1.0
            }
        elif game_phase == self.PHASE_MIDGAME:
            # Midgame: balance all factors
            weights = {
                'store': 1.0,
                'capture': 1.5,
                'extra_turn': 2.0,
                'distribution': 0.6,
                'empty_pits': 0.4,
                'tempo': 0.6,
                'mobility': 0.8,
                'chain': 1.2,
                'store_sweep': 0.8,
                'chain_setup': 1.2,
                'trap_setup': 0.7,
                'safe_distribution': 0.8
            }
        elif game_phase == self.PHASE_ENDGAME:
            # Endgame: prioritize material and immediate gains
            weights = {
                'store': 1.5,
                'capture': 2.0,
                'extra_turn': 1.8,
                'distribution': 0.4,
                'empty_pits': 1.2,
                'tempo': 0.4,
                'mobility': 0.6,
                'chain': 1.0,
                'store_sweep': 1.2,
                'chain_setup': 1.0,
                'trap_setup': 0.5,
                'safe_distribution': 0.6
            }
        else:  # PHASE_FINAL
            # Final phase: focus on precise calculation
            weights = {
                'store': 2.0,
                'capture': 2.5,
                'extra_turn': 1.5,
                'distribution': 0.2,
                'empty_pits': 2.0,
                'tempo': 0.3,
                'mobility': 0.4,
                'chain': 0.8,
                'store_sweep': 2.0,
                'chain_setup': 0.8,
                'trap_setup': 0.3,
                'safe_distribution': 0.4
            }

        # Bonus for having just had an extra turn (momentum)
        if is_extra_turn:
            # Increase weights for aggressive play after an extra turn
            weights['capture'] *= 1.2
            weights['extra_turn'] *= 1.2
            weights['chain'] *= 1.5

        # Calculate weighted sum of all components
        evaluation = store_value * weights['store']
        evaluation += capture_value * weights['capture']
        evaluation += extra_turn_value * weights['extra_turn']
        evaluation += distribution_value * weights['distribution']
        evaluation += empty_pits_value * weights['empty_pits']
        evaluation += tempo_value * weights['tempo']
        evaluation += mobility_value * weights['mobility']
        evaluation += chain_value * weights['chain']

        # Add pattern values
        for pattern_name, pattern_value in pattern_values.items():
            evaluation += pattern_value * weights.get(pattern_name, 1.0)

        # Special endgame evaluation to prioritize clearing your side
        if game_phase >= self.PHASE_ENDGAME:
            sweep_score = self._evaluate_sweep_potential(env)
            evaluation += sweep_score * (2.0 if game_phase == self.PHASE_FINAL else 1.0)

        # Store component values for debugging
        if self.debug_mode:
            self.eval_components = {
                'store': store_value * weights['store'],
                'capture': capture_value * weights['capture'],
                'extra_turn': extra_turn_value * weights['extra_turn'],
                'distribution': distribution_value * weights['distribution'],
                'empty_pits': empty_pits_value * weights['empty_pits'],
                'tempo': tempo_value * weights['tempo'],
                'mobility': mobility_value * weights['mobility'],
                'chain': chain_value * weights['chain']
            }
            for pattern_name, pattern_value in pattern_values.items():
                self.eval_components[pattern_name] = pattern_value * weights.get(pattern_name, 1.0)

            if game_phase >= self.PHASE_ENDGAME:
                self.eval_components['sweep'] = sweep_score * (2.0 if game_phase == self.PHASE_FINAL else 1.0)

        return evaluation

    def _evaluate_potential_captures(self, env: BaseEnvironment) -> float:
        """Enhanced evaluation of potential captures with more nuanced logic."""
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
                    # Add a negative evaluation to avoid moves that create capture opportunities
                    # Scale by the value of the opposite pit
                    evaluation -= env.board[opposite_idx] * 1.8

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
                        # Scale by the value of the capture plus a bonus
                        evaluation += (env.board[opposite_idx] + 1) * 2.5

                        # Extra bonus for immediate captures (1-hop away)
                        if stones == landing_idx - i or (landing_idx < i and stones == len(env.board) - i + landing_idx):
                            evaluation += 2.0

                        # Even more bonus for high-value captures
                        if env.board[opposite_idx] >= 4:
                            evaluation += 2.0

        # Also evaluate the opponent's potential captures (for defensive play)
        for i in range(opponent_side_start, opponent_side_end + 1):
            stones = env.board[i]
            if stones > 0:
                landing_idx = (i + stones) % len(env.board)

                # Skip if it would land in the player's store
                player_store = env.num_pits if player == 0 else 2 * env.num_pits + 1
                if landing_idx == player_store:
                    continue

                if (landing_idx >= opponent_side_start and
                        landing_idx <= opponent_side_end and
                        env.board[landing_idx] == 0):

                    opposite_idx = 2 * env.num_pits - landing_idx

                    if (opposite_idx >= player_side_start and
                            opposite_idx <= player_side_end and
                            env.board[opposite_idx] > 0):
                        # Penalize allowing opponent captures - heavily penalize high-value captures
                        base_penalty = (env.board[opposite_idx] + 1) * 2.0

                        # Immediate captures are more dangerous
                        if stones == landing_idx - i or (landing_idx < i and stones == len(env.board) - i + landing_idx):
                            base_penalty *= 1.5

                        # High-value captures are even more dangerous
                        if env.board[opposite_idx] >= 4:
                            base_penalty *= 1.5

                        evaluation -= base_penalty

        return evaluation

    def _evaluate_potential_extra_turns(self, env: BaseEnvironment) -> float:
        """Enhanced evaluation of potential extra turns with chaining potential."""
        # Initialize the evaluation
        evaluation = 0.0

        # Get the player's store index
        player = env.current_player
        player_store = env.num_pits if player == 0 else 2 * env.num_pits + 1

        # Get the player's side range
        player_side_start = 0 if player == 0 else env.num_pits + 1
        player_side_end = env.num_pits - 1 if player == 0 else 2 * env.num_pits

        # Get the opponent's store and side range
        opponent = 1 - player
        opponent_store = env.num_pits if opponent == 0 else 2 * env.num_pits + 1
        opponent_side_start = 0 if opponent == 0 else env.num_pits + 1
        opponent_side_end = env.num_pits - 1 if opponent == 0 else 2 * env.num_pits

        # Track potential chain moves
        chain_moves = []

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
                    base_value = 3.5  # Higher base value for extra turns

                    # Bonus for extra turns from pits with more stones (more impactful)
                    bonus = min(stones / 4, 1.2)  # Cap the bonus

                    # Higher value for extra turns that can be performed immediately
                    if stones > 0:
                        urgency_factor = 1.2  # Increased urgency factor
                    else:
                        # Discount future potential
                        urgency_factor = 0.8

                    # Calculate total value for this extra turn
                    move_value = base_value * (1 + bonus) * urgency_factor

                    # Add to evaluation
                    evaluation += move_value

                    # Store this as a potential chain move
                    chain_moves.append((i, landing_idx))

        # Analyze chain potential - find sequences of extra turns
        if chain_moves:
            # Build a graph of possible move sequences
            chain_graph = {}
            for start, end in chain_moves:
                if start not in chain_graph:
                    chain_graph[start] = []
                chain_graph[start].append(end)

            # Find longest chains
            max_chain = 0
            for start in chain_graph:
                chain_length = self._find_longest_chain(chain_graph, start, set())
                max_chain = max(max_chain, chain_length)

            # Bonus for long chains (exponential scaling)
            if max_chain > 1:
                chain_bonus = 2.0 * (2 ** (max_chain - 1))  # 2, 4, 8, 16... for chains of length 2, 3, 4, 5...
                evaluation += chain_bonus

        # Penalize opponent extra turn potential
        opponent_extra_turn_potential = 0
        for i in range(opponent_side_start, opponent_side_end + 1):
            stones = env.board[i]
            if stones > 0:
                landing_idx = (i + stones) % len(env.board)
                if landing_idx == opponent_store:
                    # Opponent has an extra turn potential
                    opponent_extra_turn_potential += 1

                    # Higher penalty for immediate extra turns
                    if i == opponent_side_start or stones <= 3:  # Easy to execute
                        opponent_extra_turn_potential += 0.5

        # Apply penalty based on opponent's extra turn potential
        evaluation -= opponent_extra_turn_potential * 1.8

        return evaluation

    def _find_longest_chain(self, graph: Dict[int, List[int]], node: int, visited: Set[int]) -> int:
        """Find the longest chain in a graph using DFS.

        Args:
            graph: Adjacency list representation of the graph.
            node: Current node.
            visited: Set of visited nodes.

        Returns:
            Length of the longest chain starting from the current node.
        """
        if node in visited:
            return 0

        visited.add(node)

        max_length = 1

        if node in graph:
            for neighbor in graph[node]:
                max_length = max(max_length, 1 + self._find_longest_chain(graph, neighbor, visited.copy()))

        return max_length

    def _evaluate_stone_distribution(self, env: BaseEnvironment) -> float:
        """Enhanced evaluation of stone distribution considering game phase."""
        # Initialize the evaluation
        evaluation = 0.0

        # Get the player's side range
        player = env.current_player
        player_side_start = 0 if player == 0 else env.num_pits + 1
        player_side_end = env.num_pits - 1 if player == 0 else 2 * env.num_pits

        # Get the game phase
        game_phase = self._determine_game_phase(env)

        # Evaluate the distribution of stones on the player's side
        for i in range(player_side_start, player_side_end + 1):
            stones = env.board[i]

            if game_phase == self.PHASE_OPENING:
                # In opening, prefer spreading stones evenly and keeping some in each pit
                if stones == 0:
                    evaluation -= 0.6  # Penalty for empty pits
                elif stones == 1:
                    evaluation -= 0.4  # Slight penalty for vulnerable single stones
                else:
                    # Bonus for having a good number of stones (not too many, not too few)
                    optimal = 4  # Typically good number for developing positions
                    deviation = abs(stones - optimal)
                    evaluation += 0.3 * (1.0 / (1.0 + deviation))

                # In opening, slight preference for stones in pits closer to our store
                # (for developing the position)
                position_weight = (i - player_side_start + 1) / (player_side_end - player_side_start + 1)
                evaluation += stones * position_weight * 0.08

            elif game_phase == self.PHASE_MIDGAME:
                # In midgame, balance between concentration and distribution
                if stones >= 12:
                    # Having too many stones in one pit can be inefficient
                    evaluation -= (stones - 11) * 0.15
                elif stones >= 8:
                    # Good number of stones for creating opportunities
                    evaluation += 0.4
                elif stones >= 4:
                    # Decent number
                    evaluation += 0.25
                elif stones == 0:
                    # Empty pits reduce options
                    evaluation -= 0.3

                # Slight preference for balanced distribution
                total_stones = sum(env.board[player_side_start:player_side_end+1])
                if total_stones > 0:  # Avoid division by zero
                    average_stones = total_stones / env.num_pits
                    deviation = abs(stones - average_stones)
                    evaluation -= deviation * 0.03

            else:  # ENDGAME or FINAL
                # In endgame, prioritize moves that clear our side
                # (This is a common endgame strategy in Mancala)
                if stones == 0:
                    evaluation += 0.5  # Increased bonus for empty pits in endgame

                # Value stones based on position (closer to our store is better)
                position_weight = (i - player_side_start + 1) / (player_side_end - player_side_start + 1)
                evaluation += stones * position_weight * 0.2

                # In final phase, heavily penalize having too many stones in a single pit
                if game_phase == self.PHASE_FINAL and stones > 8:
                    evaluation -= (stones - 8) * 0.3

        return evaluation

    def _evaluate_empty_pits(self, env: BaseEnvironment) -> float:
        """Enhanced evaluation of empty pits with phase-specific logic."""
        # Initialize the evaluation
        evaluation = 0.0

        # Get the player's side range
        player = env.current_player
        player_side_start = 0 if player == 0 else env.num_pits + 1
        player_side_end = env.num_pits - 1 if player == 0 else 2 * env.num_pits

        # Get the opponent's side range
        opponent_side_start = env.num_pits + 1 if player == 0 else 0
        opponent_side_end = 2 * env.num_pits if player == 0 else env.num_pits - 1

        # Get the game phase
        game_phase = self._determine_game_phase(env)

        # Count empty pits on the player's side
        player_empty_pits = sum(1 for i in range(player_side_start, player_side_end + 1)
                                if env.board[i] == 0)

        # Count empty pits on the opponent's side
        opponent_empty_pits = sum(1 for i in range(opponent_side_start, opponent_side_end + 1)
                                  if env.board[i] == 0)

        # Phase-specific evaluation
        if game_phase == self.PHASE_OPENING:
            # In opening, empty pits are generally bad (less flexibility)
            evaluation -= player_empty_pits * 0.5

            # But it's good if opponent has empty pits
            evaluation += opponent_empty_pits * 0.25

        elif game_phase == self.PHASE_MIDGAME:
            # In midgame, a few empty pits can be useful for strategy
            if player_empty_pits <= 2:
                evaluation += player_empty_pits * 0.15
            else:
                # Too many empty pits means fewer options
                evaluation -= (player_empty_pits - 2) * 0.25

            # Opponent empty pits are always good
            evaluation += opponent_empty_pits * 0.35

        else:  # ENDGAME or FINAL
            # In endgame, empty pits on your side are good (clearing strategy)
            # Higher bonus for FINAL phase
            multiplier = 0.6 if game_phase == self.PHASE_ENDGAME else 1.0
            evaluation += player_empty_pits * multiplier

            # But we also want to have enough to capture opponent's remaining stones
            player_total_stones = sum(env.board[player_side_start:player_side_end+1])
            if player_total_stones == 0:
                # All empty is great in endgame if we're ahead
                player_store = env.num_pits if player == 0 else 2 * env.num_pits + 1
                opponent_store = env.num_pits if 1 - player == 0 else 2 * env.num_pits + 1

                if env.board[player_store] > env.board[opponent_store]:
                    evaluation += 6.0
                else:
                    # Unless we're behind, then it's terrible
                    evaluation -= 6.0

            # If opponent has few stones and many empty pits, good for us
            opponent_total_stones = sum(env.board[opponent_side_start:opponent_side_end+1])
            if opponent_empty_pits > env.num_pits / 2 and opponent_total_stones < player_total_stones:
                evaluation += 3.0

        return evaluation

    def _evaluate_tempo(self, env: BaseEnvironment) -> float:
        """Evaluate the tempo/initiative advantage."""
        # Initialize the evaluation
        evaluation = 0.0

        # Get the player's side range
        player = env.current_player
        player_side_start = 0 if player == 0 else env.num_pits + 1
        player_side_end = env.num_pits - 1 if player == 0 else 2 * env.num_pits

        # Get the opponent's side range
        opponent_side_start = env.num_pits + 1 if player == 0 else 0
        opponent_side_end = 2 * env.num_pits if player == 0 else env.num_pits - 1

        # Count stones and pits on both sides
        player_stones = sum(env.board[player_side_start:player_side_end+1])
        opponent_stones = sum(env.board[opponent_side_start:opponent_side_end+1])

        player_non_empty_pits = sum(1 for i in range(player_side_start, player_side_end + 1)
                                    if env.board[i] > 0)
        opponent_non_empty_pits = sum(1 for i in range(opponent_side_start, opponent_side_end + 1)
                                      if env.board[i] > 0)

        # Evaluate tempo based on stone count and distribution
        # More stones and more occupied pits generally means more options and better tempo
        if opponent_stones > 0:
            stone_ratio = player_stones / opponent_stones
        else:
            stone_ratio = 2.0 if player_stones > 0 else 1.0

        if opponent_non_empty_pits > 0:
            pit_ratio = player_non_empty_pits / opponent_non_empty_pits
        else:
            pit_ratio = 2.0 if player_non_empty_pits > 0 else 1.0

        # Balance between having many stones and having them well distributed
        evaluation += (stone_ratio + pit_ratio) * 0.6

        # Extra bonus for having more options than opponent
        if player_non_empty_pits > opponent_non_empty_pits:
            evaluation += 1.2

        # Extra LARGE bonus if opponent has very few options
        if opponent_non_empty_pits <= 2 and player_non_empty_pits > 3:
            evaluation += 3.0

        # Bonus for having moves that can set up future opportunities
        setup_potential = self._evaluate_setup_moves(env)
        evaluation += setup_potential

        return evaluation

    def _evaluate_setup_moves(self, env: BaseEnvironment) -> float:
        """Evaluate moves that set up future opportunities.

        This looks for positions where one move can lead to another effective move,
        such as setting up a future capture or extra turn.

        Args:
            env: The game environment.

        Returns:
            A value representing the setup potential.
        """
        player = env.current_player
        player_side_start = 0 if player == 0 else env.num_pits + 1
        player_side_end = env.num_pits - 1 if player == 0 else 2 * env.num_pits
        player_store = env.num_pits if player == 0 else 2 * env.num_pits + 1

        setup_value = 0.0

        # Check all possible current moves
        for i in range(player_side_start, player_side_end + 1):
            stones_i = env.board[i]
            if stones_i > 0:
                landing_i = (i + stones_i) % len(env.board)

                # If this move would add a stone to another pit on our side
                if landing_i >= player_side_start and landing_i <= player_side_end:
                    # That pit (plus one stone) could then be used for something useful
                    stones_j = env.board[landing_i] + 1  # +1 because we'd add a stone

                    # Calculate where this secondary move would land
                    landing_j = (landing_i + stones_j) % len(env.board)

                    # Check if secondary move would land in store (extra turn)
                    if landing_j == player_store:
                        setup_value += 1.5

                    # Check if secondary move would lead to a capture
                    elif (landing_j >= player_side_start and
                          landing_j <= player_side_end and
                          env.board[landing_j] == 0):

                        # Calculate the opposite pit
                        opposite_idx = 2 * env.num_pits - landing_j

                        # If opposite pit has stones, this would be a capture
                        if env.board[opposite_idx] > 0:
                            # Bonus for setting up a capture
                            setup_value += 1.2 + env.board[opposite_idx] * 0.1

        return setup_value

    def _evaluate_mobility(self, env: BaseEnvironment) -> float:
        """Evaluate the mobility advantage.

        Mobility is about having a variety of effective move options.

        Args:
            env: The game environment.

        Returns:
            A value representing the mobility advantage.
        """
        player = env.current_player

        # Count effective moves for player
        valid_moves = env.get_valid_actions()

        # If no valid moves, worst possible mobility
        if not valid_moves:
            return -5.0

        # Base mobility is just the count of valid moves
        mobility = len(valid_moves) * 0.5

        # Count high-quality moves (extra turns and captures)
        quality_moves = 0
        for action in valid_moves:
            # Check if move gives extra turn
            if self._move_gives_extra_turn(env, action):
                quality_moves += 1

            # Check if move leads to capture
            if self._is_capture_move(env, action):
                quality_moves += 1

        # Bonus for having quality moves
        mobility += quality_moves * 0.8

        # Simulate opponent's next position to evaluate their mobility
        opponent_mobility = 0

        # Create a simulation environment
        sim_env = copy.deepcopy(env)

        # Temporarily make the move that gives the opponent the worst mobility
        min_opponent_mobility = float('inf')

        for action in valid_moves:
            move_env = copy.deepcopy(sim_env)
            original_player = move_env.current_player
            _, _, terminated, _, _ = move_env.step(action)

            # Skip if game ends or player gets another turn
            if terminated or move_env.current_player == original_player:
                continue

            # Count opponent's valid moves
            opponent_valid_moves = move_env.get_valid_actions()
            opponent_move_count = len(opponent_valid_moves)

            # Count opponent's quality moves
            opponent_quality_moves = 0
            for opp_action in opponent_valid_moves:
                if self._move_gives_extra_turn(move_env, opp_action):
                    opponent_quality_moves += 1

                if self._is_capture_move(move_env, opp_action):
                    opponent_quality_moves += 1

            # Calculate opponent mobility for this position
            this_opponent_mobility = opponent_move_count * 0.5 + opponent_quality_moves * 0.8

            # Track the minimum opponent mobility
            min_opponent_mobility = min(min_opponent_mobility, this_opponent_mobility)

        # If we found a move that restricts opponent mobility
        if min_opponent_mobility != float('inf'):
            opponent_mobility = min_opponent_mobility

        # Mobility advantage is our mobility minus opponent's mobility
        mobility_advantage = mobility - opponent_mobility

        return mobility_advantage

    def _move_gives_extra_turn(self, env: BaseEnvironment, action: int) -> bool:
        """Check if a move will result in an extra turn.

        Args:
            env: The game environment.
            action: The action to check.

        Returns:
            True if the move will result in an extra turn, False otherwise.
        """
        player = env.current_player

        # Calculate the pit index
        pit_idx = action if player == 0 else env.num_pits + 1 + action
        stones = env.board[pit_idx]

        # Calculate where the last stone would land
        landing_idx = (pit_idx + stones) % len(env.board)

        # Check if landing in player's store
        player_store = env.num_pits if player == 0 else 2 * env.num_pits + 1
        return landing_idx == player_store

    def _evaluate_chain_potential(self, env: BaseEnvironment) -> float:
        """Evaluate the potential for chaining multiple moves together.

        This is more sophisticated than just counting extra turns, as it looks for
        specific patterns that allow a series of moves to be chained together.

        Args:
            env: The game environment.

        Returns:
            A value representing the chain potential.
        """
        player = env.current_player
        player_side_start = 0 if player == 0 else env.num_pits + 1
        player_side_end = env.num_pits - 1 if player == 0 else 2 * env.num_pits
        player_store = env.num_pits if player == 0 else 2 * env.num_pits + 1

        # To evaluate chains, we'll simulate sequences of moves
        # First, identify moves that give extra turns
        extra_turn_moves = []
        for action in env.get_valid_actions():
            if self._move_gives_extra_turn(env, action):
                extra_turn_moves.append(action)

        # If no extra turn moves, no chain potential
        if not extra_turn_moves:
            return 0.0

        # Try to find the longest possible chain
        max_chain_length = 0
        total_chain_value = 0.0

        # Try each extra turn move as a starting point
        for start_action in extra_turn_moves:
            # Create a simulation environment
            sim_env = copy.deepcopy(env)

            # Execute the first move
            _, _, terminated, _, _ = sim_env.step(start_action)

            # If game ends, skip
            if terminated:
                continue

            # Start chain with length 1
            chain_length = 1
            chain_value = 1.0  # Base value for a single extra turn

            # Keep making moves until no more extra turns
            while True:
                # Find next extra turn moves
                next_extra_turns = []
                for action in sim_env.get_valid_actions():
                    if self._move_gives_extra_turn(sim_env, action):
                        next_extra_turns.append(action)

                # If no more extra turns, break
                if not next_extra_turns:
                    break

                # Choose the best next move (prioritize captures)
                best_next_action = None
                best_next_value = -float('inf')

                for action in next_extra_turns:
                    # Prioritize capture moves
                    action_value = 1.0  # Base value
                    if self._is_capture_move(sim_env, action):
                        # Get the number of stones that would be captured
                        pit_idx = action if player == 0 else env.num_pits + 1 + action
                        stones = sim_env.board[pit_idx]
                        landing_idx = (pit_idx + stones) % len(sim_env.board)

                        # If landing in an empty pit on player's side
                        if (landing_idx >= player_side_start and
                                landing_idx <= player_side_end and
                                sim_env.board[landing_idx] == 0):

                            # Calculate the opposite pit
                            opposite_idx = 2 * env.num_pits - landing_idx

                            # Add value for the potential capture
                            if sim_env.board[opposite_idx] > 0:
                                action_value += sim_env.board[opposite_idx]

                    # Higher value for moves that can continue the chain
                    action_env = copy.deepcopy(sim_env)
                    _, _, a_terminated, _, _ = action_env.step(action)

                    if not a_terminated:
                        # Check if there are more extra turn moves after this
                        more_extra_turns = any(self._move_gives_extra_turn(action_env, a)
                                               for a in action_env.get_valid_actions())
                        if more_extra_turns:
                            action_value += 1.0

                    # Update best action
                    if action_value > best_next_value:
                        best_next_value = action_value
                        best_next_action = action

                # No valid next action (shouldn't happen)
                if best_next_action is None:
                    break

                # Execute the next action
                _, _, terminated, _, _ = sim_env.step(best_next_action)

                # If game ends, break
                if terminated:
                    break

                # Update chain
                chain_length += 1
                chain_value += best_next_value  # Add the value of this move

                # Avoid infinite loops (shouldn't happen in real games)
                if chain_length > 10:
                    break

            # Update max chain length
            if chain_length > max_chain_length:
                max_chain_length = chain_length
                total_chain_value = chain_value

        # Bonus for longer chains (exponential scaling)
        chain_bonus = 0.0
        if max_chain_length >= 2:
            chain_bonus = (2 ** (max_chain_length - 1)) * 0.5

        return total_chain_value + chain_bonus

    def _evaluate_sweep_potential(self, env: BaseEnvironment) -> float:
        """Evaluate the potential for sweeping all stones to your store.

        This is a special endgame evaluation that looks for positions where
        you can efficiently clear your side and potentially capture all
        opponent stones.

        Args:
            env: The game environment.

        Returns:
            A value representing the sweep potential.
        """
        player = env.current_player
        player_side_start = 0 if player == 0 else env.num_pits + 1
        player_side_end = env.num_pits - 1 if player == 0 else 2 * env.num_pits

        opponent = 1 - player
        opponent_side_start = 0 if opponent == 0 else env.num_pits + 1
        opponent_side_end = env.num_pits - 1 if opponent == 0 else 2 * env.num_pits

        # Count stones on each side
        player_stones = sum(env.board[player_side_start:player_side_end+1])
        opponent_stones = sum(env.board[opponent_side_start:opponent_side_end+1])

        # Count empty pits on player's side
        player_empty_pits = sum(1 for i in range(player_side_start, player_side_end + 1)
                                if env.board[i] == 0)

        # Count pits with exactly one stone (easy to clear)
        player_single_stone_pits = sum(1 for i in range(player_side_start, player_side_end + 1)
                                       if env.board[i] == 1)

        # If player's side is almost empty, high sweep potential
        sweep_value = 0.0

        # Base sweep value depends on how empty our side is
        empty_ratio = player_empty_pits / env.num_pits

        if empty_ratio > 0.5:  # More than half empty
            # Calculate a sweep score that increases as more pits are empty
            sweep_value = 2.0 + (empty_ratio - 0.5) * 8.0

            # Bonus for having many single-stone pits (easy to clear)
            sweep_value += player_single_stone_pits * 0.8

            # Huge bonus if we're almost done (only 1-2 pits with stones)
            non_empty = env.num_pits - player_empty_pits
            if non_empty <= 2:
                sweep_value += (3 - non_empty) * 3.0

            # Extra bonus if opponent has many stones (potential big capture)
            if opponent_stones > player_stones * 2:
                sweep_value += 2.0

        return sweep_value