# Mancala AI

A Python implementation of the Mancala game with AI agents using traditional game theory algorithms and reinforcement learning techniques.

## Project Overview

Mancala AI is a comprehensive implementation of the classic Mancala board game with a focus on AI agent development. The project features:

- A modular, extensible architecture for implementing different Mancala variants
- Multiple AI agent implementations with varying levels of sophistication
- A user-friendly command-line interface for playing against AI or human opponents
- A framework designed with reinforcement learning principles

### What is Mancala?

Mancala is one of the oldest known board games, with evidence of its play dating back to ancient Egypt. The game involves moving stones around a board with the goal of capturing more stones than your opponent.

This implementation focuses on the Kalah variant of Mancala, which has the following rules:

1. The board consists of two rows of pits, with a store (or "kalah") at each end
2. Each player controls the pits on their side and their store on the right
3. Players take turns picking up all stones from one of their pits and distributing them counter-clockwise, one stone per pit
4. Players skip their opponent's store when distributing stones
5. If the last stone lands in the player's store, they get an extra turn
6. If the last stone lands in an empty pit on the player's side, they capture that stone and all stones in the opposite pit
7. The game ends when all pits on one side are empty
8. The player with the most stones in their store wins

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mancala-ai.git
   cd mancala-ai
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install numpy  # Main dependency for array operations
   ```

## Usage

### Playing the Game

To play the game, run:

```bash
python -m mancala_ai.main
```

Follow the on-screen instructions to:
1. Choose a game variant (currently only Kalah is implemented)
2. Select player types for both players (Human, Random Agent, or Minimax Agent)
3. Configure agent settings if applicable (e.g., search depth for Minimax Agent)
4. Play the game by selecting pits (1-6) when prompted

During gameplay:
- The board is displayed with Player 1's pits on top and Player 2's pits on the bottom
- Stores are shown at the ends of each row
- Valid moves are displayed before each turn
- Enter 'q', 'quit', or 'exit' to quit the game at any time

### Agent Types

The project includes the following agent types:

1. **Human Agent**: Allows a human player to make moves through the command line
2. **Random Agent**: Makes random valid moves (useful as a baseline)
3. **Minimax Agent**: Uses the Minimax algorithm with Alpha-Beta pruning to make strategic decisions
   - Configurable search depth (higher depth = stronger play but slower decisions)
   - Includes optimizations like transposition tables and move ordering

### Training AI Agents (Planned Feature)

The project includes a placeholder for training reinforcement learning agents (`train.py`), but this functionality is not yet implemented.

## Project Structure

```
mancala_ai/
├── agents/                 # AI agent implementations
│   ├── __init__.py         # Agent module initialization
│   ├── base_agent.py       # Base class for all agents
│   ├── human_agent.py      # Human player implementation
│   ├── minimax_agent.py    # Minimax algorithm with Alpha-Beta pruning
│   └── random_agent.py     # Random move selection agent
├── environments/           # Game environment implementations
│   ├── __init__.py         # Environment module initialization
│   ├── base_environment.py # Base class for all environments
│   └── kalah_environment.py # Kalah variant implementation
├── main.py                 # CLI entry point for playing games
└── train.py                # Training script (placeholder for future development)

tests/                      # Test suite
├── conftest.py             # Pytest fixtures
├── test_agents.py          # Tests for agent classes
├── test_environment.py     # Tests for base environment
├── test_kalah_environment.py # Tests for Kalah environment
├── test_minimax_agent.py   # Tests for Minimax agent
└── test_minimax_fix.py     # Additional tests for Minimax agent

run_tests.py                # Script to run tests with coverage reporting
```

### Key Components

1. **Agents**: Classes that implement different strategies for playing the game
   - All agents inherit from the `Agent` base class
   - Agents implement the `get_action` method to select moves

2. **Environments**: Classes that implement the game rules and mechanics
   - All environments inherit from the `BaseEnvironment` class
   - Environments follow the OpenAI Gym interface pattern
   - Key methods include `reset()`, `step(action)`, and `render()`

3. **Main Entry Point**: The `main.py` script provides a CLI for playing the game
   - Dynamically discovers available agents and environments
   - Handles user input and game flow

## Development Guidelines

### Adding a New Agent

To add a new agent:

1. Create a new file in the `mancala_ai/agents/` directory
2. Define a class that inherits from `Agent`
3. Implement the `get_action` method
4. Set a `DISPLAY_NAME` class variable for UI display
5. Optionally implement `get_settings` and `set_setting` for configurable parameters

Example:
```python
from mancala_ai.agents.base_agent import Agent

class MyNewAgent(Agent):
    DISPLAY_NAME = "My New Agent"

    def __init__(self, name):
        super().__init__(name)
        # Initialize agent-specific attributes

    def get_action(self, env):
        # Implement your strategy here
        valid_actions = env.get_valid_actions()
        return valid_actions[0]  # Example: always choose the first valid action
```

### Adding a New Environment

To add a new Mancala variant:

1. Create a new file in the `mancala_ai/environments/` directory
2. Define a class that inherits from `BaseEnvironment`
3. Override methods as needed to implement variant-specific rules
4. Set a `DISPLAY_NAME` class variable for UI display

Example:
```python
from mancala_ai.environments.base_environment import BaseEnvironment

class MyVariantEnvironment(BaseEnvironment):
    DISPLAY_NAME = "My Mancala Variant"

    def _execute_move(self, action):
        # Implement variant-specific move execution
        # ...
        return reward
```

## Testing

This project follows test-driven development (TDD) principles with a comprehensive test suite implemented using pytest.

### Running Tests

To run the tests with coverage reporting:

```bash
python run_tests.py
```

This will:
1. Run all tests in the `tests/` directory
2. Generate a terminal coverage report
3. Generate an HTML coverage report in the `coverage_html/` directory

### Writing Tests

When adding new features or fixing bugs:

1. Write tests first to define the expected behavior
2. Implement the feature or fix
3. Run the tests to ensure the implementation meets the requirements
4. Refactor as needed while maintaining test coverage

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Future Development

Planned features for future development:

1. Implementation of reinforcement learning agents (Q-learning, Deep Q-Network)
2. Additional Mancala variants (Oware, Congkak, etc.)
3. Graphical user interface
4. Performance optimizations for AI agents
5. Multi-game tournaments and agent comparison tools

## License

This project is licensed under the MIT License - see the LICENSE file for details.
