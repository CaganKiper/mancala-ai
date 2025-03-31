"""Main script for playing Mancala games.

This script provides a command-line interface for playing different variants
of Mancala games, with options for human and AI players.
"""

import inspect
import sys
from typing import Callable, Dict, List, Optional, Tuple, Type

import mancala_ai.agents as agents_module
import mancala_ai.environments as environments_module
from mancala_ai.agents.base_agent import Agent
from mancala_ai.environments.base_environment import BaseEnvironment


def get_available_agents() -> Dict[str, Type[Agent]]:
    """Discover all available agent classes.

    Returns:
        A dictionary mapping agent names to agent classes.
    """
    agent_classes = {}

    # Inspect the agents module to find all Agent subclasses
    for name, obj in inspect.getmembers(agents_module):
        if (inspect.isclass(obj) and 
            issubclass(obj, Agent) and 
            obj != Agent):  # Exclude the base Agent class
            # Use the DISPLAY_NAME constant from the class
            display_name = obj.DISPLAY_NAME
            agent_classes[display_name] = obj

    return agent_classes


def get_available_environments() -> Dict[str, Type[BaseEnvironment]]:
    """Discover all available environment classes.

    Returns:
        A dictionary mapping environment names to environment classes.
    """
    env_classes = {}

    # Inspect the environments module to find all BaseEnvironment subclasses
    for name, obj in inspect.getmembers(environments_module):
        if (inspect.isclass(obj) and 
            issubclass(obj, BaseEnvironment) and 
            obj != BaseEnvironment):  # Exclude the base Environment class
            # Use the DISPLAY_NAME constant from the class
            display_name = obj.DISPLAY_NAME
            env_classes[display_name] = obj

    return env_classes


def choose_game_version() -> Tuple[str, Callable[..., BaseEnvironment]]:
    """Let the user choose a game version.

    Returns:
        A tuple containing the name of the game version and a function
        to create an instance of the corresponding environment.
    """
    # Get available environments
    available_envs = get_available_environments()

    if not available_envs:
        print("No game versions available.")
        sys.exit(1)

    # Create a list of environment names for display
    env_names = list(available_envs.keys())

    print("\nAvailable game versions:")
    for i, name in enumerate(env_names, 1):
        print(f"{i}. {name}")

    # Default to the first environment if there's only one
    default_choice = "1"

    while True:
        choice_prompt = f"Choose a game version ({default_choice}): " if len(env_names) > 1 else f"Press Enter to select {env_names[0]}: "
        choice = input(choice_prompt)

        # Default to first option if empty input
        if choice == "":
            choice = default_choice

        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(env_names):
                selected_name = env_names[choice_idx]
                selected_class = available_envs[selected_name]
                return selected_name, selected_class
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(env_names)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def choose_player(player_num: int) -> Agent:
    """Let the user choose a player type.

    Args:
        player_num: The player number (1 or 2).

    Returns:
        An instance of the chosen agent type.
    """
    # Get available agents
    available_agents = get_available_agents()

    if not available_agents:
        print("No agent types available.")
        sys.exit(1)

    # Create a list of agent names for display
    agent_names = list(available_agents.keys())

    print(f"\nChoose Player {player_num}:")
    for i, name in enumerate(agent_names, 1):
        print(f"{i}. {name}")

    # Default to Human if available, otherwise first agent
    default_choice = "1"
    human_idx = next((i for i, name in enumerate(agent_names, 1) if name.lower() == "human"), None)
    if human_idx is not None:
        default_choice = str(human_idx)

    while True:
        choice_prompt = f"Select Player {player_num} type ({default_choice}): "
        choice = input(choice_prompt)

        # Default to default choice if empty input
        if choice == "":
            choice = default_choice

        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(agent_names):
                selected_name = agent_names[choice_idx]
                selected_class = available_agents[selected_name]

                # For Human agent, ask for a name
                if selected_name.lower() == "human":
                    name = input(f"Enter name for Player {player_num} (default: Player {player_num}): ")
                    if name == "":
                        name = f"Player {player_num}"
                    return selected_class(name)
                else:
                    return selected_class(f"{selected_name} Player {player_num}")
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(agent_names)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def play_game(env: BaseEnvironment, players: List[Agent]) -> Optional[int]:
    """Play a game of Mancala.

    Args:
        env: The game environment.
        players: List of agents (should have exactly 2 agents).

    Returns:
        The index of the winning player (0 or 1), or None if it's a draw.
    """
    # Reset the environment
    env.reset()
    print("\nGame started!")

    # Main game loop
    while not env.done:
        # Render the current state
        env.render()

        # Get the current player
        current_player = players[env.current_player]

        # Get action from the current player
        action = current_player.get_action(env)

        # Check if the player wants to quit
        if action == -1:
            print("Game aborted.")
            return None

        # Take a step in the environment
        _, reward, terminated, _, _ = env.step(action)

        print(f"{current_player.name} chose pit {action + 1}")

        # Optional: add a small delay or prompt to continue for better UX
        if not terminated and current_player.is_interactive:
            input("Press Enter to continue...")

    # Render the final state
    env.render()

    # Determine the winner
    winner = env.get_winner()
    if winner is not None:
        winning_player = players[winner]
        print(f"\n{winning_player.name} wins!")
    else:
        print("\nIt's a draw!")

    return winner


def main():
    """Main function to run the Mancala game."""
    print("Welcome to Mancala AI!")

    playing = True
    while playing:
        # Choose game version
        game_name, game_class = choose_game_version()

        # Create game environment
        env = game_class()

        # Choose players
        players = [
            choose_player(1),
            choose_player(2)
        ]

        # Play the game
        play_game(env, players)

        # Ask if the user wants to play again
        while True:
            choice = input("\nWhat would you like to do?\n1. Play again\n2. Change settings\n3. Exit\nYour choice: ")

            if choice == "1":
                # Play again with the same settings
                break
            elif choice == "2":
                # Change settings (will go back to the beginning of the outer loop)
                break
            elif choice == "3":
                # Exit
                playing = False
                break
            else:
                print("Invalid choice. Please try again.")

    print("Thanks for playing Mancala AI!")


if __name__ == "__main__":
    main()
