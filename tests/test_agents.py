"""Tests for the agent classes.

This module contains tests for the agent classes, including RandomAgent and HumanAgent.
"""

import pytest
import builtins
from unittest.mock import patch, MagicMock

from mancala_ai.agents.base_agent import Agent
from mancala_ai.agents.random_agent import RandomAgent
from mancala_ai.agents.human_agent import HumanAgent
from mancala_ai.environments.base_environment import BaseEnvironment


class TestAgent:
    """Tests for the base Agent class."""

    def test_initialization(self):
        """Test that the agent initializes correctly."""
        # The base Agent class is abstract, so we need to create a concrete subclass for testing
        class TestConcreteAgent(Agent):
            def get_action(self, env):
                return 0

        agent = TestConcreteAgent(name="Test Agent")
        assert agent.name == "Test Agent"
        assert not agent.is_interactive

    def test_get_action_not_implemented(self):
        """Test that the base get_action method raises NotImplementedError."""
        # Create an agent without implementing get_action
        agent = Agent(name="Test Agent")
        
        # Attempting to call get_action should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            agent.get_action(None)


class TestRandomAgent:
    """Tests for the RandomAgent class."""

    def test_initialization(self):
        """Test that the random agent initializes correctly."""
        agent = RandomAgent(name="Test Random Agent")
        assert agent.name == "Test Random Agent"
        assert not agent.is_interactive

    def test_get_action_with_valid_actions(self, base_environment, monkeypatch):
        """Test that get_action returns a valid action."""
        # Create a random agent
        agent = RandomAgent(name="Test Random Agent")
        
        # Mock the random.choice function to return a specific action
        monkeypatch.setattr("random.choice", lambda x: x[0])
        
        # Get an action
        action = agent.get_action(base_environment)
        
        # The action should be the first valid action
        assert action == 0

    def test_get_action_with_no_valid_actions(self, base_environment):
        """Test that get_action returns -1 when there are no valid actions."""
        # Create a random agent
        agent = RandomAgent(name="Test Random Agent")
        
        # Modify the environment to have no valid actions
        base_environment.board = [0] * len(base_environment.board)
        
        # Get an action
        action = agent.get_action(base_environment)
        
        # The action should be -1 (no valid actions)
        assert action == -1


class TestHumanAgent:
    """Tests for the HumanAgent class."""

    def test_initialization(self):
        """Test that the human agent initializes correctly."""
        agent = HumanAgent(name="Test Human Agent")
        assert agent.name == "Test Human Agent"
        assert agent.is_interactive

    @patch("builtins.input", return_value="1")
    def test_get_action_valid_input(self, mock_input, base_environment):
        """Test that get_action returns the correct action for valid input."""
        # Create a human agent
        agent = HumanAgent(name="Test Human Agent")
        
        # Get an action
        action = agent.get_action(base_environment)
        
        # The action should be 0 (1-1, since input is 1-indexed but actions are 0-indexed)
        assert action == 0
        
        # Check that input was called with the correct prompt
        mock_input.assert_called_once()

    @patch("builtins.input", side_effect=["invalid", "1"])
    def test_get_action_invalid_then_valid_input(self, mock_input, base_environment):
        """Test that get_action handles invalid input and then accepts valid input."""
        # Create a human agent
        agent = HumanAgent(name="Test Human Agent")
        
        # Get an action
        action = agent.get_action(base_environment)
        
        # The action should be 0 (1-1, since input is 1-indexed but actions are 0-indexed)
        assert action == 0
        
        # Check that input was called twice
        assert mock_input.call_count == 2

    @patch("builtins.input", return_value="q")
    def test_get_action_quit(self, mock_input, base_environment):
        """Test that get_action returns -1 when the user wants to quit."""
        # Create a human agent
        agent = HumanAgent(name="Test Human Agent")
        
        # Get an action
        action = agent.get_action(base_environment)
        
        # The action should be -1 (quit)
        assert action == -1
        
        # Check that input was called with the correct prompt
        mock_input.assert_called_once()

    @patch("builtins.input", return_value="7")
    def test_get_action_invalid_action(self, mock_input, base_environment):
        """Test that get_action handles invalid actions."""
        # Create a human agent
        agent = HumanAgent(name="Test Human Agent")
        
        # Mock print to capture output
        with patch("builtins.print") as mock_print:
            # Get an action (this will loop forever with our mock, so we need to patch it again)
            with patch("builtins.input", side_effect=["7", "1"]):
                action = agent.get_action(base_environment)
        
        # The action should be 0 (1-1, since input is 1-indexed but actions are 0-indexed)
        assert action == 0
        
        # Check that an error message was printed
        mock_print.assert_called()