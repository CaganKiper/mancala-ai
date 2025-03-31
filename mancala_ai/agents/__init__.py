"""Agents for Mancala game.

This package contains agent implementations for Mancala games,
including human agents and AI agents.
"""

from mancala_ai.agents.base_agent import Agent
from mancala_ai.agents.human_agent import HumanAgent
from mancala_ai.agents.random_agent import RandomAgent

__all__ = ["Agent", "HumanAgent", "RandomAgent"]
