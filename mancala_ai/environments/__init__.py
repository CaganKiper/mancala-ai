"""Environments for Mancala game.

This package contains environment implementations for Mancala games.
"""

from mancala_ai.environments.base_environment import BaseEnvironment
from mancala_ai.environments.kalah_environment import KalahEnvironment

__all__ = ["BaseEnvironment", "KalahEnvironment"]
