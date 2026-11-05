"""Happiness: a research assistant for your own corpus, running on your own machine, with a lexicon of happiness built in."""

from .agent import Agent, Settings
from .corpus import Corpus

__all__ = ["Agent", "Corpus", "Settings"]
__version__ = "0.2.0"
