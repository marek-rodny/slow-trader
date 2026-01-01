"""
Slow Trader - A rule-based trading bot for stocks, crypto, and forex.

This bot monitors prices, applies technical indicators, and places trades
based on configurable strategies. Designed for slow, thoughtful trading
rather than high-frequency operations.
"""

__version__ = "1.0.0"
__author__ = "Slow Trader"

from slow_trader.config import Config
from slow_trader.bot import TradingBot

__all__ = ["Config", "TradingBot", "__version__"]
