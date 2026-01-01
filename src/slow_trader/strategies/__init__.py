"""Trading strategies based on technical indicators."""

from slow_trader.strategies.base import Strategy, Signal
from slow_trader.strategies.ma_crossover import MACrossoverStrategy
from slow_trader.strategies.rsi_strategy import RSIStrategy
from slow_trader.strategies.macd_strategy import MACDStrategy
from slow_trader.strategies.combined import CombinedStrategy

__all__ = [
    "Strategy",
    "Signal",
    "MACrossoverStrategy",
    "RSIStrategy",
    "MACDStrategy",
    "CombinedStrategy",
]
