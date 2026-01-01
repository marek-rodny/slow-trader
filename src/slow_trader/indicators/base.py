"""Base class for technical indicators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import pandas as pd
import numpy as np


@dataclass
class IndicatorResult:
    """Result from an indicator calculation."""

    name: str
    value: float | dict[str, float]
    signal: str | None = None  # 'buy', 'sell', or None
    strength: float = 0.0  # Signal strength 0-1


class Indicator(ABC):
    """Abstract base class for all technical indicators."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Calculate the indicator value.

        Args:
            data: DataFrame with OHLCV data (columns: open, high, low, close, volume)

        Returns:
            IndicatorResult with the calculated value
        """
        pass

    @abstractmethod
    def get_signal(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Get a trading signal based on the indicator.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            IndicatorResult with signal information
        """
        pass

    def validate_data(self, data: pd.DataFrame, min_periods: int = 1) -> bool:
        """
        Validate that the data has required columns and minimum periods.

        Args:
            data: DataFrame to validate
            min_periods: Minimum number of periods required

        Returns:
            True if valid, False otherwise
        """
        required_columns = ["close"]
        if not all(col in data.columns for col in required_columns):
            return False
        if len(data) < min_periods:
            return False
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


def ensure_series(data: pd.DataFrame | pd.Series, column: str = "close") -> pd.Series:
    """
    Ensure we have a pandas Series from DataFrame or Series input.

    Args:
        data: Input DataFrame or Series
        column: Column name to extract if DataFrame

    Returns:
        pandas Series
    """
    if isinstance(data, pd.DataFrame):
        return data[column]
    return data
