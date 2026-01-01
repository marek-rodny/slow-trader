"""Moving average indicators."""

import pandas as pd
import numpy as np
from slow_trader.indicators.base import Indicator, IndicatorResult, ensure_series


class SMA(Indicator):
    """Simple Moving Average indicator."""

    def __init__(self, period: int = 20, column: str = "close"):
        """
        Initialize SMA indicator.

        Args:
            period: Number of periods for the moving average
            column: Column to calculate SMA on
        """
        super().__init__(f"SMA_{period}")
        self.period = period
        self.column = column

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate the SMA value."""
        if not self.validate_data(data, self.period):
            return IndicatorResult(name=self.name, value=np.nan)

        series = ensure_series(data, self.column)
        sma = series.rolling(window=self.period).mean()
        current_value = sma.iloc[-1]

        return IndicatorResult(
            name=self.name,
            value=current_value,
        )

    def get_signal(self, data: pd.DataFrame) -> IndicatorResult:
        """Get signal based on price vs SMA."""
        if not self.validate_data(data, self.period):
            return IndicatorResult(name=self.name, value=np.nan)

        series = ensure_series(data, self.column)
        sma = series.rolling(window=self.period).mean()
        current_price = series.iloc[-1]
        current_sma = sma.iloc[-1]

        # Signal: price above SMA is bullish, below is bearish
        signal = None
        strength = 0.0

        if current_price > current_sma:
            signal = "buy"
            strength = min((current_price - current_sma) / current_sma, 0.1) * 10
        elif current_price < current_sma:
            signal = "sell"
            strength = min((current_sma - current_price) / current_sma, 0.1) * 10

        return IndicatorResult(
            name=self.name,
            value=current_sma,
            signal=signal,
            strength=min(strength, 1.0),
        )

    def get_series(self, data: pd.DataFrame) -> pd.Series:
        """Get the full SMA series."""
        series = ensure_series(data, self.column)
        return series.rolling(window=self.period).mean()


class EMA(Indicator):
    """Exponential Moving Average indicator."""

    def __init__(self, period: int = 20, column: str = "close"):
        """
        Initialize EMA indicator.

        Args:
            period: Number of periods for the moving average
            column: Column to calculate EMA on
        """
        super().__init__(f"EMA_{period}")
        self.period = period
        self.column = column

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate the EMA value."""
        if not self.validate_data(data, self.period):
            return IndicatorResult(name=self.name, value=np.nan)

        series = ensure_series(data, self.column)
        ema = series.ewm(span=self.period, adjust=False).mean()
        current_value = ema.iloc[-1]

        return IndicatorResult(
            name=self.name,
            value=current_value,
        )

    def get_signal(self, data: pd.DataFrame) -> IndicatorResult:
        """Get signal based on price vs EMA."""
        if not self.validate_data(data, self.period):
            return IndicatorResult(name=self.name, value=np.nan)

        series = ensure_series(data, self.column)
        ema = series.ewm(span=self.period, adjust=False).mean()
        current_price = series.iloc[-1]
        current_ema = ema.iloc[-1]

        signal = None
        strength = 0.0

        if current_price > current_ema:
            signal = "buy"
            strength = min((current_price - current_ema) / current_ema, 0.1) * 10
        elif current_price < current_ema:
            signal = "sell"
            strength = min((current_ema - current_price) / current_ema, 0.1) * 10

        return IndicatorResult(
            name=self.name,
            value=current_ema,
            signal=signal,
            strength=min(strength, 1.0),
        )

    def get_series(self, data: pd.DataFrame) -> pd.Series:
        """Get the full EMA series."""
        series = ensure_series(data, self.column)
        return series.ewm(span=self.period, adjust=False).mean()


class MACrossover:
    """
    Moving Average Crossover detector.

    Detects when a fast MA crosses above/below a slow MA.
    """

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 20,
        ma_type: str = "ema",
    ):
        """
        Initialize MA Crossover detector.

        Args:
            fast_period: Period for fast moving average
            slow_period: Period for slow moving average
            ma_type: Type of MA ('sma' or 'ema')
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.ma_type = ma_type.lower()

        if self.ma_type == "ema":
            self.fast_ma = EMA(fast_period)
            self.slow_ma = EMA(slow_period)
        else:
            self.fast_ma = SMA(fast_period)
            self.slow_ma = SMA(slow_period)

        self.name = f"MA_Crossover_{fast_period}_{slow_period}"

    def detect_crossover(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Detect MA crossover signals.

        Returns:
            IndicatorResult with:
            - 'buy' signal on golden cross (fast crosses above slow)
            - 'sell' signal on death cross (fast crosses below slow)
        """
        min_periods = max(self.fast_period, self.slow_period) + 1
        if len(data) < min_periods:
            return IndicatorResult(
                name=self.name,
                value={"fast": np.nan, "slow": np.nan},
            )

        fast_series = self.fast_ma.get_series(data)
        slow_series = self.slow_ma.get_series(data)

        # Current and previous values
        fast_curr = fast_series.iloc[-1]
        fast_prev = fast_series.iloc[-2]
        slow_curr = slow_series.iloc[-1]
        slow_prev = slow_series.iloc[-2]

        signal = None
        strength = 0.0

        # Golden cross: fast crosses above slow
        if fast_prev <= slow_prev and fast_curr > slow_curr:
            signal = "buy"
            strength = min(abs(fast_curr - slow_curr) / slow_curr * 100, 1.0)

        # Death cross: fast crosses below slow
        elif fast_prev >= slow_prev and fast_curr < slow_curr:
            signal = "sell"
            strength = min(abs(slow_curr - fast_curr) / slow_curr * 100, 1.0)

        return IndicatorResult(
            name=self.name,
            value={"fast": fast_curr, "slow": slow_curr},
            signal=signal,
            strength=strength,
        )
