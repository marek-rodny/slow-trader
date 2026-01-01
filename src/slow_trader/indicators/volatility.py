"""Volatility indicators."""

import pandas as pd
import numpy as np
from slow_trader.indicators.base import Indicator, IndicatorResult, ensure_series


class BollingerBands(Indicator):
    """Bollinger Bands indicator."""

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Initialize Bollinger Bands indicator.

        Args:
            period: Moving average period
            std_dev: Number of standard deviations for bands
        """
        super().__init__(f"BB_{period}_{std_dev}")
        self.period = period
        self.std_dev = std_dev

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate Bollinger Bands values."""
        if not self.validate_data(data, self.period):
            return IndicatorResult(
                name=self.name,
                value={"upper": np.nan, "middle": np.nan, "lower": np.nan},
            )

        close = ensure_series(data, "close")
        upper, middle, lower = self._calculate_bands(close)

        return IndicatorResult(
            name=self.name,
            value={
                "upper": upper.iloc[-1],
                "middle": middle.iloc[-1],
                "lower": lower.iloc[-1],
            },
        )

    def get_signal(self, data: pd.DataFrame) -> IndicatorResult:
        """Get signal based on price position relative to bands."""
        if not self.validate_data(data, self.period):
            return IndicatorResult(
                name=self.name,
                value={"upper": np.nan, "middle": np.nan, "lower": np.nan},
            )

        close = ensure_series(data, "close")
        upper, middle, lower = self._calculate_bands(close)

        current_price = close.iloc[-1]
        upper_curr = upper.iloc[-1]
        middle_curr = middle.iloc[-1]
        lower_curr = lower.iloc[-1]

        signal = None
        strength = 0.0

        # Price near or below lower band - potential buy
        if current_price <= lower_curr:
            signal = "buy"
            strength = min((lower_curr - current_price) / lower_curr + 0.5, 1.0)

        # Price near or above upper band - potential sell
        elif current_price >= upper_curr:
            signal = "sell"
            strength = min((current_price - upper_curr) / upper_curr + 0.5, 1.0)

        # Calculate %B for additional context
        bandwidth = upper_curr - lower_curr
        percent_b = (current_price - lower_curr) / bandwidth if bandwidth > 0 else 0.5

        return IndicatorResult(
            name=self.name,
            value={
                "upper": upper_curr,
                "middle": middle_curr,
                "lower": lower_curr,
                "percent_b": percent_b,
            },
            signal=signal,
            strength=strength,
        )

    def _calculate_bands(self, close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate upper, middle, and lower bands."""
        middle = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()

        upper = middle + (std * self.std_dev)
        lower = middle - (std * self.std_dev)

        return upper, middle, lower

    def get_series(self, data: pd.DataFrame) -> dict[str, pd.Series]:
        """Get the full Bollinger Bands series."""
        close = ensure_series(data, "close")
        upper, middle, lower = self._calculate_bands(close)
        return {"upper": upper, "middle": middle, "lower": lower}


class ATR(Indicator):
    """Average True Range indicator."""

    def __init__(self, period: int = 14):
        """
        Initialize ATR indicator.

        Args:
            period: ATR calculation period
        """
        super().__init__(f"ATR_{period}")
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate ATR value."""
        if len(data) < self.period + 1:
            return IndicatorResult(name=self.name, value=np.nan)

        atr = self._calculate_atr(data)
        current_atr = atr.iloc[-1]

        return IndicatorResult(
            name=self.name,
            value=current_atr,
        )

    def get_signal(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Get volatility signal based on ATR.

        Note: ATR doesn't give buy/sell signals, but indicates volatility.
        """
        if len(data) < self.period + 1:
            return IndicatorResult(name=self.name, value=np.nan)

        atr = self._calculate_atr(data)
        current_atr = atr.iloc[-1]

        # Calculate ATR as percentage of price
        current_price = data["close"].iloc[-1]
        atr_percent = (current_atr / current_price) * 100

        # ATR can be used for position sizing and stop-loss placement
        return IndicatorResult(
            name=self.name,
            value=current_atr,
            signal=None,  # ATR doesn't provide direction
            strength=min(atr_percent / 5, 1.0),  # Normalize volatility
        )

    def _calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate ATR series."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # True Range components
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Wilder's smoothing
        atr = true_range.ewm(alpha=1 / self.period, adjust=False).mean()

        return atr

    def get_series(self, data: pd.DataFrame) -> pd.Series:
        """Get the full ATR series."""
        return self._calculate_atr(data)


class Volatility(Indicator):
    """Historical volatility indicator."""

    def __init__(self, period: int = 20):
        """
        Initialize Volatility indicator.

        Args:
            period: Calculation period
        """
        super().__init__(f"Volatility_{period}")
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate historical volatility."""
        if not self.validate_data(data, self.period):
            return IndicatorResult(name=self.name, value=np.nan)

        close = ensure_series(data, "close")
        volatility = self._calculate_volatility(close)

        return IndicatorResult(
            name=self.name,
            value=volatility.iloc[-1],
        )

    def get_signal(self, data: pd.DataFrame) -> IndicatorResult:
        """Get volatility level (not a directional signal)."""
        result = self.calculate(data)

        # High volatility might suggest caution
        # This is informational, not a trade signal
        return IndicatorResult(
            name=self.name,
            value=result.value,
            signal=None,
            strength=min(result.value / 50, 1.0) if not np.isnan(result.value) else 0.0,
        )

    def _calculate_volatility(self, close: pd.Series) -> pd.Series:
        """Calculate annualized historical volatility."""
        log_returns = np.log(close / close.shift(1))
        volatility = log_returns.rolling(window=self.period).std() * np.sqrt(252) * 100

        return volatility
