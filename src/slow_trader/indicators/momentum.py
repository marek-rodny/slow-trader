"""Momentum indicators."""

import pandas as pd
import numpy as np
from slow_trader.indicators.base import Indicator, IndicatorResult, ensure_series


class RSI(Indicator):
    """Relative Strength Index indicator."""

    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30):
        """
        Initialize RSI indicator.

        Args:
            period: RSI calculation period
            overbought: Overbought threshold (typically 70)
            oversold: Oversold threshold (typically 30)
        """
        super().__init__(f"RSI_{period}")
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate the RSI value."""
        if not self.validate_data(data, self.period + 1):
            return IndicatorResult(name=self.name, value=np.nan)

        close = ensure_series(data, "close")
        rsi = self._calculate_rsi(close)
        current_rsi = rsi.iloc[-1]

        return IndicatorResult(
            name=self.name,
            value=current_rsi,
        )

    def get_signal(self, data: pd.DataFrame) -> IndicatorResult:
        """Get signal based on RSI levels."""
        if not self.validate_data(data, self.period + 1):
            return IndicatorResult(name=self.name, value=np.nan)

        close = ensure_series(data, "close")
        rsi = self._calculate_rsi(close)
        current_rsi = rsi.iloc[-1]

        signal = None
        strength = 0.0

        if current_rsi <= self.oversold:
            # Oversold - potential buy
            signal = "buy"
            strength = (self.oversold - current_rsi) / self.oversold
        elif current_rsi >= self.overbought:
            # Overbought - potential sell
            signal = "sell"
            strength = (current_rsi - self.overbought) / (100 - self.overbought)

        return IndicatorResult(
            name=self.name,
            value=current_rsi,
            signal=signal,
            strength=min(strength, 1.0),
        )

    def _calculate_rsi(self, close: pd.Series) -> pd.Series:
        """Calculate RSI series."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()

        # Use Wilder's smoothing for subsequent values
        for i in range(self.period, len(close)):
            avg_gain.iloc[i] = (
                avg_gain.iloc[i - 1] * (self.period - 1) + gain.iloc[i]
            ) / self.period
            avg_loss.iloc[i] = (
                avg_loss.iloc[i - 1] * (self.period - 1) + loss.iloc[i]
            ) / self.period

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def get_series(self, data: pd.DataFrame) -> pd.Series:
        """Get the full RSI series."""
        close = ensure_series(data, "close")
        return self._calculate_rsi(close)


class MACD(Indicator):
    """Moving Average Convergence Divergence indicator."""

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ):
        """
        Initialize MACD indicator.

        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
        """
        super().__init__(f"MACD_{fast_period}_{slow_period}_{signal_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate the MACD values."""
        min_periods = self.slow_period + self.signal_period
        if not self.validate_data(data, min_periods):
            return IndicatorResult(
                name=self.name,
                value={"macd": np.nan, "signal": np.nan, "histogram": np.nan},
            )

        close = ensure_series(data, "close")
        macd, signal, histogram = self._calculate_macd(close)

        return IndicatorResult(
            name=self.name,
            value={
                "macd": macd.iloc[-1],
                "signal": signal.iloc[-1],
                "histogram": histogram.iloc[-1],
            },
        )

    def get_signal(self, data: pd.DataFrame) -> IndicatorResult:
        """Get signal based on MACD crossover."""
        min_periods = self.slow_period + self.signal_period + 1
        if not self.validate_data(data, min_periods):
            return IndicatorResult(
                name=self.name,
                value={"macd": np.nan, "signal": np.nan, "histogram": np.nan},
            )

        close = ensure_series(data, "close")
        macd, signal_line, histogram = self._calculate_macd(close)

        # Current and previous values
        macd_curr = macd.iloc[-1]
        macd_prev = macd.iloc[-2]
        signal_curr = signal_line.iloc[-1]
        signal_prev = signal_line.iloc[-2]

        signal = None
        strength = 0.0

        # Bullish crossover: MACD crosses above signal
        if macd_prev <= signal_prev and macd_curr > signal_curr:
            signal = "buy"
            strength = min(abs(macd_curr - signal_curr) / abs(signal_curr) if signal_curr != 0 else 0.5, 1.0)

        # Bearish crossover: MACD crosses below signal
        elif macd_prev >= signal_prev and macd_curr < signal_curr:
            signal = "sell"
            strength = min(abs(signal_curr - macd_curr) / abs(signal_curr) if signal_curr != 0 else 0.5, 1.0)

        return IndicatorResult(
            name=self.name,
            value={
                "macd": macd_curr,
                "signal": signal_curr,
                "histogram": histogram.iloc[-1],
            },
            signal=signal,
            strength=strength,
        )

    def _calculate_macd(self, close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD, signal line, and histogram."""
        fast_ema = close.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow_period, adjust=False).mean()

        macd = fast_ema - slow_ema
        signal_line = macd.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd - signal_line

        return macd, signal_line, histogram

    def get_series(self, data: pd.DataFrame) -> dict[str, pd.Series]:
        """Get the full MACD series."""
        close = ensure_series(data, "close")
        macd, signal, histogram = self._calculate_macd(close)
        return {"macd": macd, "signal": signal, "histogram": histogram}


class Stochastic(Indicator):
    """Stochastic Oscillator indicator."""

    def __init__(
        self,
        k_period: int = 14,
        d_period: int = 3,
        overbought: float = 80,
        oversold: float = 20,
    ):
        """
        Initialize Stochastic indicator.

        Args:
            k_period: %K period
            d_period: %D smoothing period
            overbought: Overbought threshold
            oversold: Oversold threshold
        """
        super().__init__(f"Stochastic_{k_period}_{d_period}")
        self.k_period = k_period
        self.d_period = d_period
        self.overbought = overbought
        self.oversold = oversold

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate the Stochastic values."""
        if len(data) < self.k_period + self.d_period:
            return IndicatorResult(
                name=self.name,
                value={"k": np.nan, "d": np.nan},
            )

        k, d = self._calculate_stochastic(data)

        return IndicatorResult(
            name=self.name,
            value={"k": k.iloc[-1], "d": d.iloc[-1]},
        )

    def get_signal(self, data: pd.DataFrame) -> IndicatorResult:
        """Get signal based on Stochastic crossover and levels."""
        if len(data) < self.k_period + self.d_period + 1:
            return IndicatorResult(
                name=self.name,
                value={"k": np.nan, "d": np.nan},
            )

        k, d = self._calculate_stochastic(data)

        k_curr = k.iloc[-1]
        k_prev = k.iloc[-2]
        d_curr = d.iloc[-1]
        d_prev = d.iloc[-2]

        signal = None
        strength = 0.0

        # Buy signal: %K crosses above %D in oversold zone
        if k_prev <= d_prev and k_curr > d_curr and k_curr < self.oversold:
            signal = "buy"
            strength = (self.oversold - k_curr) / self.oversold

        # Sell signal: %K crosses below %D in overbought zone
        elif k_prev >= d_prev and k_curr < d_curr and k_curr > self.overbought:
            signal = "sell"
            strength = (k_curr - self.overbought) / (100 - self.overbought)

        return IndicatorResult(
            name=self.name,
            value={"k": k_curr, "d": d_curr},
            signal=signal,
            strength=min(strength, 1.0),
        )

    def _calculate_stochastic(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Calculate %K and %D."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        lowest_low = low.rolling(window=self.k_period).min()
        highest_high = high.rolling(window=self.k_period).max()

        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=self.d_period).mean()

        return k, d
