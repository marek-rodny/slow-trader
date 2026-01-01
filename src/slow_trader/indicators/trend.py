"""Trend indicators."""

import pandas as pd
import numpy as np
from slow_trader.indicators.base import Indicator, IndicatorResult, ensure_series


class ADX(Indicator):
    """Average Directional Index indicator."""

    def __init__(self, period: int = 14):
        """
        Initialize ADX indicator.

        Args:
            period: ADX calculation period
        """
        super().__init__(f"ADX_{period}")
        self.period = period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate ADX value."""
        min_periods = self.period * 2
        if len(data) < min_periods:
            return IndicatorResult(
                name=self.name,
                value={"adx": np.nan, "plus_di": np.nan, "minus_di": np.nan},
            )

        adx, plus_di, minus_di = self._calculate_adx(data)

        return IndicatorResult(
            name=self.name,
            value={
                "adx": adx.iloc[-1],
                "plus_di": plus_di.iloc[-1],
                "minus_di": minus_di.iloc[-1],
            },
        )

    def get_signal(self, data: pd.DataFrame) -> IndicatorResult:
        """Get signal based on ADX and DI crossover."""
        min_periods = self.period * 2 + 1
        if len(data) < min_periods:
            return IndicatorResult(
                name=self.name,
                value={"adx": np.nan, "plus_di": np.nan, "minus_di": np.nan},
            )

        adx, plus_di, minus_di = self._calculate_adx(data)

        adx_curr = adx.iloc[-1]
        plus_di_curr = plus_di.iloc[-1]
        plus_di_prev = plus_di.iloc[-2]
        minus_di_curr = minus_di.iloc[-1]
        minus_di_prev = minus_di.iloc[-2]

        signal = None
        strength = 0.0

        # Strong trend threshold
        if adx_curr >= 25:
            # +DI crosses above -DI: bullish
            if plus_di_prev <= minus_di_prev and plus_di_curr > minus_di_curr:
                signal = "buy"
                strength = min(adx_curr / 50, 1.0)

            # -DI crosses above +DI: bearish
            elif minus_di_prev <= plus_di_prev and minus_di_curr > plus_di_curr:
                signal = "sell"
                strength = min(adx_curr / 50, 1.0)

        return IndicatorResult(
            name=self.name,
            value={
                "adx": adx_curr,
                "plus_di": plus_di_curr,
                "minus_di": minus_di_curr,
            },
            signal=signal,
            strength=strength,
        )

    def _calculate_adx(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate ADX, +DI, and -DI."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Smooth with Wilder's method
        alpha = 1 / self.period
        atr = true_range.ewm(alpha=alpha, adjust=False).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()

        # Calculate +DI and -DI
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr

        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=alpha, adjust=False).mean()

        return adx, plus_di, minus_di


class TrendSignal(Indicator):
    """
    Combined trend detection using multiple methods.

    Uses price action, moving averages, and momentum to determine trend.
    """

    def __init__(
        self,
        short_period: int = 10,
        medium_period: int = 20,
        long_period: int = 50,
    ):
        """
        Initialize TrendSignal indicator.

        Args:
            short_period: Short-term MA period
            medium_period: Medium-term MA period
            long_period: Long-term MA period
        """
        super().__init__("TrendSignal")
        self.short_period = short_period
        self.medium_period = medium_period
        self.long_period = long_period

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate trend strength and direction."""
        if len(data) < self.long_period:
            return IndicatorResult(
                name=self.name,
                value={"trend": "neutral", "strength": 0},
            )

        close = ensure_series(data, "close")

        # Calculate EMAs
        short_ema = close.ewm(span=self.short_period, adjust=False).mean()
        medium_ema = close.ewm(span=self.medium_period, adjust=False).mean()
        long_ema = close.ewm(span=self.long_period, adjust=False).mean()

        # Current values
        price = close.iloc[-1]
        short = short_ema.iloc[-1]
        medium = medium_ema.iloc[-1]
        long_val = long_ema.iloc[-1]

        # Determine trend
        trend = self._determine_trend(price, short, medium, long_val)
        strength = self._calculate_strength(price, short, medium, long_val)

        return IndicatorResult(
            name=self.name,
            value={
                "trend": trend,
                "strength": strength,
                "short_ema": short,
                "medium_ema": medium,
                "long_ema": long_val,
            },
        )

    def get_signal(self, data: pd.DataFrame) -> IndicatorResult:
        """Get trend-based trading signal."""
        result = self.calculate(data)

        if isinstance(result.value, dict):
            trend = result.value.get("trend", "neutral")
            strength = result.value.get("strength", 0)

            signal = None
            if trend == "strong_uptrend":
                signal = "buy"
            elif trend == "strong_downtrend":
                signal = "sell"

            return IndicatorResult(
                name=self.name,
                value=result.value,
                signal=signal,
                strength=strength,
            )

        return result

    def _determine_trend(
        self,
        price: float,
        short: float,
        medium: float,
        long_val: float,
    ) -> str:
        """Determine trend based on MA alignment."""
        # Strong uptrend: price > short > medium > long
        if price > short > medium > long_val:
            return "strong_uptrend"

        # Uptrend: price > medium and short > long
        if price > medium and short > long_val:
            return "uptrend"

        # Strong downtrend: price < short < medium < long
        if price < short < medium < long_val:
            return "strong_downtrend"

        # Downtrend: price < medium and short < long
        if price < medium and short < long_val:
            return "downtrend"

        return "neutral"

    def _calculate_strength(
        self,
        price: float,
        short: float,
        medium: float,
        long_val: float,
    ) -> float:
        """Calculate trend strength (0-1)."""
        # Measure alignment and separation
        if long_val == 0:
            return 0.0

        # Calculate percentage separations
        short_sep = abs(short - long_val) / long_val
        medium_sep = abs(medium - long_val) / long_val
        price_sep = abs(price - long_val) / long_val

        # Average separation as strength indicator
        avg_sep = (short_sep + medium_sep + price_sep) / 3

        # Normalize to 0-1 range (5% separation = full strength)
        return min(avg_sep / 0.05, 1.0)


class SuperTrend(Indicator):
    """SuperTrend indicator for trend following."""

    def __init__(self, period: int = 10, multiplier: float = 3.0):
        """
        Initialize SuperTrend indicator.

        Args:
            period: ATR period
            multiplier: ATR multiplier for bands
        """
        super().__init__(f"SuperTrend_{period}_{multiplier}")
        self.period = period
        self.multiplier = multiplier

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate SuperTrend value."""
        if len(data) < self.period + 1:
            return IndicatorResult(
                name=self.name,
                value={"supertrend": np.nan, "direction": 0},
            )

        supertrend, direction = self._calculate_supertrend(data)

        return IndicatorResult(
            name=self.name,
            value={
                "supertrend": supertrend.iloc[-1],
                "direction": direction.iloc[-1],
            },
        )

    def get_signal(self, data: pd.DataFrame) -> IndicatorResult:
        """Get signal based on SuperTrend direction change."""
        if len(data) < self.period + 2:
            return IndicatorResult(
                name=self.name,
                value={"supertrend": np.nan, "direction": 0},
            )

        supertrend, direction = self._calculate_supertrend(data)

        dir_curr = direction.iloc[-1]
        dir_prev = direction.iloc[-2]
        st_curr = supertrend.iloc[-1]

        signal = None
        strength = 0.5

        # Direction change from down to up
        if dir_prev == -1 and dir_curr == 1:
            signal = "buy"
            strength = 0.8

        # Direction change from up to down
        elif dir_prev == 1 and dir_curr == -1:
            signal = "sell"
            strength = 0.8

        return IndicatorResult(
            name=self.name,
            value={"supertrend": st_curr, "direction": dir_curr},
            signal=signal,
            strength=strength,
        )

    def _calculate_supertrend(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Calculate SuperTrend and direction."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Calculate ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1 / self.period, adjust=False).mean()

        # Calculate basic bands
        hl2 = (high + low) / 2
        upper_band = hl2 + (self.multiplier * atr)
        lower_band = hl2 - (self.multiplier * atr)

        # Initialize SuperTrend
        supertrend = pd.Series(index=data.index, dtype=float)
        direction = pd.Series(index=data.index, dtype=int)

        supertrend.iloc[self.period] = upper_band.iloc[self.period]
        direction.iloc[self.period] = -1

        for i in range(self.period + 1, len(data)):
            if close.iloc[i - 1] <= supertrend.iloc[i - 1]:
                supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i - 1])
            else:
                supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i - 1])

            if close.iloc[i] > supertrend.iloc[i]:
                direction.iloc[i] = 1
            else:
                direction.iloc[i] = -1

        return supertrend, direction
