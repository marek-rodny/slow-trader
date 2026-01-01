"""MACD-based trading strategies."""

import pandas as pd
from slow_trader.strategies.base import Strategy, TradeSignal, Signal
from slow_trader.indicators.momentum import MACD


class MACDStrategy(Strategy):
    """
    MACD Signal Line Crossover Strategy.

    Generates buy signals when MACD crosses above signal line
    and sell signals when MACD crosses below signal line.
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        params: dict | None = None,
    ):
        """
        Initialize MACD strategy.

        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            params: Additional parameters
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

        super().__init__(
            name="macd",
            params={
                "fast_period": fast_period,
                "slow_period": slow_period,
                "signal_period": signal_period,
                **(params or {}),
            },
        )

        self.macd = MACD(fast_period, slow_period, signal_period)

    def _calculate_min_periods(self) -> int:
        """Calculate minimum periods needed."""
        return (
            self.params.get("slow_period", 26) +
            self.params.get("signal_period", 9) + 2
        )

    def analyze(self, data: pd.DataFrame, symbol: str) -> TradeSignal:
        """Analyze data and generate signal."""
        if not self.validate_data(data):
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=symbol,
                strategy=self.name,
                reason="Insufficient data",
            )

        # Get MACD signal
        result = self.macd.get_signal(data)
        current_price = data["close"].iloc[-1]

        if isinstance(result.value, dict):
            indicators = {
                "macd": result.value.get("macd"),
                "signal": result.value.get("signal"),
                "histogram": result.value.get("histogram"),
                "price": current_price,
            }
        else:
            indicators = {"macd": result.value, "price": current_price}

        if result.signal == "buy":
            return TradeSignal(
                signal=Signal.BUY,
                symbol=symbol,
                strategy=self.name,
                strength=result.strength,
                price=current_price,
                reason="MACD crossed above signal line",
                indicators=indicators,
            )
        elif result.signal == "sell":
            return TradeSignal(
                signal=Signal.SELL,
                symbol=symbol,
                strategy=self.name,
                strength=result.strength,
                price=current_price,
                reason="MACD crossed below signal line",
                indicators=indicators,
            )
        else:
            # Provide trend context
            if isinstance(result.value, dict):
                macd_val = result.value.get("macd", 0)
                if macd_val > 0:
                    trend = "bullish"
                elif macd_val < 0:
                    trend = "bearish"
                else:
                    trend = "neutral"
            else:
                trend = "unknown"

            return TradeSignal(
                signal=Signal.HOLD,
                symbol=symbol,
                strategy=self.name,
                reason=f"No MACD crossover, trend is {trend}",
                indicators=indicators,
            )


class MACDHistogramStrategy(Strategy):
    """
    MACD Histogram Strategy.

    Uses the MACD histogram to detect momentum changes.
    Buy when histogram turns positive, sell when it turns negative.
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        params: dict | None = None,
    ):
        """
        Initialize MACD Histogram strategy.

        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            params: Additional parameters
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

        super().__init__(
            name="macd_histogram",
            params={
                "fast_period": fast_period,
                "slow_period": slow_period,
                "signal_period": signal_period,
                **(params or {}),
            },
        )

        self.macd = MACD(fast_period, slow_period, signal_period)

    def _calculate_min_periods(self) -> int:
        """Calculate minimum periods needed."""
        return (
            self.params.get("slow_period", 26) +
            self.params.get("signal_period", 9) + 3
        )

    def analyze(self, data: pd.DataFrame, symbol: str) -> TradeSignal:
        """Analyze data and generate signal."""
        if not self.validate_data(data):
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=symbol,
                strategy=self.name,
                reason="Insufficient data",
            )

        # Get MACD series
        macd_data = self.macd.get_series(data)
        histogram = macd_data["histogram"]

        current_price = data["close"].iloc[-1]
        hist_curr = histogram.iloc[-1]
        hist_prev = histogram.iloc[-2]

        indicators = {
            "macd": macd_data["macd"].iloc[-1],
            "signal": macd_data["signal"].iloc[-1],
            "histogram": hist_curr,
            "prev_histogram": hist_prev,
            "price": current_price,
        }

        # Histogram turns positive (crosses zero from below)
        if hist_prev <= 0 and hist_curr > 0:
            strength = min(abs(hist_curr) / abs(hist_prev) if hist_prev != 0 else 0.5, 1.0)
            return TradeSignal(
                signal=Signal.BUY,
                symbol=symbol,
                strategy=self.name,
                strength=strength,
                price=current_price,
                reason="MACD histogram turned positive",
                indicators=indicators,
            )

        # Histogram turns negative (crosses zero from above)
        elif hist_prev >= 0 and hist_curr < 0:
            strength = min(abs(hist_curr) / abs(hist_prev) if hist_prev != 0 else 0.5, 1.0)
            return TradeSignal(
                signal=Signal.SELL,
                symbol=symbol,
                strategy=self.name,
                strength=strength,
                price=current_price,
                reason="MACD histogram turned negative",
                indicators=indicators,
            )

        # Histogram increasing (momentum building)
        elif hist_curr > hist_prev and hist_curr > 0:
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=symbol,
                strategy=self.name,
                reason="Bullish momentum building",
                indicators=indicators,
            )

        # Histogram decreasing (momentum fading)
        elif hist_curr < hist_prev and hist_curr < 0:
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=symbol,
                strategy=self.name,
                reason="Bearish momentum building",
                indicators=indicators,
            )

        else:
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=symbol,
                strategy=self.name,
                reason="No significant histogram change",
                indicators=indicators,
            )
