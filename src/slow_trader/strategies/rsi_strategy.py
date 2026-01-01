"""RSI-based trading strategies."""

import pandas as pd
from slow_trader.strategies.base import Strategy, TradeSignal, Signal
from slow_trader.indicators.momentum import RSI


class RSIStrategy(Strategy):
    """
    RSI Overbought/Oversold Strategy.

    Generates buy signals when RSI is oversold (below threshold)
    and sell signals when RSI is overbought (above threshold).
    """

    def __init__(
        self,
        period: int = 14,
        overbought: float = 70,
        oversold: float = 30,
        params: dict | None = None,
    ):
        """
        Initialize RSI strategy.

        Args:
            period: RSI calculation period
            overbought: Overbought threshold (e.g., 70)
            oversold: Oversold threshold (e.g., 30)
            params: Additional parameters
        """
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

        super().__init__(
            name="rsi",
            params={
                "period": period,
                "overbought": overbought,
                "oversold": oversold,
                **(params or {}),
            },
        )

        self.rsi = RSI(period, overbought, oversold)

    def _calculate_min_periods(self) -> int:
        """Calculate minimum periods needed."""
        return self.params.get("period", 14) + 2

    def analyze(self, data: pd.DataFrame, symbol: str) -> TradeSignal:
        """Analyze data and generate signal."""
        if not self.validate_data(data):
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=symbol,
                strategy=self.name,
                reason="Insufficient data",
            )

        # Get RSI signal
        result = self.rsi.get_signal(data)
        current_price = data["close"].iloc[-1]

        indicators = {
            "rsi": result.value,
            "overbought": self.overbought,
            "oversold": self.oversold,
            "price": current_price,
        }

        if result.signal == "buy":
            return TradeSignal(
                signal=Signal.BUY,
                symbol=symbol,
                strategy=self.name,
                strength=result.strength,
                price=current_price,
                reason=f"RSI oversold at {result.value:.1f}",
                indicators=indicators,
            )
        elif result.signal == "sell":
            return TradeSignal(
                signal=Signal.SELL,
                symbol=symbol,
                strategy=self.name,
                strength=result.strength,
                price=current_price,
                reason=f"RSI overbought at {result.value:.1f}",
                indicators=indicators,
            )
        else:
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=symbol,
                strategy=self.name,
                reason=f"RSI neutral at {result.value:.1f}",
                indicators=indicators,
            )


class RSIDivergenceStrategy(Strategy):
    """
    RSI Divergence Strategy.

    Detects bullish divergence (price makes lower low, RSI makes higher low)
    and bearish divergence (price makes higher high, RSI makes lower high).
    """

    def __init__(
        self,
        period: int = 14,
        lookback: int = 10,
        params: dict | None = None,
    ):
        """
        Initialize RSI Divergence strategy.

        Args:
            period: RSI calculation period
            lookback: Number of periods to look back for divergence
            params: Additional parameters
        """
        self.period = period
        self.lookback = lookback

        super().__init__(
            name="rsi_divergence",
            params={
                "period": period,
                "lookback": lookback,
                **(params or {}),
            },
        )

        self.rsi = RSI(period)

    def _calculate_min_periods(self) -> int:
        """Calculate minimum periods needed."""
        return self.params.get("period", 14) + self.params.get("lookback", 10) + 2

    def analyze(self, data: pd.DataFrame, symbol: str) -> TradeSignal:
        """Analyze data and generate signal."""
        if not self.validate_data(data):
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=symbol,
                strategy=self.name,
                reason="Insufficient data",
            )

        # Calculate RSI series
        rsi_series = self.rsi.get_series(data)

        # Get recent data
        recent_close = data["close"].iloc[-self.lookback:]
        recent_rsi = rsi_series.iloc[-self.lookback:]

        current_price = data["close"].iloc[-1]
        current_rsi = rsi_series.iloc[-1]

        indicators = {
            "rsi": current_rsi,
            "price": current_price,
            "lookback": self.lookback,
        }

        # Find local extremes
        price_min_idx = recent_close.idxmin()
        price_max_idx = recent_close.idxmax()
        rsi_min_idx = recent_rsi.idxmin()
        rsi_max_idx = recent_rsi.idxmax()

        # Check for bullish divergence
        # Price makes lower low, but RSI makes higher low
        if (recent_close.iloc[-1] <= recent_close.min() * 1.02 and
            recent_rsi.iloc[-1] > recent_rsi.min() * 1.05):
            return TradeSignal(
                signal=Signal.BUY,
                symbol=symbol,
                strategy=self.name,
                strength=0.7,
                price=current_price,
                reason="Bullish RSI divergence detected",
                indicators=indicators,
            )

        # Check for bearish divergence
        # Price makes higher high, but RSI makes lower high
        if (recent_close.iloc[-1] >= recent_close.max() * 0.98 and
            recent_rsi.iloc[-1] < recent_rsi.max() * 0.95):
            return TradeSignal(
                signal=Signal.SELL,
                symbol=symbol,
                strategy=self.name,
                strength=0.7,
                price=current_price,
                reason="Bearish RSI divergence detected",
                indicators=indicators,
            )

        return TradeSignal(
            signal=Signal.HOLD,
            symbol=symbol,
            strategy=self.name,
            reason="No RSI divergence detected",
            indicators=indicators,
        )
