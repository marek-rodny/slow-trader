"""Moving Average Crossover Strategy."""

import pandas as pd
from slow_trader.strategies.base import Strategy, TradeSignal, Signal
from slow_trader.indicators.moving_averages import SMA, EMA, MACrossover


class MACrossoverStrategy(Strategy):
    """
    Moving Average Crossover Strategy.

    Generates buy signals when fast MA crosses above slow MA (golden cross)
    and sell signals when fast MA crosses below slow MA (death cross).
    """

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 20,
        ma_type: str = "ema",
        params: dict | None = None,
    ):
        """
        Initialize MA Crossover strategy.

        Args:
            fast_period: Period for fast moving average
            slow_period: Period for slow moving average
            ma_type: Type of MA ('sma' or 'ema')
            params: Additional parameters
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.ma_type = ma_type

        super().__init__(
            name="ma_crossover",
            params={
                "fast_period": fast_period,
                "slow_period": slow_period,
                "ma_type": ma_type,
                **(params or {}),
            },
        )

        # Initialize crossover detector
        self.crossover = MACrossover(fast_period, slow_period, ma_type)

    def _calculate_min_periods(self) -> int:
        """Calculate minimum periods needed."""
        return max(
            self.params.get("fast_period", 10),
            self.params.get("slow_period", 20),
        ) + 2

    def analyze(self, data: pd.DataFrame, symbol: str) -> TradeSignal:
        """Analyze data and generate signal."""
        if not self.validate_data(data):
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=symbol,
                strategy=self.name,
                reason="Insufficient data",
            )

        # Detect crossover
        result = self.crossover.detect_crossover(data)

        current_price = data["close"].iloc[-1]

        # Build indicators dict
        indicators = {
            "fast_ma": result.value.get("fast") if isinstance(result.value, dict) else None,
            "slow_ma": result.value.get("slow") if isinstance(result.value, dict) else None,
            "price": current_price,
        }

        if result.signal == "buy":
            return TradeSignal(
                signal=Signal.BUY,
                symbol=symbol,
                strategy=self.name,
                strength=result.strength,
                price=current_price,
                reason="Golden cross: fast MA crossed above slow MA",
                indicators=indicators,
            )
        elif result.signal == "sell":
            return TradeSignal(
                signal=Signal.SELL,
                symbol=symbol,
                strategy=self.name,
                strength=result.strength,
                price=current_price,
                reason="Death cross: fast MA crossed below slow MA",
                indicators=indicators,
            )
        else:
            # No crossover, but check trend
            fast = indicators["fast_ma"]
            slow = indicators["slow_ma"]

            if fast and slow:
                if fast > slow:
                    trend = "bullish"
                else:
                    trend = "bearish"
            else:
                trend = "neutral"

            return TradeSignal(
                signal=Signal.HOLD,
                symbol=symbol,
                strategy=self.name,
                strength=0.0,
                price=current_price,
                reason=f"No crossover, trend is {trend}",
                indicators=indicators,
            )


class TripleMAStrategy(Strategy):
    """
    Triple Moving Average Strategy.

    Uses three MAs (short, medium, long) for more refined signals.
    Buy when short > medium > long, sell when short < medium < long.
    """

    def __init__(
        self,
        short_period: int = 5,
        medium_period: int = 10,
        long_period: int = 20,
        ma_type: str = "ema",
        params: dict | None = None,
    ):
        """
        Initialize Triple MA strategy.

        Args:
            short_period: Short MA period
            medium_period: Medium MA period
            long_period: Long MA period
            ma_type: Type of MA ('sma' or 'ema')
            params: Additional parameters
        """
        self.short_period = short_period
        self.medium_period = medium_period
        self.long_period = long_period
        self.ma_type = ma_type

        super().__init__(
            name="triple_ma",
            params={
                "short_period": short_period,
                "medium_period": medium_period,
                "long_period": long_period,
                "ma_type": ma_type,
                **(params or {}),
            },
        )

        # Initialize MAs
        if ma_type.lower() == "ema":
            self.short_ma = EMA(short_period)
            self.medium_ma = EMA(medium_period)
            self.long_ma = EMA(long_period)
        else:
            self.short_ma = SMA(short_period)
            self.medium_ma = SMA(medium_period)
            self.long_ma = SMA(long_period)

    def _calculate_min_periods(self) -> int:
        """Calculate minimum periods needed."""
        return self.params.get("long_period", 20) + 2

    def analyze(self, data: pd.DataFrame, symbol: str) -> TradeSignal:
        """Analyze data and generate signal."""
        if not self.validate_data(data):
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=symbol,
                strategy=self.name,
                reason="Insufficient data",
            )

        # Calculate MAs
        short = self.short_ma.get_series(data).iloc[-1]
        medium = self.medium_ma.get_series(data).iloc[-1]
        long_val = self.long_ma.get_series(data).iloc[-1]

        current_price = data["close"].iloc[-1]

        indicators = {
            "short_ma": short,
            "medium_ma": medium,
            "long_ma": long_val,
            "price": current_price,
        }

        # Check for aligned uptrend or downtrend
        if short > medium > long_val and current_price > short:
            # Strong bullish alignment
            strength = min((short - long_val) / long_val * 10, 1.0)
            return TradeSignal(
                signal=Signal.BUY,
                symbol=symbol,
                strategy=self.name,
                strength=strength,
                price=current_price,
                reason="Bullish alignment: price > short > medium > long",
                indicators=indicators,
            )

        elif short < medium < long_val and current_price < short:
            # Strong bearish alignment
            strength = min((long_val - short) / long_val * 10, 1.0)
            return TradeSignal(
                signal=Signal.SELL,
                symbol=symbol,
                strategy=self.name,
                strength=strength,
                price=current_price,
                reason="Bearish alignment: price < short < medium < long",
                indicators=indicators,
            )

        else:
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=symbol,
                strategy=self.name,
                reason="MAs not aligned",
                indicators=indicators,
            )
