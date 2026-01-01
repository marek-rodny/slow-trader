"""Combined multi-indicator strategies."""

import pandas as pd
from slow_trader.strategies.base import Strategy, TradeSignal, Signal
from slow_trader.indicators.moving_averages import EMA
from slow_trader.indicators.momentum import RSI, MACD
from slow_trader.indicators.volatility import BollingerBands, ATR
from slow_trader.indicators.trend import ADX, TrendSignal


class CombinedStrategy(Strategy):
    """
    Combined Strategy using multiple indicators.

    Requires confirmation from multiple indicators before generating signals.
    This reduces false signals but may miss some opportunities.
    """

    def __init__(
        self,
        ema_period: int = 20,
        rsi_period: int = 14,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        min_confirmations: int = 2,
        params: dict | None = None,
    ):
        """
        Initialize Combined strategy.

        Args:
            ema_period: EMA period for trend
            rsi_period: RSI period
            rsi_overbought: RSI overbought level
            rsi_oversold: RSI oversold level
            macd_fast: MACD fast period
            macd_slow: MACD slow period
            macd_signal: MACD signal period
            min_confirmations: Minimum indicators that must agree
            params: Additional parameters
        """
        self.ema_period = ema_period
        self.rsi_period = rsi_period
        self.min_confirmations = min_confirmations

        super().__init__(
            name="combined",
            params={
                "ema_period": ema_period,
                "rsi_period": rsi_period,
                "rsi_overbought": rsi_overbought,
                "rsi_oversold": rsi_oversold,
                "macd_fast": macd_fast,
                "macd_slow": macd_slow,
                "macd_signal": macd_signal,
                "min_confirmations": min_confirmations,
                **(params or {}),
            },
        )

        # Initialize indicators
        self.ema = EMA(ema_period)
        self.rsi = RSI(rsi_period, rsi_overbought, rsi_oversold)
        self.macd = MACD(macd_fast, macd_slow, macd_signal)

    def _calculate_min_periods(self) -> int:
        """Calculate minimum periods needed."""
        return max(
            self.params.get("ema_period", 20),
            self.params.get("rsi_period", 14) + 1,
            self.params.get("macd_slow", 26) + self.params.get("macd_signal", 9),
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

        current_price = data["close"].iloc[-1]

        # Get signals from each indicator
        ema_result = self.ema.get_signal(data)
        rsi_result = self.rsi.get_signal(data)
        macd_result = self.macd.get_signal(data)

        # Count buy and sell confirmations
        buy_count = 0
        sell_count = 0
        reasons = []

        # EMA signal
        if ema_result.signal == "buy":
            buy_count += 1
            reasons.append("EMA bullish")
        elif ema_result.signal == "sell":
            sell_count += 1
            reasons.append("EMA bearish")

        # RSI signal
        if rsi_result.signal == "buy":
            buy_count += 1
            reasons.append(f"RSI oversold ({rsi_result.value:.1f})")
        elif rsi_result.signal == "sell":
            sell_count += 1
            reasons.append(f"RSI overbought ({rsi_result.value:.1f})")

        # MACD signal
        if macd_result.signal == "buy":
            buy_count += 1
            reasons.append("MACD bullish crossover")
        elif macd_result.signal == "sell":
            sell_count += 1
            reasons.append("MACD bearish crossover")

        # Collect indicator values
        macd_val = macd_result.value if isinstance(macd_result.value, dict) else {}
        indicators = {
            "ema": ema_result.value,
            "rsi": rsi_result.value,
            "macd": macd_val.get("macd") if macd_val else None,
            "macd_signal": macd_val.get("signal") if macd_val else None,
            "macd_histogram": macd_val.get("histogram") if macd_val else None,
            "price": current_price,
            "buy_confirmations": buy_count,
            "sell_confirmations": sell_count,
        }

        # Generate signal based on confirmations
        if buy_count >= self.min_confirmations:
            strength = buy_count / 3  # Normalize to 0-1
            return TradeSignal(
                signal=Signal.BUY,
                symbol=symbol,
                strategy=self.name,
                strength=strength,
                price=current_price,
                reason=f"Buy confirmed: {', '.join(reasons)}",
                indicators=indicators,
            )

        elif sell_count >= self.min_confirmations:
            strength = sell_count / 3
            return TradeSignal(
                signal=Signal.SELL,
                symbol=symbol,
                strategy=self.name,
                strength=strength,
                price=current_price,
                reason=f"Sell confirmed: {', '.join(reasons)}",
                indicators=indicators,
            )

        else:
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=symbol,
                strategy=self.name,
                reason=f"Insufficient confirmations (buy: {buy_count}, sell: {sell_count})",
                indicators=indicators,
            )


class TrendFollowingStrategy(Strategy):
    """
    Trend Following Strategy.

    Uses multiple trend indicators to identify and follow strong trends.
    Only trades in the direction of the trend with momentum confirmation.
    """

    def __init__(
        self,
        short_ema: int = 10,
        long_ema: int = 50,
        adx_period: int = 14,
        adx_threshold: float = 25,
        atr_period: int = 14,
        params: dict | None = None,
    ):
        """
        Initialize Trend Following strategy.

        Args:
            short_ema: Short EMA period
            long_ema: Long EMA period
            adx_period: ADX period for trend strength
            adx_threshold: Minimum ADX for trend trading
            atr_period: ATR period for volatility
            params: Additional parameters
        """
        self.short_ema_period = short_ema
        self.long_ema_period = long_ema
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold

        super().__init__(
            name="trend_following",
            params={
                "short_ema": short_ema,
                "long_ema": long_ema,
                "adx_period": adx_period,
                "adx_threshold": adx_threshold,
                "atr_period": atr_period,
                **(params or {}),
            },
        )

        # Initialize indicators
        self.short_ema = EMA(short_ema)
        self.long_ema = EMA(long_ema)
        self.adx = ADX(adx_period)
        self.atr = ATR(atr_period)

    def _calculate_min_periods(self) -> int:
        """Calculate minimum periods needed."""
        return max(
            self.params.get("long_ema", 50),
            self.params.get("adx_period", 14) * 2,
            self.params.get("atr_period", 14),
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

        current_price = data["close"].iloc[-1]

        # Get indicator values
        short_ema = self.short_ema.get_series(data).iloc[-1]
        long_ema = self.long_ema.get_series(data).iloc[-1]
        adx_result = self.adx.calculate(data)
        atr_result = self.atr.calculate(data)

        adx_val = adx_result.value if isinstance(adx_result.value, dict) else {}
        adx = adx_val.get("adx", 0) if adx_val else 0
        plus_di = adx_val.get("plus_di", 0) if adx_val else 0
        minus_di = adx_val.get("minus_di", 0) if adx_val else 0

        indicators = {
            "short_ema": short_ema,
            "long_ema": long_ema,
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "atr": atr_result.value,
            "price": current_price,
        }

        # Check for strong trend
        if adx < self.adx_threshold:
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=symbol,
                strategy=self.name,
                reason=f"Weak trend (ADX: {adx:.1f} < {self.adx_threshold})",
                indicators=indicators,
            )

        # Uptrend: short EMA > long EMA and +DI > -DI
        if short_ema > long_ema and plus_di > minus_di:
            # Additional check: price above short EMA
            if current_price > short_ema:
                strength = min(adx / 50, 1.0)
                return TradeSignal(
                    signal=Signal.BUY,
                    symbol=symbol,
                    strategy=self.name,
                    strength=strength,
                    price=current_price,
                    reason=f"Strong uptrend confirmed (ADX: {adx:.1f})",
                    indicators=indicators,
                )

        # Downtrend: short EMA < long EMA and -DI > +DI
        elif short_ema < long_ema and minus_di > plus_di:
            # Additional check: price below short EMA
            if current_price < short_ema:
                strength = min(adx / 50, 1.0)
                return TradeSignal(
                    signal=Signal.SELL,
                    symbol=symbol,
                    strategy=self.name,
                    strength=strength,
                    price=current_price,
                    reason=f"Strong downtrend confirmed (ADX: {adx:.1f})",
                    indicators=indicators,
                )

        return TradeSignal(
            signal=Signal.HOLD,
            symbol=symbol,
            strategy=self.name,
            reason="Trend not confirmed",
            indicators=indicators,
        )


class MeanReversionStrategy(Strategy):
    """
    Mean Reversion Strategy using Bollinger Bands.

    Trades when price deviates significantly from the mean,
    expecting it to revert back.
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_extreme: float = 20,  # Distance from 50 considered extreme
        params: dict | None = None,
    ):
        """
        Initialize Mean Reversion strategy.

        Args:
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands standard deviation multiplier
            rsi_period: RSI period for confirmation
            rsi_extreme: RSI distance from 50 for extreme readings
            params: Additional parameters
        """
        super().__init__(
            name="mean_reversion",
            params={
                "bb_period": bb_period,
                "bb_std": bb_std,
                "rsi_period": rsi_period,
                "rsi_extreme": rsi_extreme,
                **(params or {}),
            },
        )

        self.bb = BollingerBands(bb_period, bb_std)
        self.rsi = RSI(rsi_period)

    def _calculate_min_periods(self) -> int:
        """Calculate minimum periods needed."""
        return max(
            self.params.get("bb_period", 20),
            self.params.get("rsi_period", 14) + 1,
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

        current_price = data["close"].iloc[-1]

        # Get indicator values
        bb_result = self.bb.calculate(data)
        rsi_result = self.rsi.calculate(data)

        bb_val = bb_result.value if isinstance(bb_result.value, dict) else {}
        upper = bb_val.get("upper", 0)
        middle = bb_val.get("middle", 0)
        lower = bb_val.get("lower", 0)
        rsi = rsi_result.value

        indicators = {
            "bb_upper": upper,
            "bb_middle": middle,
            "bb_lower": lower,
            "rsi": rsi,
            "price": current_price,
        }

        rsi_extreme = self.params.get("rsi_extreme", 20)

        # Price at lower band and RSI oversold
        if current_price <= lower and rsi < (50 - rsi_extreme):
            strength = min((lower - current_price) / lower * 10 + 0.3, 1.0)
            return TradeSignal(
                signal=Signal.BUY,
                symbol=symbol,
                strategy=self.name,
                strength=strength,
                price=current_price,
                stop_loss=lower * 0.98,  # 2% below lower band
                take_profit=middle,  # Target the middle band
                reason=f"Mean reversion buy: price at lower BB, RSI {rsi:.1f}",
                indicators=indicators,
            )

        # Price at upper band and RSI overbought
        elif current_price >= upper and rsi > (50 + rsi_extreme):
            strength = min((current_price - upper) / upper * 10 + 0.3, 1.0)
            return TradeSignal(
                signal=Signal.SELL,
                symbol=symbol,
                strategy=self.name,
                strength=strength,
                price=current_price,
                stop_loss=upper * 1.02,  # 2% above upper band
                take_profit=middle,  # Target the middle band
                reason=f"Mean reversion sell: price at upper BB, RSI {rsi:.1f}",
                indicators=indicators,
            )

        return TradeSignal(
            signal=Signal.HOLD,
            symbol=symbol,
            strategy=self.name,
            reason="Price within normal range",
            indicators=indicators,
        )
