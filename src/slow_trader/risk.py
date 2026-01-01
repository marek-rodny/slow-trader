"""Risk management for the trading bot."""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any
from slow_trader.exchanges.base import Order, OrderSide, Position
from slow_trader.strategies.base import TradeSignal, Signal
from slow_trader.utils.logger import get_logger
from slow_trader.utils.helpers import calculate_position_size, round_quantity

logger = get_logger("slow_trader.risk")


@dataclass
class RiskLimits:
    """Risk management limits."""
    max_position_size: float = 0.1  # Max 10% of portfolio per position
    max_daily_loss: float = 0.05  # Max 5% daily loss
    max_drawdown: float = 0.15  # Max 15% drawdown from peak
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.05  # 5% take profit
    max_open_positions: int = 5  # Max concurrent positions
    min_trade_interval_minutes: int = 30  # Min time between trades
    max_trades_per_day: int = 10  # Max trades per day


@dataclass
class TradeRecord:
    """Record of a trade for tracking."""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: float | None = None
    pnl: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class RiskManager:
    """
    Manages trading risk and enforces limits.

    Tracks positions, daily PnL, and enforces risk rules.
    """

    def __init__(self, limits: RiskLimits | None = None):
        """
        Initialize risk manager.

        Args:
            limits: Risk limits configuration
        """
        self.limits = limits or RiskLimits()

        # Tracking state
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.last_trade_time: datetime | None = None
        self.current_date: date = date.today()

        # Historical tracking
        self.peak_portfolio_value: float = 0.0
        self.trade_history: list[TradeRecord] = []

    def reset_daily(self) -> None:
        """Reset daily tracking (call at start of each day)."""
        today = date.today()
        if today != self.current_date:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.current_date = today
            logger.info("Daily risk counters reset")

    def update_portfolio_peak(self, current_value: float) -> None:
        """Update peak portfolio value for drawdown calculation."""
        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value

    def check_drawdown(self, current_value: float) -> bool:
        """
        Check if current drawdown exceeds limit.

        Args:
            current_value: Current portfolio value

        Returns:
            True if within limits, False if drawdown exceeded
        """
        if self.peak_portfolio_value <= 0:
            self.peak_portfolio_value = current_value
            return True

        drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value

        if drawdown > self.limits.max_drawdown:
            logger.warning(
                f"Drawdown limit exceeded: {drawdown:.1%} > {self.limits.max_drawdown:.1%}"
            )
            return False

        return True

    def check_daily_loss(self) -> bool:
        """
        Check if daily loss limit exceeded.

        Returns:
            True if within limits
        """
        # Note: daily_pnl is negative for losses
        if self.daily_pnl < -self.limits.max_daily_loss * self.peak_portfolio_value:
            logger.warning(
                f"Daily loss limit exceeded: ${abs(self.daily_pnl):.2f}"
            )
            return False
        return True

    def check_trade_frequency(self) -> bool:
        """
        Check if enough time has passed since last trade.

        Returns:
            True if allowed to trade
        """
        if self.last_trade_time is None:
            return True

        elapsed = (datetime.now() - self.last_trade_time).total_seconds() / 60
        if elapsed < self.limits.min_trade_interval_minutes:
            logger.debug(
                f"Trade too soon: {elapsed:.1f}min < {self.limits.min_trade_interval_minutes}min"
            )
            return False
        return True

    def check_daily_trade_count(self) -> bool:
        """
        Check if daily trade limit reached.

        Returns:
            True if within limits
        """
        if self.daily_trades >= self.limits.max_trades_per_day:
            logger.warning(
                f"Daily trade limit reached: {self.daily_trades} >= {self.limits.max_trades_per_day}"
            )
            return False
        return True

    def check_position_count(self, current_positions: int) -> bool:
        """
        Check if max positions reached.

        Args:
            current_positions: Number of current open positions

        Returns:
            True if can open new position
        """
        if current_positions >= self.limits.max_open_positions:
            logger.warning(
                f"Max positions reached: {current_positions} >= {self.limits.max_open_positions}"
            )
            return False
        return True

    def can_trade(
        self,
        portfolio_value: float,
        current_positions: int,
    ) -> tuple[bool, str]:
        """
        Check if trading is allowed based on all risk rules.

        Args:
            portfolio_value: Current portfolio value
            current_positions: Number of open positions

        Returns:
            Tuple of (allowed, reason)
        """
        self.reset_daily()
        self.update_portfolio_peak(portfolio_value)

        if not self.check_drawdown(portfolio_value):
            return False, "Drawdown limit exceeded"

        if not self.check_daily_loss():
            return False, "Daily loss limit exceeded"

        if not self.check_trade_frequency():
            return False, "Trade frequency limit"

        if not self.check_daily_trade_count():
            return False, "Daily trade count exceeded"

        if not self.check_position_count(current_positions):
            return False, "Max positions reached"

        return True, "OK"

    def calculate_position_size(
        self,
        signal: TradeSignal,
        portfolio_value: float,
        current_price: float,
    ) -> float:
        """
        Calculate appropriate position size based on risk rules.

        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            current_price: Current asset price

        Returns:
            Position size (quantity)
        """
        # Calculate stop loss price
        if signal.stop_loss:
            stop_loss = signal.stop_loss
        else:
            if signal.signal == Signal.BUY:
                stop_loss = current_price * (1 - self.limits.stop_loss_pct)
            else:
                stop_loss = current_price * (1 + self.limits.stop_loss_pct)

        # Calculate position size based on risk per trade
        risk_per_trade = 0.01  # Risk 1% per trade
        position_size = calculate_position_size(
            portfolio_value=portfolio_value,
            risk_per_trade=risk_per_trade,
            entry_price=current_price,
            stop_loss_price=stop_loss,
            max_position_pct=self.limits.max_position_size,
        )

        # Apply signal strength adjustment
        position_size *= max(signal.strength, 0.5)  # At least 50% of calculated size

        return round_quantity(position_size, 8)

    def calculate_stop_loss(
        self,
        entry_price: float,
        side: OrderSide,
        atr: float | None = None,
    ) -> float:
        """
        Calculate stop loss price.

        Args:
            entry_price: Entry price
            side: Trade side
            atr: Optional ATR value for dynamic stops

        Returns:
            Stop loss price
        """
        if atr:
            # Use 2x ATR for stop loss
            stop_distance = atr * 2
        else:
            stop_distance = entry_price * self.limits.stop_loss_pct

        if side == OrderSide.BUY:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def calculate_take_profit(
        self,
        entry_price: float,
        side: OrderSide,
        risk_reward: float = 2.0,
        atr: float | None = None,
    ) -> float:
        """
        Calculate take profit price.

        Args:
            entry_price: Entry price
            side: Trade side
            risk_reward: Risk/reward ratio
            atr: Optional ATR value

        Returns:
            Take profit price
        """
        if atr:
            stop_distance = atr * 2
            take_profit_distance = stop_distance * risk_reward
        else:
            take_profit_distance = entry_price * self.limits.take_profit_pct

        if side == OrderSide.BUY:
            return entry_price + take_profit_distance
        else:
            return entry_price - take_profit_distance

    def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        exit_price: float | None = None,
    ) -> None:
        """
        Record a trade for tracking.

        Args:
            symbol: Trading pair
            side: Trade side
            quantity: Trade quantity
            entry_price: Entry price
            exit_price: Exit price (if closing)
        """
        pnl = 0.0
        if exit_price:
            if side.lower() == "buy":
                pnl = (exit_price - entry_price) * quantity
            else:
                pnl = (entry_price - exit_price) * quantity

            self.daily_pnl += pnl

        self.trade_history.append(TradeRecord(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
        ))

        self.daily_trades += 1
        self.last_trade_time = datetime.now()

        logger.info(
            f"Trade recorded: {side} {quantity} {symbol} @ {entry_price}"
            + (f" -> {exit_price} (PnL: ${pnl:.2f})" if exit_price else "")
        )

    def get_stats(self) -> dict[str, Any]:
        """Get risk management statistics."""
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for t in self.trade_history if t.pnl > 0)
        total_pnl = sum(t.pnl for t in self.trade_history)

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": total_trades - winning_trades,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "total_pnl": total_pnl,
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "peak_portfolio": self.peak_portfolio_value,
        }
