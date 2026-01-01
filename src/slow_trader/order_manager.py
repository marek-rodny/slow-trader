"""Order management for the trading bot."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from slow_trader.exchanges.base import (
    Exchange,
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
    Position,
)
from slow_trader.strategies.base import TradeSignal, Signal
from slow_trader.risk import RiskManager, RiskLimits
from slow_trader.utils.logger import get_logger, TradeLogger

logger = get_logger("slow_trader.orders")


@dataclass
class ManagedPosition:
    """A position with associated orders and management."""
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    entry_order_id: str
    stop_loss_order_id: str | None = None
    take_profit_order_id: str | None = None
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    created_at: datetime = field(default_factory=datetime.now)


class OrderManager:
    """
    Manages order execution and position tracking.

    Handles order placement, position management, and stop/take-profit orders.
    """

    def __init__(
        self,
        exchange: Exchange,
        risk_manager: RiskManager | None = None,
        dry_run: bool = True,
    ):
        """
        Initialize order manager.

        Args:
            exchange: Exchange connector
            risk_manager: Risk manager instance
            dry_run: If True, simulate orders without execution
        """
        self.exchange = exchange
        self.risk_manager = risk_manager or RiskManager()
        self.dry_run = dry_run

        # Track managed positions
        self.managed_positions: dict[str, ManagedPosition] = {}

        # Trade logger
        self.trade_logger = TradeLogger()

    def execute_signal(
        self,
        signal: TradeSignal,
        portfolio_value: float,
    ) -> Order | None:
        """
        Execute a trading signal.

        Args:
            signal: Trade signal to execute
            portfolio_value: Current portfolio value

        Returns:
            Executed order or None
        """
        if not signal.is_actionable():
            logger.debug(f"Signal not actionable: {signal.signal.value}")
            return None

        # Check risk limits
        current_positions = len(self.exchange.get_positions())
        can_trade, reason = self.risk_manager.can_trade(portfolio_value, current_positions)

        if not can_trade:
            logger.warning(f"Trade blocked by risk manager: {reason}")
            return None

        # Get current price
        ticker = self.exchange.get_ticker(signal.symbol)
        current_price = ticker.get("last", signal.price)

        # Calculate position size
        quantity = self.risk_manager.calculate_position_size(
            signal, portfolio_value, current_price
        )

        if quantity <= 0:
            logger.warning(f"Calculated position size is zero or negative")
            return None

        # Determine order side
        if signal.signal == Signal.BUY:
            side = OrderSide.BUY
        elif signal.signal == Signal.SELL:
            side = OrderSide.SELL
        elif signal.signal == Signal.CLOSE_LONG:
            return self.close_position(signal.symbol, OrderSide.BUY)
        elif signal.signal == Signal.CLOSE_SHORT:
            return self.close_position(signal.symbol, OrderSide.SELL)
        else:
            return None

        # Log the signal
        self.trade_logger.log_signal(
            signal.symbol,
            signal.signal.value,
            signal.strategy,
            signal.indicators,
        )

        # Place the order
        if self.dry_run:
            logger.info(f"[DRY RUN] Would place order: {side.value} {quantity} {signal.symbol}")
            return self._simulate_order(signal.symbol, side, quantity, current_price)

        return self._place_order_with_stops(
            symbol=signal.symbol,
            side=side,
            quantity=quantity,
            current_price=current_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )

    def _place_order_with_stops(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        current_price: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> Order | None:
        """
        Place an order with stop loss and take profit.

        Args:
            symbol: Trading pair
            side: Order side
            quantity: Order quantity
            current_price: Current price
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Main order or None
        """
        try:
            # Place main order
            order = self.exchange.place_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity,
            )

            if order.status == OrderStatus.REJECTED:
                logger.error(f"Order rejected: {order.extra.get('reason', 'Unknown')}")
                return order

            # Log the order
            self.trade_logger.log_order(
                symbol=symbol,
                side=side.value,
                quantity=quantity,
                price=order.filled_price or current_price,
                order_type="market",
                order_id=order.id,
            )

            # Record trade for risk management
            self.risk_manager.record_trade(
                symbol=symbol,
                side=side.value,
                quantity=quantity,
                entry_price=order.filled_price or current_price,
            )

            # Calculate stop/take-profit if not provided
            entry_price = order.filled_price or current_price

            if stop_loss is None:
                stop_loss = self.risk_manager.calculate_stop_loss(entry_price, side)

            if take_profit is None:
                take_profit = self.risk_manager.calculate_take_profit(entry_price, side)

            # Track the managed position
            managed = ManagedPosition(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                entry_order_id=order.id,
                stop_loss_price=stop_loss,
                take_profit_price=take_profit,
            )

            # Place stop loss order
            if stop_loss:
                try:
                    stop_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
                    stop_order = self.exchange.place_order(
                        symbol=symbol,
                        side=stop_side,
                        order_type=OrderType.STOP_LOSS,
                        quantity=quantity,
                        stop_price=stop_loss,
                    )
                    managed.stop_loss_order_id = stop_order.id
                    logger.info(f"Stop loss placed at {stop_loss}")
                except Exception as e:
                    logger.warning(f"Failed to place stop loss: {e}")

            # Place take profit order
            if take_profit:
                try:
                    tp_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
                    tp_order = self.exchange.place_order(
                        symbol=symbol,
                        side=tp_side,
                        order_type=OrderType.LIMIT,
                        quantity=quantity,
                        price=take_profit,
                    )
                    managed.take_profit_order_id = tp_order.id
                    logger.info(f"Take profit placed at {take_profit}")
                except Exception as e:
                    logger.warning(f"Failed to place take profit: {e}")

            self.managed_positions[symbol] = managed
            return order

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            self.trade_logger.log_error(f"Order placement failed for {symbol}", e)
            return None

    def _simulate_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
    ) -> Order:
        """Simulate an order for dry run mode."""
        import uuid

        order = Order(
            id=f"sim_{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=price,
            status=OrderStatus.FILLED,
            filled_quantity=quantity,
            filled_price=price,
        )

        # Track in managed positions
        self.managed_positions[symbol] = ManagedPosition(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=price,
            entry_order_id=order.id,
            stop_loss_price=self.risk_manager.calculate_stop_loss(price, side),
            take_profit_price=self.risk_manager.calculate_take_profit(price, side),
        )

        # Record trade
        self.risk_manager.record_trade(
            symbol=symbol,
            side=side.value,
            quantity=quantity,
            entry_price=price,
        )

        return order

    def close_position(
        self,
        symbol: str,
        side: OrderSide | None = None,
    ) -> Order | None:
        """
        Close a position.

        Args:
            symbol: Trading pair
            side: Expected side of position to close

        Returns:
            Close order or None
        """
        # Get current position
        positions = self.exchange.get_positions(symbol)
        if not positions:
            logger.warning(f"No position found for {symbol}")
            return None

        position = positions[0]

        if side and position.side != side:
            logger.warning(f"Position side mismatch: expected {side}, got {position.side}")
            return None

        # Close position
        close_side = OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY

        if self.dry_run:
            ticker = self.exchange.get_ticker(symbol)
            current_price = ticker.get("last", 0)
            logger.info(
                f"[DRY RUN] Would close position: {close_side.value} "
                f"{position.quantity} {symbol} @ {current_price}"
            )

            # Calculate simulated PnL
            if position.side == OrderSide.BUY:
                pnl = (current_price - position.entry_price) * position.quantity
            else:
                pnl = (position.entry_price - current_price) * position.quantity

            logger.info(f"[DRY RUN] Simulated PnL: ${pnl:.2f}")
            return self._simulate_order(symbol, close_side, position.quantity, current_price)

        try:
            order = self.exchange.place_order(
                symbol=symbol,
                side=close_side,
                order_type=OrderType.MARKET,
                quantity=position.quantity,
            )

            if order.status == OrderStatus.FILLED:
                self.trade_logger.log_fill(
                    symbol=symbol,
                    side=close_side.value,
                    quantity=position.quantity,
                    price=order.filled_price,
                    order_id=order.id,
                )

                # Record trade exit
                self.risk_manager.record_trade(
                    symbol=symbol,
                    side=position.side.value,
                    quantity=position.quantity,
                    entry_price=position.entry_price,
                    exit_price=order.filled_price,
                )

                # Cancel any pending stop/take-profit orders
                if symbol in self.managed_positions:
                    managed = self.managed_positions[symbol]
                    self._cancel_pending_orders(managed)
                    del self.managed_positions[symbol]

            return order

        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return None

    def _cancel_pending_orders(self, managed: ManagedPosition) -> None:
        """Cancel pending stop loss and take profit orders."""
        if managed.stop_loss_order_id:
            try:
                self.exchange.cancel_order(managed.stop_loss_order_id, managed.symbol)
                logger.info(f"Cancelled stop loss order {managed.stop_loss_order_id}")
            except Exception as e:
                logger.warning(f"Failed to cancel stop loss: {e}")

        if managed.take_profit_order_id:
            try:
                self.exchange.cancel_order(managed.take_profit_order_id, managed.symbol)
                logger.info(f"Cancelled take profit order {managed.take_profit_order_id}")
            except Exception as e:
                logger.warning(f"Failed to cancel take profit: {e}")

    def check_positions(self) -> None:
        """
        Check and update managed positions.

        Called periodically to sync positions with exchange.
        """
        for symbol, managed in list(self.managed_positions.items()):
            try:
                positions = self.exchange.get_positions(symbol)

                if not positions:
                    # Position was closed (stop or take profit hit)
                    logger.info(f"Position {symbol} was closed")
                    del self.managed_positions[symbol]
                    continue

                position = positions[0]

                # Check if stop loss or take profit was hit
                if managed.stop_loss_order_id:
                    order = self.exchange.get_order(managed.stop_loss_order_id, symbol)
                    if order and order.status == OrderStatus.FILLED:
                        logger.info(f"Stop loss hit for {symbol} at {order.filled_price}")
                        self.risk_manager.record_trade(
                            symbol=symbol,
                            side=managed.side.value,
                            quantity=managed.quantity,
                            entry_price=managed.entry_price,
                            exit_price=order.filled_price,
                        )

                if managed.take_profit_order_id:
                    order = self.exchange.get_order(managed.take_profit_order_id, symbol)
                    if order and order.status == OrderStatus.FILLED:
                        logger.info(f"Take profit hit for {symbol} at {order.filled_price}")
                        self.risk_manager.record_trade(
                            symbol=symbol,
                            side=managed.side.value,
                            quantity=managed.quantity,
                            entry_price=managed.entry_price,
                            exit_price=order.filled_price,
                        )

            except Exception as e:
                logger.error(f"Error checking position {symbol}: {e}")

    def get_positions_summary(self) -> list[dict[str, Any]]:
        """Get summary of all managed positions."""
        summary = []

        for symbol, managed in self.managed_positions.items():
            try:
                ticker = self.exchange.get_ticker(symbol)
                current_price = ticker.get("last", managed.entry_price)

                if managed.side == OrderSide.BUY:
                    pnl = (current_price - managed.entry_price) * managed.quantity
                    pnl_pct = ((current_price - managed.entry_price) / managed.entry_price) * 100
                else:
                    pnl = (managed.entry_price - current_price) * managed.quantity
                    pnl_pct = ((managed.entry_price - current_price) / managed.entry_price) * 100

                summary.append({
                    "symbol": symbol,
                    "side": managed.side.value,
                    "quantity": managed.quantity,
                    "entry_price": managed.entry_price,
                    "current_price": current_price,
                    "stop_loss": managed.stop_loss_price,
                    "take_profit": managed.take_profit_price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "created_at": managed.created_at.isoformat(),
                })

            except Exception as e:
                logger.error(f"Error getting position summary for {symbol}: {e}")

        return summary
