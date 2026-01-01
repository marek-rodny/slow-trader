"""Demo/paper trading exchange for testing strategies."""

from datetime import datetime, timedelta
from typing import Any
import uuid
import pandas as pd
import numpy as np

from slow_trader.exchanges.base import (
    Exchange,
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
    Position,
    Balance,
)
from slow_trader.utils.logger import get_logger

logger = get_logger("slow_trader.demo_exchange")


class DemoExchange(Exchange):
    """
    Demo exchange for paper trading and backtesting.

    Simulates a real exchange with configurable starting balance
    and generates or uses historical price data.
    """

    def __init__(
        self,
        starting_balance: dict[str, float] | None = None,
        fee_rate: float = 0.001,  # 0.1% trading fee
        slippage: float = 0.0005,  # 0.05% slippage
    ):
        """
        Initialize demo exchange.

        Args:
            starting_balance: Starting balance per currency
            fee_rate: Trading fee rate (e.g., 0.001 = 0.1%)
            slippage: Simulated slippage rate
        """
        super().__init__(name="demo", testnet=True)
        self.fee_rate = fee_rate
        self.slippage = slippage

        # Initialize balances
        self.balances: dict[str, Balance] = {}
        if starting_balance:
            for currency, amount in starting_balance.items():
                self.balances[currency] = Balance(
                    currency=currency,
                    total=amount,
                    available=amount,
                    locked=0.0,
                )
        else:
            # Default starting balance
            self.balances = {
                "USDT": Balance(currency="USDT", total=10000.0, available=10000.0),
                "USD": Balance(currency="USD", total=10000.0, available=10000.0),
            }

        # Order and position tracking
        self.orders: dict[str, Order] = {}
        self.positions: dict[str, Position] = {}

        # Simulated price data
        self.prices: dict[str, float] = {}
        self.price_history: dict[str, pd.DataFrame] = {}

        self.connected = False

    def connect(self) -> bool:
        """Connect to demo exchange."""
        self.connected = True
        logger.info("Connected to demo exchange")
        return True

    def disconnect(self) -> None:
        """Disconnect from demo exchange."""
        self.connected = False
        logger.info("Disconnected from demo exchange")

    def set_price(self, symbol: str, price: float) -> None:
        """
        Set current price for a symbol (for simulation).

        Args:
            symbol: Trading pair symbol
            price: Current price
        """
        self.prices[symbol] = price

    def set_price_history(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Set historical price data for a symbol.

        Args:
            symbol: Trading pair symbol
            data: DataFrame with OHLCV data
        """
        self.price_history[symbol] = data.copy()
        if len(data) > 0:
            self.prices[symbol] = data["close"].iloc[-1]

    def generate_sample_data(
        self,
        symbol: str,
        periods: int = 500,
        timeframe: str = "1h",
        start_price: float = 100.0,
        volatility: float = 0.02,
    ) -> pd.DataFrame:
        """
        Generate sample OHLCV data for testing.

        Args:
            symbol: Trading pair symbol
            periods: Number of periods to generate
            timeframe: Candle timeframe
            start_price: Starting price
            volatility: Price volatility

        Returns:
            DataFrame with generated OHLCV data
        """
        # Parse timeframe to timedelta
        tf_map = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "4h": timedelta(hours=4),
            "1d": timedelta(days=1),
        }
        delta = tf_map.get(timeframe, timedelta(hours=1))

        # Generate timestamps
        end_time = datetime.now()
        timestamps = [end_time - delta * i for i in range(periods)]
        timestamps.reverse()

        # Generate random walk price data
        np.random.seed(42)  # For reproducibility
        returns = np.random.normal(0, volatility, periods)
        prices = start_price * np.exp(np.cumsum(returns))

        # Generate OHLCV data
        data = []
        for i, (ts, close) in enumerate(zip(timestamps, prices)):
            # Create realistic OHLC from close
            open_price = close * (1 + np.random.uniform(-0.005, 0.005))
            high = max(open_price, close) * (1 + np.random.uniform(0, 0.01))
            low = min(open_price, close) * (1 - np.random.uniform(0, 0.01))
            volume = np.random.uniform(1000, 10000)

            data.append({
                "timestamp": ts,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            })

        df = pd.DataFrame(data)
        self.set_price_history(symbol, df)

        return df

    def get_balance(self, currency: str | None = None) -> list[Balance] | Balance:
        """Get account balance."""
        if currency:
            return self.balances.get(
                currency,
                Balance(currency=currency, total=0.0, available=0.0),
            )
        return list(self.balances.values())

    def get_ticker(self, symbol: str) -> dict[str, float]:
        """Get current ticker data."""
        price = self.prices.get(symbol, 100.0)

        # Simulate bid/ask spread
        spread = price * 0.001
        return {
            "symbol": symbol,
            "bid": price - spread / 2,
            "ask": price + spread / 2,
            "last": price,
            "volume": 10000.0,
        }

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
    ) -> pd.DataFrame:
        """Get OHLCV data."""
        if symbol in self.price_history:
            df = self.price_history[symbol]
            return df.tail(limit).copy()

        # Generate sample data if not available
        return self.generate_sample_data(symbol, periods=limit, timeframe=timeframe)

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: float | None = None,
        stop_price: float | None = None,
    ) -> Order:
        """Place a new order."""
        order_id = str(uuid.uuid4())[:8]

        # Get current price
        current_price = self.prices.get(symbol, 100.0)

        # Apply slippage for market orders
        if order_type == OrderType.MARKET:
            if side == OrderSide.BUY:
                fill_price = current_price * (1 + self.slippage)
            else:
                fill_price = current_price * (1 - self.slippage)
        else:
            fill_price = price or current_price

        # Calculate order value and fees
        order_value = quantity * fill_price
        fee = order_value * self.fee_rate

        # Parse symbol for currencies
        base, quote = self._parse_symbol(symbol)

        # Check balance
        if side == OrderSide.BUY:
            required = order_value + fee
            if quote not in self.balances or self.balances[quote].available < required:
                order = Order(
                    id=order_id,
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    status=OrderStatus.REJECTED,
                )
                order.extra["reason"] = "Insufficient balance"
                return order

            # Deduct from quote currency
            self.balances[quote].available -= required
            self.balances[quote].total -= fee

            # Add to base currency
            if base not in self.balances:
                self.balances[base] = Balance(currency=base, total=0.0, available=0.0)
            self.balances[base].total += quantity
            self.balances[base].available += quantity

        else:  # SELL
            if base not in self.balances or self.balances[base].available < quantity:
                order = Order(
                    id=order_id,
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    status=OrderStatus.REJECTED,
                )
                order.extra["reason"] = "Insufficient balance"
                return order

            # Deduct from base currency
            self.balances[base].available -= quantity
            self.balances[base].total -= quantity

            # Add to quote currency (minus fee)
            if quote not in self.balances:
                self.balances[quote] = Balance(currency=quote, total=0.0, available=0.0)
            self.balances[quote].total += order_value - fee
            self.balances[quote].available += order_value - fee

        # Create filled order (market orders fill immediately in simulation)
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=OrderStatus.FILLED if order_type == OrderType.MARKET else OrderStatus.OPEN,
            filled_quantity=quantity if order_type == OrderType.MARKET else 0.0,
            filled_price=fill_price if order_type == OrderType.MARKET else 0.0,
        )
        order.extra["fee"] = fee

        self.orders[order_id] = order

        # Update position
        self._update_position(symbol, side, quantity, fill_price)

        logger.info(
            f"Order placed: {side.value} {quantity} {symbol} @ {fill_price:.2f} "
            f"(fee: {fee:.2f})"
        )

        return order

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order."""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status == OrderStatus.OPEN:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now()
                logger.info(f"Order cancelled: {order_id}")
                return True
        return False

    def get_order(self, order_id: str, symbol: str) -> Order | None:
        """Get order details."""
        return self.orders.get(order_id)

    def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Get all open orders."""
        orders = [o for o in self.orders.values() if o.status == OrderStatus.OPEN]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def get_positions(self, symbol: str | None = None) -> list[Position]:
        """Get open positions."""
        positions = list(self.positions.values())
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]

        # Update current prices and PnL
        for pos in positions:
            current = self.prices.get(pos.symbol, pos.entry_price)
            pos.current_price = current
            if pos.side == OrderSide.BUY:
                pos.unrealized_pnl = (current - pos.entry_price) * pos.quantity
            else:
                pos.unrealized_pnl = (pos.entry_price - current) * pos.quantity

        return positions

    def get_portfolio_value(self) -> float:
        """Get total portfolio value in quote currency."""
        total = 0.0

        # Add cash balances
        for balance in self.balances.values():
            if balance.currency in ("USDT", "USD", "BUSD"):
                total += balance.total
            else:
                # Convert to quote using current price
                symbol = f"{balance.currency}/USDT"
                price = self.prices.get(symbol, 0.0)
                total += balance.total * price

        return total

    def _parse_symbol(self, symbol: str) -> tuple[str, str]:
        """Parse symbol into base and quote currencies."""
        if "/" in symbol:
            parts = symbol.split("/")
            return parts[0], parts[1]
        # Assume last 3-4 chars are quote
        if symbol.endswith("USDT"):
            return symbol[:-4], "USDT"
        if symbol.endswith("USD"):
            return symbol[:-3], "USD"
        return symbol[:3], symbol[3:]

    def _update_position(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
    ) -> None:
        """Update or create position after order fill."""
        if symbol in self.positions:
            pos = self.positions[symbol]

            if pos.side == side:
                # Adding to position
                total_cost = pos.entry_price * pos.quantity + price * quantity
                pos.quantity += quantity
                pos.entry_price = total_cost / pos.quantity
            else:
                # Reducing or closing position
                if quantity >= pos.quantity:
                    # Close position
                    if pos.side == OrderSide.BUY:
                        realized = (price - pos.entry_price) * pos.quantity
                    else:
                        realized = (pos.entry_price - price) * pos.quantity
                    pos.realized_pnl += realized
                    del self.positions[symbol]
                else:
                    # Partial close
                    if pos.side == OrderSide.BUY:
                        realized = (price - pos.entry_price) * quantity
                    else:
                        realized = (pos.entry_price - price) * quantity
                    pos.realized_pnl += realized
                    pos.quantity -= quantity
        else:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=price,
                current_price=price,
            )
