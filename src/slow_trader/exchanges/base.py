"""Base class for exchange connectors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import pandas as pd


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"


class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Represents a trading order."""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None = None
    stop_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    extra: dict = field(default_factory=dict)


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Balance:
    """Account balance for a currency."""
    currency: str
    total: float
    available: float
    locked: float = 0.0


@dataclass
class OHLCV:
    """OHLCV candlestick data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class Exchange(ABC):
    """Abstract base class for all exchange connectors."""

    def __init__(self, name: str, testnet: bool = True):
        """
        Initialize exchange connector.

        Args:
            name: Exchange name
            testnet: Whether to use testnet/sandbox mode
        """
        self.name = name
        self.testnet = testnet

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the exchange.

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the exchange."""
        pass

    @abstractmethod
    def get_balance(self, currency: str | None = None) -> list[Balance] | Balance:
        """
        Get account balance.

        Args:
            currency: Specific currency to get balance for (optional)

        Returns:
            Balance or list of balances
        """
        pass

    @abstractmethod
    def get_ticker(self, symbol: str) -> dict[str, float]:
        """
        Get current ticker data for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")

        Returns:
            Dictionary with bid, ask, last price, etc.
        """
        pass

    @abstractmethod
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Get OHLCV candlestick data.

        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        pass

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: float | None = None,
        stop_price: float | None = None,
    ) -> Order:
        """
        Place a new order.

        Args:
            symbol: Trading pair symbol
            side: Buy or sell
            order_type: Type of order
            quantity: Order quantity
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)

        Returns:
            Created order
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Order ID to cancel
            symbol: Trading pair symbol

        Returns:
            True if cancellation successful
        """
        pass

    @abstractmethod
    def get_order(self, order_id: str, symbol: str) -> Order | None:
        """
        Get order details.

        Args:
            order_id: Order ID
            symbol: Trading pair symbol

        Returns:
            Order or None if not found
        """
        pass

    @abstractmethod
    def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """
        Get all open orders.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of open orders
        """
        pass

    @abstractmethod
    def get_positions(self, symbol: str | None = None) -> list[Position]:
        """
        Get open positions.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of positions
        """
        pass

    def get_portfolio_value(self) -> float:
        """
        Get total portfolio value.

        Returns:
            Total portfolio value in quote currency
        """
        balances = self.get_balance()
        if isinstance(balances, Balance):
            return balances.total
        return sum(b.total for b in balances)

    def __repr__(self) -> str:
        mode = "testnet" if self.testnet else "live"
        return f"{self.__class__.__name__}(name='{self.name}', mode='{mode}')"
