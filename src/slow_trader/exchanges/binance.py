"""Binance exchange connector using ccxt."""

from datetime import datetime
from typing import Any
import pandas as pd

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

logger = get_logger("slow_trader.binance")


class BinanceExchange(Exchange):
    """
    Binance exchange connector.

    Uses the ccxt library for unified API access to Binance.
    Supports both spot and futures trading.
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
        market_type: str = "spot",  # 'spot' or 'futures'
    ):
        """
        Initialize Binance connector.

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use testnet (default True for safety)
            market_type: 'spot' or 'futures'
        """
        super().__init__(name="binance", testnet=testnet)
        self.api_key = api_key
        self.api_secret = api_secret
        self.market_type = market_type
        self.exchange = None

    def connect(self) -> bool:
        """Connect to Binance."""
        try:
            import ccxt

            # Select exchange class based on market type
            if self.market_type == "futures":
                exchange_class = ccxt.binanceusdm
            else:
                exchange_class = ccxt.binance

            # Configure exchange
            config = {
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "sandbox": self.testnet,
                "options": {
                    "defaultType": self.market_type,
                },
            }

            self.exchange = exchange_class(config)

            # Test connection
            self.exchange.load_markets()
            logger.info(f"Connected to Binance {'testnet' if self.testnet else 'mainnet'}")
            return True

        except ImportError:
            logger.error("ccxt library not installed. Run: pip install ccxt")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from Binance."""
        self.exchange = None
        logger.info("Disconnected from Binance")

    def get_balance(self, currency: str | None = None) -> list[Balance] | Balance:
        """Get account balance."""
        if not self.exchange:
            raise RuntimeError("Not connected to exchange")

        try:
            balance_data = self.exchange.fetch_balance()

            balances = []
            for curr, info in balance_data.get("total", {}).items():
                if info > 0:
                    balances.append(Balance(
                        currency=curr,
                        total=info,
                        available=balance_data.get("free", {}).get(curr, 0.0),
                        locked=balance_data.get("used", {}).get(curr, 0.0),
                    ))

            if currency:
                for b in balances:
                    if b.currency == currency:
                        return b
                return Balance(currency=currency, total=0.0, available=0.0)

            return balances

        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            raise

    def get_ticker(self, symbol: str) -> dict[str, float]:
        """Get current ticker data."""
        if not self.exchange:
            raise RuntimeError("Not connected to exchange")

        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                "symbol": symbol,
                "bid": ticker.get("bid", 0.0),
                "ask": ticker.get("ask", 0.0),
                "last": ticker.get("last", 0.0),
                "volume": ticker.get("baseVolume", 0.0),
                "high": ticker.get("high", 0.0),
                "low": ticker.get("low", 0.0),
            }
        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            raise

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
    ) -> pd.DataFrame:
        """Get OHLCV candlestick data."""
        if not self.exchange:
            raise RuntimeError("Not connected to exchange")

        try:
            # Binance timeframe mapping
            tf_map = {
                "1m": "1m",
                "5m": "5m",
                "15m": "15m",
                "1h": "1h",
                "4h": "4h",
                "1d": "1d",
            }
            binance_tf = tf_map.get(timeframe, "1h")

            ohlcv = self.exchange.fetch_ohlcv(symbol, binance_tf, limit=limit)

            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            return df

        except Exception as e:
            logger.error(f"Failed to get OHLCV for {symbol}: {e}")
            raise

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
        if not self.exchange:
            raise RuntimeError("Not connected to exchange")

        try:
            # Map order type
            type_map = {
                OrderType.MARKET: "market",
                OrderType.LIMIT: "limit",
                OrderType.STOP_LOSS: "stop_loss",
                OrderType.STOP_LIMIT: "stop_loss_limit",
                OrderType.TAKE_PROFIT: "take_profit",
            }
            ccxt_type = type_map.get(order_type, "market")

            # Build params
            params = {}
            if stop_price:
                params["stopPrice"] = stop_price

            # Place order
            result = self.exchange.create_order(
                symbol=symbol,
                type=ccxt_type,
                side=side.value,
                amount=quantity,
                price=price,
                params=params,
            )

            # Map status
            status_map = {
                "open": OrderStatus.OPEN,
                "closed": OrderStatus.FILLED,
                "canceled": OrderStatus.CANCELLED,
                "rejected": OrderStatus.REJECTED,
            }

            order = Order(
                id=result.get("id", ""),
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                status=status_map.get(result.get("status", ""), OrderStatus.OPEN),
                filled_quantity=result.get("filled", 0.0),
                filled_price=result.get("average", 0.0) or 0.0,
            )

            logger.info(f"Order placed: {order}")
            return order

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order."""
        if not self.exchange:
            raise RuntimeError("Not connected to exchange")

        try:
            self.exchange.cancel_order(order_id, symbol)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_order(self, order_id: str, symbol: str) -> Order | None:
        """Get order details."""
        if not self.exchange:
            raise RuntimeError("Not connected to exchange")

        try:
            result = self.exchange.fetch_order(order_id, symbol)

            status_map = {
                "open": OrderStatus.OPEN,
                "closed": OrderStatus.FILLED,
                "canceled": OrderStatus.CANCELLED,
            }

            side = OrderSide.BUY if result.get("side") == "buy" else OrderSide.SELL

            return Order(
                id=result.get("id", ""),
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,  # ccxt doesn't always return type
                quantity=result.get("amount", 0.0),
                price=result.get("price"),
                status=status_map.get(result.get("status", ""), OrderStatus.OPEN),
                filled_quantity=result.get("filled", 0.0),
                filled_price=result.get("average", 0.0) or 0.0,
            )
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Get all open orders."""
        if not self.exchange:
            raise RuntimeError("Not connected to exchange")

        try:
            results = self.exchange.fetch_open_orders(symbol)

            orders = []
            for result in results:
                side = OrderSide.BUY if result.get("side") == "buy" else OrderSide.SELL
                orders.append(Order(
                    id=result.get("id", ""),
                    symbol=result.get("symbol", ""),
                    side=side,
                    order_type=OrderType.MARKET,
                    quantity=result.get("amount", 0.0),
                    price=result.get("price"),
                    status=OrderStatus.OPEN,
                    filled_quantity=result.get("filled", 0.0),
                ))

            return orders

        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def get_positions(self, symbol: str | None = None) -> list[Position]:
        """Get open positions (futures only)."""
        if not self.exchange:
            raise RuntimeError("Not connected to exchange")

        if self.market_type != "futures":
            logger.warning("Positions are only available for futures trading")
            return []

        try:
            results = self.exchange.fetch_positions(symbols=[symbol] if symbol else None)

            positions = []
            for result in results:
                if result.get("contracts", 0) > 0:
                    side = OrderSide.BUY if result.get("side") == "long" else OrderSide.SELL
                    positions.append(Position(
                        symbol=result.get("symbol", ""),
                        side=side,
                        quantity=result.get("contracts", 0.0),
                        entry_price=result.get("entryPrice", 0.0),
                        current_price=result.get("markPrice", 0.0),
                        unrealized_pnl=result.get("unrealizedPnl", 0.0),
                    ))

            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
