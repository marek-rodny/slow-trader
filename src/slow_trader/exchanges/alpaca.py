"""Alpaca exchange connector for stock trading."""

from datetime import datetime, timedelta
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

logger = get_logger("slow_trader.alpaca")


class AlpacaExchange(Exchange):
    """
    Alpaca exchange connector for commission-free stock trading.

    Supports both paper and live trading.
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
    ):
        """
        Initialize Alpaca connector.

        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            testnet: Use paper trading (default True for safety)
        """
        super().__init__(name="alpaca", testnet=testnet)
        self.api_key = api_key
        self.api_secret = api_secret
        self.api = None
        self.data_api = None

    def connect(self) -> bool:
        """Connect to Alpaca."""
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient

            # Create trading client
            self.api = TradingClient(
                api_key=self.api_key,
                secret_key=self.api_secret,
                paper=self.testnet,
            )

            # Create data client
            self.data_api = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.api_secret,
            )

            # Test connection
            account = self.api.get_account()
            logger.info(
                f"Connected to Alpaca {'paper' if self.testnet else 'live'} trading. "
                f"Account status: {account.status}"
            )
            return True

        except ImportError:
            logger.error("alpaca-trade-api not installed. Run: pip install alpaca-trade-api")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from Alpaca."""
        self.api = None
        self.data_api = None
        logger.info("Disconnected from Alpaca")

    def get_balance(self, currency: str | None = None) -> list[Balance] | Balance:
        """Get account balance."""
        if not self.api:
            raise RuntimeError("Not connected to exchange")

        try:
            account = self.api.get_account()

            # Alpaca accounts are in USD
            cash_balance = Balance(
                currency="USD",
                total=float(account.equity),
                available=float(account.cash),
                locked=float(account.equity) - float(account.cash),
            )

            if currency:
                if currency == "USD":
                    return cash_balance
                return Balance(currency=currency, total=0.0, available=0.0)

            return [cash_balance]

        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            raise

    def get_ticker(self, symbol: str) -> dict[str, float]:
        """Get current ticker data."""
        if not self.data_api:
            raise RuntimeError("Not connected to exchange")

        try:
            from alpaca.data.requests import StockLatestQuoteRequest

            # Remove any slash from symbol (e.g., "AAPL/USD" -> "AAPL")
            clean_symbol = symbol.split("/")[0] if "/" in symbol else symbol

            request = StockLatestQuoteRequest(symbol_or_symbols=clean_symbol)
            quote = self.data_api.get_stock_latest_quote(request)[clean_symbol]

            return {
                "symbol": symbol,
                "bid": float(quote.bid_price),
                "ask": float(quote.ask_price),
                "last": (float(quote.bid_price) + float(quote.ask_price)) / 2,
                "bid_size": float(quote.bid_size),
                "ask_size": float(quote.ask_size),
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
        if not self.data_api:
            raise RuntimeError("Not connected to exchange")

        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            # Clean symbol
            clean_symbol = symbol.split("/")[0] if "/" in symbol else symbol

            # Map timeframe
            tf_map = {
                "1m": TimeFrame.Minute,
                "5m": TimeFrame.Minute,
                "15m": TimeFrame.Minute,
                "1h": TimeFrame.Hour,
                "4h": TimeFrame.Hour,
                "1d": TimeFrame.Day,
            }
            alpaca_tf = tf_map.get(timeframe, TimeFrame.Hour)

            # Calculate start time based on limit
            tf_minutes = {
                "1m": 1, "5m": 5, "15m": 15,
                "1h": 60, "4h": 240, "1d": 1440,
            }
            minutes = tf_minutes.get(timeframe, 60) * limit
            start = datetime.now() - timedelta(minutes=minutes)

            request = StockBarsRequest(
                symbol_or_symbols=clean_symbol,
                timeframe=alpaca_tf,
                start=start,
            )

            bars = self.data_api.get_stock_bars(request)[clean_symbol]

            data = []
            for bar in bars:
                data.append({
                    "timestamp": bar.timestamp,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                })

            df = pd.DataFrame(data)
            return df.tail(limit)

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
        if not self.api:
            raise RuntimeError("Not connected to exchange")

        try:
            from alpaca.trading.requests import (
                MarketOrderRequest,
                LimitOrderRequest,
                StopOrderRequest,
                StopLimitOrderRequest,
            )
            from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce

            # Clean symbol
            clean_symbol = symbol.split("/")[0] if "/" in symbol else symbol

            # Map side
            alpaca_side = AlpacaSide.BUY if side == OrderSide.BUY else AlpacaSide.SELL

            # Create order request based on type
            if order_type == OrderType.MARKET:
                request = MarketOrderRequest(
                    symbol=clean_symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=TimeInForce.DAY,
                )
            elif order_type == OrderType.LIMIT:
                request = LimitOrderRequest(
                    symbol=clean_symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=price,
                )
            elif order_type == OrderType.STOP_LOSS:
                request = StopOrderRequest(
                    symbol=clean_symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=TimeInForce.DAY,
                    stop_price=stop_price,
                )
            elif order_type == OrderType.STOP_LIMIT:
                request = StopLimitOrderRequest(
                    symbol=clean_symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=TimeInForce.DAY,
                    stop_price=stop_price,
                    limit_price=price,
                )
            else:
                request = MarketOrderRequest(
                    symbol=clean_symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=TimeInForce.DAY,
                )

            result = self.api.submit_order(request)

            # Map status
            status_map = {
                "new": OrderStatus.OPEN,
                "accepted": OrderStatus.OPEN,
                "filled": OrderStatus.FILLED,
                "partially_filled": OrderStatus.PARTIALLY_FILLED,
                "canceled": OrderStatus.CANCELLED,
                "rejected": OrderStatus.REJECTED,
            }

            order = Order(
                id=str(result.id),
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                status=status_map.get(str(result.status).lower(), OrderStatus.OPEN),
                filled_quantity=float(result.filled_qty) if result.filled_qty else 0.0,
                filled_price=float(result.filled_avg_price) if result.filled_avg_price else 0.0,
            )

            logger.info(f"Order placed: {order}")
            return order

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order."""
        if not self.api:
            raise RuntimeError("Not connected to exchange")

        try:
            self.api.cancel_order_by_id(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_order(self, order_id: str, symbol: str) -> Order | None:
        """Get order details."""
        if not self.api:
            raise RuntimeError("Not connected to exchange")

        try:
            result = self.api.get_order_by_id(order_id)

            status_map = {
                "new": OrderStatus.OPEN,
                "accepted": OrderStatus.OPEN,
                "filled": OrderStatus.FILLED,
                "partially_filled": OrderStatus.PARTIALLY_FILLED,
                "canceled": OrderStatus.CANCELLED,
                "rejected": OrderStatus.REJECTED,
            }

            side = OrderSide.BUY if str(result.side).lower() == "buy" else OrderSide.SELL

            return Order(
                id=str(result.id),
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=float(result.qty),
                status=status_map.get(str(result.status).lower(), OrderStatus.OPEN),
                filled_quantity=float(result.filled_qty) if result.filled_qty else 0.0,
                filled_price=float(result.filled_avg_price) if result.filled_avg_price else 0.0,
            )
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Get all open orders."""
        if not self.api:
            raise RuntimeError("Not connected to exchange")

        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            results = self.api.get_orders(request)

            orders = []
            for result in results:
                side = OrderSide.BUY if str(result.side).lower() == "buy" else OrderSide.SELL
                orders.append(Order(
                    id=str(result.id),
                    symbol=str(result.symbol),
                    side=side,
                    order_type=OrderType.MARKET,
                    quantity=float(result.qty),
                    status=OrderStatus.OPEN,
                    filled_quantity=float(result.filled_qty) if result.filled_qty else 0.0,
                ))

            if symbol:
                clean_symbol = symbol.split("/")[0] if "/" in symbol else symbol
                orders = [o for o in orders if clean_symbol in o.symbol]

            return orders

        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def get_positions(self, symbol: str | None = None) -> list[Position]:
        """Get open positions."""
        if not self.api:
            raise RuntimeError("Not connected to exchange")

        try:
            results = self.api.get_all_positions()

            positions = []
            for result in results:
                side = OrderSide.BUY if float(result.qty) > 0 else OrderSide.SELL
                positions.append(Position(
                    symbol=str(result.symbol),
                    side=side,
                    quantity=abs(float(result.qty)),
                    entry_price=float(result.avg_entry_price),
                    current_price=float(result.current_price),
                    unrealized_pnl=float(result.unrealized_pl),
                ))

            if symbol:
                clean_symbol = symbol.split("/")[0] if "/" in symbol else symbol
                positions = [p for p in positions if clean_symbol in p.symbol]

            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
