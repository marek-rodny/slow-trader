"""Logging utilities for the trading bot."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler


_loggers: dict[str, logging.Logger] = {}
console = Console()


def setup_logger(
    name: str = "slow_trader",
    level: str = "INFO",
    log_file: str | Path | None = None,
) -> logging.Logger:
    """
    Set up a logger with rich formatting.

    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging

    Returns:
        Configured logger instance
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()

    # Rich console handler for pretty output
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
    )
    rich_handler.setLevel(logging.DEBUG)
    rich_format = logging.Formatter("%(message)s")
    rich_handler.setFormatter(rich_format)
    logger.addHandler(rich_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger


def get_logger(name: str = "slow_trader") -> logging.Logger:
    """
    Get an existing logger or create a new one.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    if name not in _loggers:
        return setup_logger(name)
    return _loggers[name]


class TradeLogger:
    """Specialized logger for trade events."""

    def __init__(self, log_dir: str | Path = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("slow_trader.trades")

        # Trade history file
        self.trade_file = self.log_dir / "trades.log"

    def log_signal(self, symbol: str, signal: str, strategy: str, indicators: dict):
        """Log a trading signal."""
        self.logger.info(
            f"📊 Signal: {signal} for {symbol} from {strategy} | "
            f"Indicators: {indicators}"
        )

    def log_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str,
        order_id: str | None = None,
    ):
        """Log an order placement."""
        self.logger.info(
            f"📝 Order: {side} {quantity} {symbol} @ {price} ({order_type}) | "
            f"ID: {order_id or 'N/A'}"
        )

        # Also write to trade file
        with open(self.trade_file, "a") as f:
            timestamp = datetime.now().isoformat()
            f.write(
                f"{timestamp},{symbol},{side},{quantity},{price},{order_type},{order_id}\n"
            )

    def log_fill(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_id: str,
    ):
        """Log an order fill."""
        self.logger.info(
            f"✅ Filled: {side} {quantity} {symbol} @ {price} | ID: {order_id}"
        )

    def log_error(self, message: str, error: Exception | None = None):
        """Log an error."""
        if error:
            self.logger.error(f"❌ Error: {message} | {type(error).__name__}: {error}")
        else:
            self.logger.error(f"❌ Error: {message}")
