"""Configuration management for the trading bot."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml
import os
from dotenv import load_dotenv


@dataclass
class ExchangeConfig:
    """Configuration for an exchange connection."""

    name: str
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    extra: dict = field(default_factory=dict)


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""

    name: str
    enabled: bool = True
    params: dict = field(default_factory=dict)


@dataclass
class RiskConfig:
    """Risk management configuration."""

    max_position_size: float = 0.1  # Max 10% of portfolio per trade
    max_daily_loss: float = 0.05  # Max 5% daily loss
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.05  # 5% take profit
    max_open_positions: int = 5


@dataclass
class TradingConfig:
    """Trading pair configuration."""

    symbol: str
    base_currency: str
    quote_currency: str
    min_order_size: float = 0.0
    price_precision: int = 2
    quantity_precision: int = 8


@dataclass
class Config:
    """Main configuration for the trading bot."""

    exchange: ExchangeConfig
    strategies: list[StrategyConfig]
    risk: RiskConfig
    trading_pairs: list[TradingConfig]

    # Scheduling
    check_interval_minutes: int = 15
    trading_hours_start: int = 9  # 9 AM
    trading_hours_end: int = 16  # 4 PM
    trading_days: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri

    # General settings
    dry_run: bool = True  # Paper trading mode
    log_level: str = "INFO"
    data_dir: str = "./data"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file."""
        load_dotenv()  # Load environment variables

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls._parse_config(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create configuration from a dictionary."""
        return cls._parse_config(data)

    @classmethod
    def _parse_config(cls, data: dict[str, Any]) -> "Config":
        """Parse configuration from a dictionary."""
        # Parse exchange config
        exchange_data = data.get("exchange", {})
        exchange = ExchangeConfig(
            name=exchange_data.get("name", "demo"),
            api_key=os.environ.get(
                exchange_data.get("api_key_env", ""),
                exchange_data.get("api_key", "")
            ),
            api_secret=os.environ.get(
                exchange_data.get("api_secret_env", ""),
                exchange_data.get("api_secret", "")
            ),
            testnet=exchange_data.get("testnet", True),
            extra=exchange_data.get("extra", {}),
        )

        # Parse strategies
        strategies = []
        for strat_data in data.get("strategies", []):
            strategies.append(StrategyConfig(
                name=strat_data.get("name", "ma_crossover"),
                enabled=strat_data.get("enabled", True),
                params=strat_data.get("params", {}),
            ))

        # Parse risk config
        risk_data = data.get("risk", {})
        risk = RiskConfig(
            max_position_size=risk_data.get("max_position_size", 0.1),
            max_daily_loss=risk_data.get("max_daily_loss", 0.05),
            stop_loss_pct=risk_data.get("stop_loss_pct", 0.02),
            take_profit_pct=risk_data.get("take_profit_pct", 0.05),
            max_open_positions=risk_data.get("max_open_positions", 5),
        )

        # Parse trading pairs
        trading_pairs = []
        for pair_data in data.get("trading_pairs", []):
            trading_pairs.append(TradingConfig(
                symbol=pair_data.get("symbol", "BTC/USDT"),
                base_currency=pair_data.get("base_currency", "BTC"),
                quote_currency=pair_data.get("quote_currency", "USDT"),
                min_order_size=pair_data.get("min_order_size", 0.0),
                price_precision=pair_data.get("price_precision", 2),
                quantity_precision=pair_data.get("quantity_precision", 8),
            ))

        return cls(
            exchange=exchange,
            strategies=strategies,
            risk=risk,
            trading_pairs=trading_pairs,
            check_interval_minutes=data.get("check_interval_minutes", 15),
            trading_hours_start=data.get("trading_hours_start", 9),
            trading_hours_end=data.get("trading_hours_end", 16),
            trading_days=data.get("trading_days", [0, 1, 2, 3, 4]),
            dry_run=data.get("dry_run", True),
            log_level=data.get("log_level", "INFO"),
            data_dir=data.get("data_dir", "./data"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            "exchange": {
                "name": self.exchange.name,
                "testnet": self.exchange.testnet,
                "extra": self.exchange.extra,
            },
            "strategies": [
                {"name": s.name, "enabled": s.enabled, "params": s.params}
                for s in self.strategies
            ],
            "risk": {
                "max_position_size": self.risk.max_position_size,
                "max_daily_loss": self.risk.max_daily_loss,
                "stop_loss_pct": self.risk.stop_loss_pct,
                "take_profit_pct": self.risk.take_profit_pct,
                "max_open_positions": self.risk.max_open_positions,
            },
            "trading_pairs": [
                {
                    "symbol": p.symbol,
                    "base_currency": p.base_currency,
                    "quote_currency": p.quote_currency,
                    "min_order_size": p.min_order_size,
                    "price_precision": p.price_precision,
                    "quantity_precision": p.quantity_precision,
                }
                for p in self.trading_pairs
            ],
            "check_interval_minutes": self.check_interval_minutes,
            "trading_hours_start": self.trading_hours_start,
            "trading_hours_end": self.trading_hours_end,
            "trading_days": self.trading_days,
            "dry_run": self.dry_run,
            "log_level": self.log_level,
            "data_dir": self.data_dir,
        }
