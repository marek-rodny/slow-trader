"""Main trading bot implementation."""

import time
from datetime import datetime
from typing import Any
import schedule

from slow_trader.config import Config, StrategyConfig
from slow_trader.exchanges.base import Exchange
from slow_trader.exchanges.demo import DemoExchange
from slow_trader.exchanges.binance import BinanceExchange
from slow_trader.exchanges.alpaca import AlpacaExchange
from slow_trader.strategies.base import Strategy, StrategyManager
from slow_trader.strategies.ma_crossover import MACrossoverStrategy, TripleMAStrategy
from slow_trader.strategies.rsi_strategy import RSIStrategy, RSIDivergenceStrategy
from slow_trader.strategies.macd_strategy import MACDStrategy, MACDHistogramStrategy
from slow_trader.strategies.combined import (
    CombinedStrategy,
    TrendFollowingStrategy,
    MeanReversionStrategy,
)
from slow_trader.risk import RiskManager, RiskLimits
from slow_trader.order_manager import OrderManager
from slow_trader.utils.logger import setup_logger, get_logger

logger = get_logger("slow_trader.bot")


class TradingBot:
    """
    Main trading bot that coordinates all components.

    Monitors prices, analyzes with strategies, and executes trades.
    """

    def __init__(self, config: Config):
        """
        Initialize the trading bot.

        Args:
            config: Bot configuration
        """
        self.config = config
        self.running = False

        # Setup logging
        setup_logger(
            name="slow_trader",
            level=config.log_level,
            log_file=f"{config.data_dir}/logs/bot.log",
        )

        # Initialize exchange
        self.exchange = self._create_exchange()

        # Initialize risk manager
        risk_limits = RiskLimits(
            max_position_size=config.risk.max_position_size,
            max_daily_loss=config.risk.max_daily_loss,
            stop_loss_pct=config.risk.stop_loss_pct,
            take_profit_pct=config.risk.take_profit_pct,
            max_open_positions=config.risk.max_open_positions,
        )
        self.risk_manager = RiskManager(risk_limits)

        # Initialize order manager
        self.order_manager = OrderManager(
            exchange=self.exchange,
            risk_manager=self.risk_manager,
            dry_run=config.dry_run,
        )

        # Initialize strategy manager
        self.strategy_manager = StrategyManager()
        self._setup_strategies()

        logger.info(f"Trading bot initialized (dry_run={config.dry_run})")

    def _create_exchange(self) -> Exchange:
        """Create exchange instance based on config."""
        exchange_name = self.config.exchange.name.lower()

        if exchange_name == "demo":
            exchange = DemoExchange()
            # Generate sample data for demo
            for pair in self.config.trading_pairs:
                exchange.generate_sample_data(pair.symbol, periods=500)

        elif exchange_name == "binance":
            exchange = BinanceExchange(
                api_key=self.config.exchange.api_key,
                api_secret=self.config.exchange.api_secret,
                testnet=self.config.exchange.testnet,
                market_type=self.config.exchange.extra.get("market_type", "spot"),
            )

        elif exchange_name == "alpaca":
            exchange = AlpacaExchange(
                api_key=self.config.exchange.api_key,
                api_secret=self.config.exchange.api_secret,
                testnet=self.config.exchange.testnet,
            )

        else:
            logger.warning(f"Unknown exchange '{exchange_name}', using demo")
            exchange = DemoExchange()

        return exchange

    def _setup_strategies(self) -> None:
        """Setup trading strategies from config."""
        strategy_map = {
            "ma_crossover": MACrossoverStrategy,
            "triple_ma": TripleMAStrategy,
            "rsi": RSIStrategy,
            "rsi_divergence": RSIDivergenceStrategy,
            "macd": MACDStrategy,
            "macd_histogram": MACDHistogramStrategy,
            "combined": CombinedStrategy,
            "trend_following": TrendFollowingStrategy,
            "mean_reversion": MeanReversionStrategy,
        }

        for strat_config in self.config.strategies:
            if not strat_config.enabled:
                continue

            strategy_class = strategy_map.get(strat_config.name)
            if strategy_class:
                strategy = strategy_class(**strat_config.params)
                self.strategy_manager.add_strategy(strategy)
                logger.info(f"Strategy loaded: {strat_config.name}")
            else:
                logger.warning(f"Unknown strategy: {strat_config.name}")

        if not self.strategy_manager.strategies:
            # Default to combined strategy
            logger.info("No strategies configured, using default combined strategy")
            self.strategy_manager.add_strategy(CombinedStrategy())

    def connect(self) -> bool:
        """Connect to the exchange."""
        return self.exchange.connect()

    def disconnect(self) -> None:
        """Disconnect from the exchange."""
        self.exchange.disconnect()

    def is_trading_time(self) -> bool:
        """Check if current time is within trading hours."""
        now = datetime.now()

        # Check trading days
        if now.weekday() not in self.config.trading_days:
            return False

        # Check trading hours
        hour = now.hour
        if hour < self.config.trading_hours_start or hour >= self.config.trading_hours_end:
            return False

        return True

    def analyze_symbol(self, symbol: str) -> dict[str, Any]:
        """
        Analyze a single symbol and return signals.

        Args:
            symbol: Trading pair symbol

        Returns:
            Dictionary with analysis results
        """
        try:
            # Get OHLCV data
            data = self.exchange.get_ohlcv(symbol, timeframe="1h", limit=200)

            if data is None or len(data) < 50:
                logger.warning(f"Insufficient data for {symbol}")
                return {"symbol": symbol, "error": "Insufficient data"}

            # Get ticker
            ticker = self.exchange.get_ticker(symbol)
            current_price = ticker.get("last", 0)

            # Get signals from all strategies
            signals = self.strategy_manager.get_signals(data, symbol)

            # Get consensus signal
            consensus = self.strategy_manager.get_consensus(data, symbol)

            return {
                "symbol": symbol,
                "price": current_price,
                "consensus": {
                    "signal": consensus.signal.value,
                    "strength": consensus.strength,
                    "reason": consensus.reason,
                },
                "strategies": [
                    {
                        "name": s.strategy,
                        "signal": s.signal.value,
                        "strength": s.strength,
                        "reason": s.reason,
                    }
                    for s in signals
                ],
            }

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}

    def check_and_trade(self) -> None:
        """
        Main trading loop iteration.

        Analyzes all trading pairs and executes signals.
        """
        if not self.is_trading_time():
            logger.debug("Outside trading hours")
            return

        logger.info("Running trading check...")

        # Get portfolio value
        portfolio_value = self.exchange.get_portfolio_value()
        self.risk_manager.update_portfolio_peak(portfolio_value)

        # Check each trading pair
        for pair in self.config.trading_pairs:
            try:
                logger.info(f"Analyzing {pair.symbol}...")

                # Get OHLCV data
                data = self.exchange.get_ohlcv(pair.symbol, timeframe="1h", limit=200)

                if data is None or len(data) < 50:
                    logger.warning(f"Insufficient data for {pair.symbol}")
                    continue

                # Get consensus signal
                signal = self.strategy_manager.get_consensus(data, pair.symbol)

                logger.info(
                    f"{pair.symbol}: {signal.signal.value} "
                    f"(strength: {signal.strength:.2f}) - {signal.reason}"
                )

                # Execute signal if actionable
                if signal.is_actionable():
                    order = self.order_manager.execute_signal(signal, portfolio_value)
                    if order:
                        logger.info(f"Order executed: {order}")

            except Exception as e:
                logger.error(f"Error processing {pair.symbol}: {e}")

        # Check existing positions
        self.order_manager.check_positions()

        logger.info("Trading check complete")

    def run(self) -> None:
        """
        Start the trading bot.

        Runs continuously, checking for signals at configured intervals.
        """
        if not self.connect():
            logger.error("Failed to connect to exchange")
            return

        self.running = True
        logger.info(f"Trading bot started (checking every {self.config.check_interval_minutes} minutes)")

        # Schedule the trading check
        schedule.every(self.config.check_interval_minutes).minutes.do(self.check_and_trade)

        # Run initial check
        self.check_and_trade()

        try:
            while self.running:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the trading bot."""
        self.running = False
        self.disconnect()
        logger.info("Trading bot stopped")

        # Print summary
        stats = self.risk_manager.get_stats()
        logger.info("=" * 50)
        logger.info("Trading Session Summary")
        logger.info("=" * 50)
        logger.info(f"Total trades: {stats['total_trades']}")
        logger.info(f"Winning trades: {stats['winning_trades']}")
        logger.info(f"Win rate: {stats['win_rate']:.1%}")
        logger.info(f"Total PnL: ${stats['total_pnl']:.2f}")
        logger.info(f"Daily PnL: ${stats['daily_pnl']:.2f}")

    def run_once(self) -> dict[str, Any]:
        """
        Run a single analysis iteration.

        Useful for testing or manual triggering.

        Returns:
            Analysis results for all pairs
        """
        if not self.connect():
            return {"error": "Failed to connect to exchange"}

        results = {
            "timestamp": datetime.now().isoformat(),
            "trading_time": self.is_trading_time(),
            "portfolio_value": self.exchange.get_portfolio_value(),
            "positions": self.order_manager.get_positions_summary(),
            "analyses": [],
        }

        for pair in self.config.trading_pairs:
            analysis = self.analyze_symbol(pair.symbol)
            results["analyses"].append(analysis)

        return results

    def backtest(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """
        Run a simple backtest on historical data.

        Args:
            symbol: Trading pair to backtest
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Backtest results
        """
        if not self.connect():
            return {"error": "Failed to connect to exchange"}

        logger.info(f"Running backtest for {symbol}...")

        # Get historical data
        data = self.exchange.get_ohlcv(symbol, timeframe="1h", limit=500)

        if data is None or len(data) < 100:
            return {"error": "Insufficient data for backtest"}

        # Track simulated trades
        trades = []
        position = None
        portfolio = 10000.0
        initial_portfolio = portfolio

        # Iterate through data
        for i in range(100, len(data)):
            # Get data window
            window = data.iloc[:i + 1]
            current_price = window["close"].iloc[-1]

            # Get signal
            signal = self.strategy_manager.get_consensus(window, symbol)

            # Execute signal
            if signal.signal.value == "buy" and position is None:
                # Enter long
                quantity = (portfolio * 0.1) / current_price
                position = {
                    "side": "long",
                    "quantity": quantity,
                    "entry_price": current_price,
                    "entry_idx": i,
                }
                trades.append({
                    "type": "buy",
                    "price": current_price,
                    "quantity": quantity,
                    "idx": i,
                })

            elif signal.signal.value == "sell" and position and position["side"] == "long":
                # Exit long
                pnl = (current_price - position["entry_price"]) * position["quantity"]
                portfolio += pnl
                trades.append({
                    "type": "sell",
                    "price": current_price,
                    "quantity": position["quantity"],
                    "pnl": pnl,
                    "idx": i,
                })
                position = None

        # Close any open position
        if position:
            final_price = data["close"].iloc[-1]
            pnl = (final_price - position["entry_price"]) * position["quantity"]
            portfolio += pnl
            trades.append({
                "type": "close",
                "price": final_price,
                "quantity": position["quantity"],
                "pnl": pnl,
                "idx": len(data) - 1,
            })

        # Calculate results
        total_trades = len([t for t in trades if t["type"] in ("buy", "sell")])
        winning_trades = len([t for t in trades if t.get("pnl", 0) > 0])
        total_pnl = portfolio - initial_portfolio
        return_pct = (portfolio / initial_portfolio - 1) * 100

        return {
            "symbol": symbol,
            "period": f"{len(data)} candles",
            "initial_portfolio": initial_portfolio,
            "final_portfolio": portfolio,
            "total_pnl": total_pnl,
            "return_pct": return_pct,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "trades": trades,
        }
