"""Command-line interface for the trading bot."""

import sys
from pathlib import Path
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="slow-trader")
def main():
    """
    Slow Trader - A rule-based trading bot for stocks, crypto, and forex.

    This bot monitors prices, applies technical indicators, and places trades
    based on configurable strategies. Designed for slow, thoughtful trading.
    """
    pass


@main.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    default="config.yaml",
    help="Path to configuration file",
)
@click.option(
    "--dry-run/--live",
    default=True,
    help="Run in dry-run (paper trading) mode",
)
def run(config: str, dry_run: bool):
    """Start the trading bot."""
    from slow_trader.config import Config
    from slow_trader.bot import TradingBot

    console.print(Panel.fit(
        "[bold green]Slow Trader[/bold green]\n"
        f"Config: {config}\n"
        f"Mode: {'Dry Run (Paper Trading)' if dry_run else '[red]LIVE TRADING[/red]'}",
        title="Starting Bot",
    ))

    if not dry_run:
        if not click.confirm(
            "⚠️  You are about to start LIVE trading with real money. Continue?",
            default=False,
        ):
            console.print("[yellow]Aborted.[/yellow]")
            return

    try:
        cfg = Config.from_yaml(config)
        cfg.dry_run = dry_run

        bot = TradingBot(cfg)
        bot.run()

    except FileNotFoundError:
        console.print(f"[red]Error: Config file '{config}' not found[/red]")
        console.print("Run 'slow-trader init' to create a default config")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@main.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    default="config.yaml",
    help="Path to configuration file",
)
@click.option(
    "--symbol", "-s",
    help="Analyze specific symbol only",
)
def analyze(config: str, symbol: str | None):
    """Analyze markets without trading."""
    from slow_trader.config import Config
    from slow_trader.bot import TradingBot

    console.print("[bold]Running market analysis...[/bold]\n")

    try:
        cfg = Config.from_yaml(config)
        cfg.dry_run = True

        bot = TradingBot(cfg)
        results = bot.run_once()

        # Display results
        console.print(f"[dim]Timestamp: {results['timestamp']}[/dim]")
        console.print(f"Portfolio Value: ${results['portfolio_value']:,.2f}\n")

        # Filter by symbol if specified
        analyses = results["analyses"]
        if symbol:
            analyses = [a for a in analyses if a.get("symbol") == symbol]

        for analysis in analyses:
            if "error" in analysis:
                console.print(f"[red]{analysis['symbol']}: {analysis['error']}[/red]")
                continue

            # Create analysis table
            table = Table(title=f"[bold]{analysis['symbol']}[/bold] @ ${analysis['price']:,.2f}")

            table.add_column("Strategy", style="cyan")
            table.add_column("Signal", style="bold")
            table.add_column("Strength")
            table.add_column("Reason", style="dim")

            # Add consensus row
            consensus = analysis["consensus"]
            signal_color = {
                "buy": "green",
                "sell": "red",
                "hold": "yellow",
            }.get(consensus["signal"], "white")

            table.add_row(
                "[bold]CONSENSUS[/bold]",
                f"[{signal_color}]{consensus['signal'].upper()}[/{signal_color}]",
                f"{consensus['strength']:.2f}",
                consensus["reason"],
            )

            table.add_section()

            # Add strategy rows
            for strat in analysis.get("strategies", []):
                signal_color = {
                    "buy": "green",
                    "sell": "red",
                    "hold": "yellow",
                }.get(strat["signal"], "white")

                table.add_row(
                    strat["name"],
                    f"[{signal_color}]{strat['signal'].upper()}[/{signal_color}]",
                    f"{strat['strength']:.2f}",
                    strat["reason"][:50] + "..." if len(strat["reason"]) > 50 else strat["reason"],
                )

            console.print(table)
            console.print()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@main.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    default="config.yaml",
    help="Path to configuration file",
)
@click.argument("symbol")
def backtest(config: str, symbol: str):
    """Run a backtest on historical data."""
    from slow_trader.config import Config
    from slow_trader.bot import TradingBot

    console.print(f"[bold]Running backtest for {symbol}...[/bold]\n")

    try:
        cfg = Config.from_yaml(config)
        cfg.dry_run = True

        bot = TradingBot(cfg)
        results = bot.backtest(symbol)

        if "error" in results:
            console.print(f"[red]Error: {results['error']}[/red]")
            return

        # Display results
        console.print(Panel.fit(
            f"[bold]Backtest Results: {symbol}[/bold]\n"
            f"Period: {results['period']}\n"
            f"Initial Portfolio: ${results['initial_portfolio']:,.2f}\n"
            f"Final Portfolio: ${results['final_portfolio']:,.2f}\n"
            f"Total PnL: ${results['total_pnl']:,.2f}\n"
            f"Return: {results['return_pct']:.2f}%\n"
            f"Total Trades: {results['total_trades']}\n"
            f"Winning Trades: {results['winning_trades']}\n"
            f"Win Rate: {results['win_rate']:.1%}",
            title="Backtest Summary",
            border_style="green" if results['total_pnl'] > 0 else "red",
        ))

        # Show trade details
        if results.get("trades"):
            console.print("\n[bold]Trade History:[/bold]")
            table = Table()
            table.add_column("#", style="dim")
            table.add_column("Type")
            table.add_column("Price", justify="right")
            table.add_column("Quantity", justify="right")
            table.add_column("PnL", justify="right")

            for i, trade in enumerate(results["trades"], 1):
                pnl = trade.get("pnl")
                pnl_str = ""
                if pnl is not None:
                    color = "green" if pnl > 0 else "red"
                    pnl_str = f"[{color}]${pnl:,.2f}[/{color}]"

                trade_color = {
                    "buy": "green",
                    "sell": "red",
                    "close": "yellow",
                }.get(trade["type"], "white")

                table.add_row(
                    str(i),
                    f"[{trade_color}]{trade['type'].upper()}[/{trade_color}]",
                    f"${trade['price']:,.2f}",
                    f"{trade['quantity']:.6f}",
                    pnl_str,
                )

            console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@main.command()
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="config.yaml",
    help="Output path for config file",
)
@click.option(
    "--exchange", "-e",
    type=click.Choice(["demo", "binance", "alpaca"]),
    default="demo",
    help="Exchange to configure",
)
def init(output: str, exchange: str):
    """Create a default configuration file."""
    config_content = f'''# Slow Trader Configuration
# Generated by slow-trader init

# Exchange Configuration
exchange:
  name: {exchange}
  testnet: true
  # API keys can be set via environment variables
  api_key_env: "{exchange.upper()}_API_KEY"
  api_secret_env: "{exchange.upper()}_API_SECRET"

# Trading Pairs
trading_pairs:
  - symbol: "BTC/USDT"
    base_currency: "BTC"
    quote_currency: "USDT"
    min_order_size: 0.001
    price_precision: 2
    quantity_precision: 6

  - symbol: "ETH/USDT"
    base_currency: "ETH"
    quote_currency: "USDT"
    min_order_size: 0.01
    price_precision: 2
    quantity_precision: 5

# Trading Strategies
strategies:
  - name: combined
    enabled: true
    params:
      ema_period: 20
      rsi_period: 14
      rsi_overbought: 70
      rsi_oversold: 30
      min_confirmations: 2

  - name: ma_crossover
    enabled: true
    params:
      fast_period: 10
      slow_period: 20
      ma_type: ema

  - name: rsi
    enabled: true
    params:
      period: 14
      overbought: 70
      oversold: 30

# Risk Management
risk:
  max_position_size: 0.1  # Max 10% of portfolio per trade
  max_daily_loss: 0.05    # Stop trading after 5% daily loss
  stop_loss_pct: 0.02     # 2% stop loss
  take_profit_pct: 0.05   # 5% take profit
  max_open_positions: 5   # Max concurrent positions

# Scheduling
check_interval_minutes: 15  # Check for signals every 15 minutes
trading_hours_start: 9      # Start trading at 9 AM
trading_hours_end: 17       # Stop trading at 5 PM
trading_days: [0, 1, 2, 3, 4]  # Monday to Friday (0=Monday)

# General Settings
dry_run: true  # Paper trading mode (set to false for live trading)
log_level: INFO
data_dir: ./data
'''

    output_path = Path(output)
    if output_path.exists():
        if not click.confirm(f"Config file '{output}' already exists. Overwrite?"):
            console.print("[yellow]Aborted.[/yellow]")
            return

    output_path.write_text(config_content)
    console.print(f"[green]Configuration file created: {output}[/green]")
    console.print("\nNext steps:")
    console.print("  1. Edit the config file to customize your settings")
    console.print("  2. Set your API keys as environment variables")
    console.print("  3. Run 'slow-trader analyze' to test your setup")
    console.print("  4. Run 'slow-trader run' to start the bot")


@main.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    default="config.yaml",
    help="Path to configuration file",
)
def status(config: str):
    """Show bot status and open positions."""
    from slow_trader.config import Config
    from slow_trader.bot import TradingBot

    console.print("[bold]Bot Status[/bold]\n")

    try:
        cfg = Config.from_yaml(config)
        cfg.dry_run = True

        bot = TradingBot(cfg)

        if not bot.connect():
            console.print("[red]Failed to connect to exchange[/red]")
            return

        # Get portfolio info
        portfolio_value = bot.exchange.get_portfolio_value()
        positions = bot.exchange.get_positions()
        open_orders = bot.exchange.get_open_orders()

        # Display status
        console.print(Panel.fit(
            f"Exchange: {cfg.exchange.name}\n"
            f"Mode: {'Testnet' if cfg.exchange.testnet else 'Live'}\n"
            f"Dry Run: {cfg.dry_run}\n"
            f"Portfolio Value: ${portfolio_value:,.2f}\n"
            f"Open Positions: {len(positions)}\n"
            f"Open Orders: {len(open_orders)}",
            title="Status",
        ))

        # Show positions
        if positions:
            console.print("\n[bold]Open Positions:[/bold]")
            table = Table()
            table.add_column("Symbol")
            table.add_column("Side")
            table.add_column("Quantity", justify="right")
            table.add_column("Entry", justify="right")
            table.add_column("Current", justify="right")
            table.add_column("PnL", justify="right")

            for pos in positions:
                pnl_color = "green" if pos.unrealized_pnl >= 0 else "red"
                table.add_row(
                    pos.symbol,
                    pos.side.value.upper(),
                    f"{pos.quantity:.6f}",
                    f"${pos.entry_price:,.2f}",
                    f"${pos.current_price:,.2f}",
                    f"[{pnl_color}]${pos.unrealized_pnl:,.2f}[/{pnl_color}]",
                )

            console.print(table)

        # Show open orders
        if open_orders:
            console.print("\n[bold]Open Orders:[/bold]")
            table = Table()
            table.add_column("ID")
            table.add_column("Symbol")
            table.add_column("Side")
            table.add_column("Type")
            table.add_column("Quantity", justify="right")
            table.add_column("Price", justify="right")

            for order in open_orders:
                table.add_row(
                    order.id[:8] + "...",
                    order.symbol,
                    order.side.value.upper(),
                    order.order_type.value,
                    f"{order.quantity:.6f}",
                    f"${order.price:,.2f}" if order.price else "-",
                )

            console.print(table)

        bot.disconnect()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@main.command()
def list_strategies():
    """List all available trading strategies."""
    strategies = [
        ("ma_crossover", "Moving Average Crossover", "Golden/death cross signals"),
        ("triple_ma", "Triple Moving Average", "Three MA alignment strategy"),
        ("rsi", "RSI Overbought/Oversold", "RSI extreme level signals"),
        ("rsi_divergence", "RSI Divergence", "Price-RSI divergence detection"),
        ("macd", "MACD Signal Crossover", "MACD/signal line crossover"),
        ("macd_histogram", "MACD Histogram", "MACD histogram momentum"),
        ("combined", "Combined Multi-Indicator", "Multiple indicator confirmation"),
        ("trend_following", "Trend Following", "ADX-based trend trading"),
        ("mean_reversion", "Mean Reversion", "Bollinger Bands reversion"),
    ]

    console.print("[bold]Available Trading Strategies[/bold]\n")

    table = Table()
    table.add_column("Name", style="cyan")
    table.add_column("Full Name")
    table.add_column("Description")

    for name, full_name, desc in strategies:
        table.add_row(name, full_name, desc)

    console.print(table)
    console.print("\nTo use a strategy, add it to your config.yaml under 'strategies'")


if __name__ == "__main__":
    main()
