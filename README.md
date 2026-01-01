# Slow Trader

A rule-based trading bot for stocks, crypto, and forex. Designed for slow, thoughtful trading rather than high-frequency operations.

## Features

- **Multiple Exchange Support**: Binance (crypto), Alpaca (stocks), and demo mode for testing
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, ADX, ATR, and more
- **Rule-Based Strategies**: Moving Average Crossover, RSI, MACD, Trend Following, Mean Reversion
- **Risk Management**: Position sizing, stop-loss, take-profit, daily loss limits
- **Configurable Scheduling**: Trade at specific intervals and hours
- **Paper Trading**: Test strategies without risking real money
- **Backtesting**: Evaluate strategies on historical data

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/slow-trader.git
cd slow-trader

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### 2. Configuration

```bash
# Create a default configuration file
slow-trader init --exchange demo

# Edit the configuration
nano config.yaml
```

### 3. Run the Bot

```bash
# Analyze markets (no trading)
slow-trader analyze

# Run in paper trading mode
slow-trader run --dry-run

# Run with live trading (use with caution!)
slow-trader run --live
```

## Configuration

The bot is configured via a YAML file. See `config.example.yaml` for all options.

### Exchange Setup

**Demo Mode** (for testing):
```yaml
exchange:
  name: demo
  testnet: true
```

**Binance** (crypto):
```yaml
exchange:
  name: binance
  testnet: true  # Use testnet for testing!
  api_key_env: "BINANCE_API_KEY"
  api_secret_env: "BINANCE_API_SECRET"
  extra:
    market_type: spot  # or "futures"
```

**Alpaca** (stocks):
```yaml
exchange:
  name: alpaca
  testnet: true  # Use paper trading!
  api_key_env: "ALPACA_API_KEY"
  api_secret_env: "ALPACA_API_SECRET"
```

### Trading Strategies

Available strategies:

| Strategy | Description |
|----------|-------------|
| `ma_crossover` | Moving Average Crossover (golden/death cross) |
| `triple_ma` | Triple Moving Average alignment |
| `rsi` | RSI overbought/oversold levels |
| `rsi_divergence` | RSI divergence detection |
| `macd` | MACD signal line crossover |
| `macd_histogram` | MACD histogram momentum |
| `combined` | Multi-indicator confirmation |
| `trend_following` | ADX-based trend trading |
| `mean_reversion` | Bollinger Bands reversion |

Configure strategies in your config file:

```yaml
strategies:
  - name: combined
    enabled: true
    params:
      ema_period: 20
      rsi_period: 14
      min_confirmations: 2

  - name: ma_crossover
    enabled: true
    params:
      fast_period: 10
      slow_period: 20
      ma_type: ema
```

### Risk Management

```yaml
risk:
  max_position_size: 0.1   # Max 10% per trade
  max_daily_loss: 0.05     # Stop at 5% daily loss
  stop_loss_pct: 0.02      # 2% stop loss
  take_profit_pct: 0.05    # 5% take profit
  max_open_positions: 5    # Max 5 concurrent positions
```

## CLI Commands

```bash
# Initialize configuration
slow-trader init --exchange demo

# Analyze markets
slow-trader analyze -c config.yaml
slow-trader analyze -s BTC/USDT  # Specific symbol

# Run the bot
slow-trader run --dry-run
slow-trader run --live

# Backtest a strategy
slow-trader backtest BTC/USDT

# Check status
slow-trader status

# List available strategies
slow-trader list-strategies
```

## Project Structure

```
slow-trader/
├── src/slow_trader/
│   ├── __init__.py
│   ├── bot.py              # Main bot logic
│   ├── cli.py              # Command-line interface
│   ├── config.py           # Configuration management
│   ├── risk.py             # Risk management
│   ├── order_manager.py    # Order execution
│   ├── indicators/         # Technical indicators
│   │   ├── moving_averages.py
│   │   ├── momentum.py
│   │   ├── volatility.py
│   │   └── trend.py
│   ├── exchanges/          # Exchange connectors
│   │   ├── demo.py
│   │   ├── binance.py
│   │   └── alpaca.py
│   ├── strategies/         # Trading strategies
│   │   ├── ma_crossover.py
│   │   ├── rsi_strategy.py
│   │   ├── macd_strategy.py
│   │   └── combined.py
│   └── utils/
│       ├── logger.py
│       └── helpers.py
├── config.example.yaml
├── requirements.txt
└── README.md
```

## Safety Warnings

**IMPORTANT: This bot can trade with real money. Use at your own risk!**

1. **Always start with paper trading** (`dry_run: true`)
2. **Use testnet/sandbox** when available
3. **Never risk more than you can afford to lose**
4. **Backtest strategies before live trading**
5. **Monitor the bot regularly**
6. **Keep API keys secure** - use environment variables

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
ruff src/
```

## License

MIT License - See LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Trading cryptocurrencies, stocks, and forex carries significant risk. Past performance does not guarantee future results. Always do your own research and consider consulting a financial advisor before trading.
