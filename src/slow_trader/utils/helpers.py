"""Helper functions for the trading bot."""

from decimal import Decimal, ROUND_DOWN
import math


def round_price(price: float, precision: int = 2) -> float:
    """
    Round a price to the specified precision.

    Args:
        price: The price to round
        precision: Number of decimal places

    Returns:
        Rounded price
    """
    factor = 10 ** precision
    return math.floor(price * factor) / factor


def round_quantity(quantity: float, precision: int = 8) -> float:
    """
    Round a quantity down to the specified precision.

    Args:
        quantity: The quantity to round
        precision: Number of decimal places

    Returns:
        Rounded quantity
    """
    d = Decimal(str(quantity))
    return float(d.quantize(Decimal(10) ** -precision, rounding=ROUND_DOWN))


def calculate_position_size(
    portfolio_value: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss_price: float,
    max_position_pct: float = 0.1,
) -> float:
    """
    Calculate position size based on risk management rules.

    Args:
        portfolio_value: Total portfolio value
        risk_per_trade: Maximum risk per trade as a decimal (e.g., 0.02 for 2%)
        entry_price: Entry price for the trade
        stop_loss_price: Stop loss price
        max_position_pct: Maximum position size as percentage of portfolio

    Returns:
        Position size (quantity)
    """
    if entry_price <= 0 or stop_loss_price <= 0:
        return 0.0

    # Calculate risk per share
    risk_per_share = abs(entry_price - stop_loss_price)
    if risk_per_share == 0:
        return 0.0

    # Calculate position size based on risk
    risk_amount = portfolio_value * risk_per_trade
    position_size = risk_amount / risk_per_share

    # Apply maximum position size limit
    max_position_value = portfolio_value * max_position_pct
    max_position_size = max_position_value / entry_price

    return min(position_size, max_position_size)


def calculate_pnl(
    entry_price: float,
    exit_price: float,
    quantity: float,
    side: str,
) -> float:
    """
    Calculate profit/loss for a trade.

    Args:
        entry_price: Entry price
        exit_price: Exit price
        quantity: Trade quantity
        side: Trade side ('buy' or 'sell')

    Returns:
        Profit/loss amount
    """
    if side.lower() == "buy":
        return (exit_price - entry_price) * quantity
    else:  # sell/short
        return (entry_price - exit_price) * quantity


def calculate_pnl_percent(entry_price: float, exit_price: float, side: str) -> float:
    """
    Calculate profit/loss percentage.

    Args:
        entry_price: Entry price
        exit_price: Exit price
        side: Trade side ('buy' or 'sell')

    Returns:
        Profit/loss percentage
    """
    if entry_price == 0:
        return 0.0

    if side.lower() == "buy":
        return ((exit_price - entry_price) / entry_price) * 100
    else:  # sell/short
        return ((entry_price - exit_price) / entry_price) * 100


def format_currency(amount: float, symbol: str = "$", decimals: int = 2) -> str:
    """
    Format an amount as currency.

    Args:
        amount: Amount to format
        symbol: Currency symbol
        decimals: Number of decimal places

    Returns:
        Formatted currency string
    """
    if amount >= 0:
        return f"{symbol}{amount:,.{decimals}f}"
    else:
        return f"-{symbol}{abs(amount):,.{decimals}f}"


def format_percent(value: float, decimals: int = 2) -> str:
    """
    Format a value as percentage.

    Args:
        value: Value to format (already as percentage, not decimal)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{decimals}f}%"
