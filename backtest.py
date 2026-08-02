"""
Market Strategy — backtesting six classic systems on Bank Nifty.

Downloads daily OHLCV data, runs six rule-based strategies over the same
period, and reports the metrics that decide whether an edge is real:
expectancy, profit factor, and drawdown — not win rate.

Costs are charged on every trade, because that is what killed the first
version of this project (see README).

Usage:
    python backtest.py
    python backtest.py --symbol ^NSEI --period 5y
"""

import argparse
import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import yfinance as yf

COMMISSION_PCT = 0.001   # 0.1% per trade
SLIPPAGE_PCT = 0.0005    # 0.05% per trade
COST_PER_TRADE = COMMISSION_PCT + SLIPPAGE_PCT


@dataclass
class Result:
    strategy: str
    total_return_pct: float
    win_rate_pct: float
    total_trades: int
    profit_factor: float
    max_drawdown_pct: float
    expectancy_pct: float


def load_prices(symbol: str, period: str) -> pd.DataFrame:
    """Fetch daily OHLCV. Fails loudly rather than silently returning junk."""
    df = yf.download(symbol, period=period, interval="1d",
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {symbol!r} over {period!r}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


# ---------------------------------------------------------------------------
# Signal generators
#
# Each returns a Series of positions: 1 = long, 0 = flat.
# Long-only, one position at a time, evaluated on the close.
# ---------------------------------------------------------------------------

def rsi_signals(df: pd.DataFrame, period: int = 14,
                low: int = 30, high: int = 70) -> pd.Series:
    """Buy oversold, sell overbought. Classic mean reversion."""
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()

    # A window with no down-closes has no average loss to divide by. RSI is
    # defined as 100 there — maximally overbought — but mapping the zero to NaN
    # made it undefined instead, so the overbought exit never fired during the
    # strongest part of a rally. A flat window (no gains either) is 50.
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100.0)
    rsi = rsi.mask((loss == 0) & (gain == 0), 50.0)

    position, holding = [], 0
    for value in rsi:
        if np.isnan(value):
            position.append(0)
            continue
        if holding == 0 and value < low:
            holding = 1
        elif holding == 1 and value > high:
            holding = 0
        position.append(holding)
    return pd.Series(position, index=df.index)


def macd_signals(df: pd.DataFrame) -> pd.Series:
    """Long while the MACD line sits above its signal line."""
    fast = df["Close"].ewm(span=12, adjust=False).mean()
    slow = df["Close"].ewm(span=26, adjust=False).mean()
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False).mean()
    return (macd > signal).astype(int)


def ema_cross_signals(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """Golden cross / death cross."""
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    return (ema_fast > ema_slow).astype(int)


def donchian_signals(df: pd.DataFrame, entry: int = 20, exit_: int = 10) -> pd.Series:
    """Turtle-style breakout: buy N-day highs, exit on M-day lows."""
    upper = df["High"].rolling(entry).max().shift(1)
    lower = df["Low"].rolling(exit_).min().shift(1)

    position, holding = [], 0
    for close, up, down in zip(df["Close"], upper, lower):
        if np.isnan(up) or np.isnan(down):
            position.append(0)
            continue
        if holding == 0 and close > up:
            holding = 1
        elif holding == 1 and close < down:
            holding = 0
        position.append(holding)
    return pd.Series(position, index=df.index)


def bollinger_signals(df: pd.DataFrame, period: int = 20, sd: float = 2.0) -> pd.Series:
    """Buy the lower band, exit at the mean. Mean reversion."""
    mid = df["Close"].rolling(period).mean()
    dev = df["Close"].rolling(period).std()
    lower = mid - sd * dev

    position, holding = [], 0
    for close, low_band, middle in zip(df["Close"], lower, mid):
        if np.isnan(low_band):
            position.append(0)
            continue
        if holding == 0 and close < low_band:
            holding = 1
        elif holding == 1 and close > middle:
            holding = 0
        position.append(holding)
    return pd.Series(position, index=df.index)


def supertrend_signals(df: pd.DataFrame, period: int = 10,
                       multiplier: float = 3.0) -> pd.Series:
    """ATR-based trend following."""
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean()

    hl2 = (df["High"] + df["Low"]) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    position, holding = [], 0
    for close, up, down in zip(df["Close"], upper, lower):
        if np.isnan(up):
            position.append(0)
            continue
        if holding == 0 and close > up:
            holding = 1
        elif holding == 1 and close < down:
            holding = 0
        position.append(holding)
    return pd.Series(position, index=df.index)


STRATEGIES = {
    "rsi": ("RSI Oversold/Overbought", rsi_signals),
    "macd": ("MACD Crossover", macd_signals),
    "ema_cross": ("EMA 20/50 Golden/Death Cross", ema_cross_signals),
    "donchian": ("Donchian Channel Breakout", donchian_signals),
    "bollinger": ("Bollinger Band Mean Reversion", bollinger_signals),
    "supertrend": ("Supertrend (ATR Trend Following)", supertrend_signals),
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def extract_trades(df: pd.DataFrame, position: pd.Series) -> list:
    """Walk the position series and return each completed trade's net return."""
    trades = []
    entry_price = None

    for i in range(1, len(position)):
        was_flat = position.iloc[i - 1] == 0
        now_long = position.iloc[i] == 1

        if was_flat and now_long:
            entry_price = df["Close"].iloc[i]
        elif not was_flat and not now_long and entry_price is not None:
            exit_price = df["Close"].iloc[i]
            gross = (exit_price - entry_price) / entry_price
            trades.append(gross - COST_PER_TRADE)   # costs on every trade
            entry_price = None

    # close any open position at the final bar
    if entry_price is not None:
        gross = (df["Close"].iloc[-1] - entry_price) / entry_price
        trades.append(gross - COST_PER_TRADE)

    return trades


def evaluate(name: str, trades: list) -> Result:
    """Turn a list of trade returns into the metrics that matter."""
    if not trades:
        return Result(name, 0.0, 0.0, 0, 0.0, 0.0, 0.0)

    arr = np.array(trades)
    wins, losses = arr[arr > 0], arr[arr <= 0]

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    equity = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(equity)
    max_dd = ((equity - peak) / peak).min() * 100

    return Result(
        strategy=name,
        total_return_pct=round((equity[-1] - 1) * 100, 2),
        win_rate_pct=round(len(wins) / len(arr) * 100, 1),
        total_trades=len(arr),
        profit_factor=round(profit_factor, 2),
        max_drawdown_pct=round(max_dd, 2),
        expectancy_pct=round(arr.mean() * 100, 2),
    )


def buy_and_hold(df: pd.DataFrame) -> float:
    """The benchmark every strategy has to beat to be worth running."""
    return round((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest six classic strategies.")
    parser.add_argument("--symbol", default="^NSEBANK", help="Yahoo Finance ticker")
    parser.add_argument("--period", default="2y", help="e.g. 1y, 2y, 5y")
    parser.add_argument("--out", default="results/comparison_local.json")
    args = parser.parse_args()

    df = load_prices(args.symbol, args.period)
    benchmark = buy_and_hold(df)

    print(f"\n{args.symbol}  ·  {len(df)} daily candles  ·  "
          f"{df.index[0].date()} to {df.index[-1].date()}")
    print(f"Buy & hold: {benchmark}%\n")

    results = []
    for key, (label, signal_fn) in STRATEGIES.items():
        trades = extract_trades(df, signal_fn(df))
        results.append(evaluate(label, trades))

    results.sort(key=lambda r: r.total_return_pct, reverse=True)

    header = f"{'Strategy':<34}{'Return':>9}{'Trades':>8}{'Win%':>8}{'PF':>7}{'MaxDD':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.strategy:<34}{r.total_return_pct:>8}%{r.total_trades:>8}"
              f"{r.win_rate_pct:>8}{r.profit_factor:>7}{r.max_drawdown_pct:>8}%")

    beat = [r.strategy for r in results if r.total_return_pct > benchmark]
    print(f"\nBeat buy & hold: {', '.join(beat) if beat else 'none'}")

    with open(args.out, "w") as f:
        json.dump({
            "symbol": args.symbol,
            "period": args.period,
            "candles": len(df),
            "buy_and_hold_return_pct": benchmark,
            "results": [asdict(r) for r in results],
        }, f, indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()