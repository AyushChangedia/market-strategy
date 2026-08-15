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
from typing import Optional

import numpy as np
import pandas as pd

COMMISSION_PCT = 0.001   # 0.1% per trade
SLIPPAGE_PCT = 0.0005    # 0.05% per trade
COST_PER_TRADE = COMMISSION_PCT + SLIPPAGE_PCT


@dataclass
class Result:
    strategy: str
    total_return_pct: float
    win_rate_pct: float
    total_trades: int
    scratch_trades: int
    # None when there are no losing trades to divide by — an undefined ratio,
    # not an infinite one. float("inf") serialises as the bare token Infinity,
    # which json.dump emits happily but strict JSON parsers reject.
    profit_factor: Optional[float]
    sharpe_ratio: float
    max_drawdown_pct: float
    expectancy_pct: float


def load_prices(symbol: str, period: str) -> pd.DataFrame:
    """Fetch daily OHLCV. Fails loudly rather than silently returning junk."""
    # Imported here so the signal and evaluation functions — which are pure and
    # touch no network — can be imported and tested without yfinance installed.
    import yfinance as yf

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

def rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI over a close series, defined at both extremes.

    A window with no down-closes has no average loss to divide by. RSI is 100
    there — maximally overbought — but mapping the zero to NaN left it
    undefined instead, so the overbought exit never fired during the strongest
    part of a rally. A window with no movement at all is 50.

    Split out from rsi_signals so the indicator can be asserted on directly;
    the position series it feeds is integral and cannot express "undefined".
    """
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()

    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100.0)
    rsi = rsi.mask((loss == 0) & (gain == 0), 50.0)
    return rsi


def rsi_signals(df: pd.DataFrame, period: int = 14,
                low: int = 30, high: int = 70) -> pd.Series:
    """Buy oversold, sell overbought. Classic mean reversion."""
    rsi = rsi_series(df["Close"], period)

    position, holding = [], 0
    for value in rsi:
        if np.isnan(value):
            position.append(holding)
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
            position.append(holding)
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
            position.append(holding)
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
            position.append(holding)
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
        return Result(strategy=name, total_return_pct=0.0, win_rate_pct=0.0,
                      total_trades=0, scratch_trades=0, profit_factor=0.0,
                      sharpe_ratio=0.0, max_drawdown_pct=0.0,
                      expectancy_pct=0.0)

    arr = np.array(trades)
    # A trade that came back exactly flat is neither a win nor a loss. Bundling
    # it with the losers understated win rate and, because it adds nothing to
    # gross loss, quietly moved the profit factor's denominator count without
    # moving the denominator.
    wins, losses = arr[arr > 0], arr[arr < 0]
    scratches = int((arr == 0).sum())

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

    equity = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(equity)
    max_dd = ((equity - peak) / peak).min() * 100

    # Sharpe over the trade sequence: mean return per trade against its own
    # dispersion, annualised by the number of trades actually taken. Reported
    # per-trade rather than per-day because the strategies hold for wildly
    # different spans, so a daily series would be mostly zeroes.
    sharpe = 0.0
    if len(arr) > 1 and arr.std(ddof=1) > 0:
        sharpe = float(arr.mean() / arr.std(ddof=1) * np.sqrt(len(arr)))

    return Result(
        strategy=name,
        total_return_pct=round((equity[-1] - 1) * 100, 2),
        win_rate_pct=round(len(wins) / len(arr) * 100, 1),
        scratch_trades=scratches,
        total_trades=len(arr),
        profit_factor=None if profit_factor is None else round(profit_factor, 2),
        sharpe_ratio=round(sharpe, 2),
        max_drawdown_pct=round(max_dd, 2),
        expectancy_pct=round(arr.mean() * 100, 2),
    )


def buy_and_hold(df: pd.DataFrame) -> float:
    """The benchmark every strategy has to beat to be worth running."""
    return round((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100, 2)


# ---------------------------------------------------------------------------
# Walk-forward validation
#
# A single backtest number says almost nothing: the parameters were chosen
# knowing the whole series. Walk-forward splits the history into consecutive
# blocks, fits nothing but *evaluates* on a held-out tail of each block, and
# asks how much of the in-sample return survived. That is what Finding 3 in the
# README rests on, and until now the code that produced it was not in the repo.
# ---------------------------------------------------------------------------

@dataclass
class Fold:
    fold: int
    train_from: str
    train_to: str
    train_return_pct: float
    train_trades: int
    test_from: str
    test_to: str
    test_return_pct: float
    test_trades: int
    fold_robustness_score: float


def fold_bounds(n_candles: int, n_splits: int = 3,
                train_ratio: float = 0.7) -> list:
    """
    Consecutive non-overlapping blocks, each cut into a train then a test half.

    Blocks do not overlap and preserve order, so no fold is ever evaluated on
    data that precedes the data it was measured against.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must sit strictly between 0 and 1")

    size = n_candles // n_splits
    if size < 2:
        raise ValueError(
            f"{n_candles} candles cannot be split into {n_splits} usable folds"
        )

    bounds = []
    for i in range(n_splits):
        start = i * size
        cut = start + int(size * train_ratio)
        bounds.append(((start, cut), (cut, start + size)))
    return bounds


def fold_robustness(train_pct: float, test_pct: float) -> float:
    """
    The share of the in-sample return that survived out of sample, clamped 0..1.

    A fold that made nothing in training has nothing to degrade from, so it
    scores 1.0 provided the test half did not do worse.
    """
    if train_pct <= 0:
        return 1.0 if test_pct >= train_pct else 0.0
    return round(max(0.0, min(test_pct / train_pct, 1.0)), 2)


VERDICTS = (
    (0.7, "ROBUST - out-of-sample performance held up"),
    (0.4, "WEAK - significant out-of-sample degradation, likely overfitted"),
    (0.0, "FRAGILE - out-of-sample performance collapsed"),
)


def verdict_for(score: float) -> str:
    for threshold, text in VERDICTS:
        if score >= threshold:
            return text
    return VERDICTS[-1][1]


def segment_performance(df: pd.DataFrame, signal_fn) -> tuple:
    """Compound return and trade count for one slice, evaluated in isolation."""
    trades = extract_trades(df, signal_fn(df))
    if not trades:
        return 0.0, 0
    equity = np.cumprod(1 + np.array(trades))
    return round((equity[-1] - 1) * 100, 2), len(trades)


def _day(df: pd.DataFrame, i: int) -> str:
    return str(pd.Timestamp(df.index[i]).date())


def walk_forward(df: pd.DataFrame, strategy: str = "rsi",
                 n_splits: int = 3, train_ratio: float = 0.7) -> dict:
    """Run one strategy fold by fold and report how well it held up."""
    if strategy not in STRATEGIES:
        raise KeyError(f"unknown strategy {strategy!r}; pick one of {sorted(STRATEGIES)}")
    label, signal_fn = STRATEGIES[strategy]

    folds = []
    for i, ((tr_start, tr_end), (te_start, te_end)) in enumerate(
            fold_bounds(len(df), n_splits, train_ratio), start=1):
        train, test = df.iloc[tr_start:tr_end], df.iloc[te_start:te_end]
        train_pct, train_n = segment_performance(train, signal_fn)
        test_pct, test_n = segment_performance(test, signal_fn)

        folds.append(Fold(
            fold=i,
            train_from=_day(df, tr_start), train_to=_day(df, tr_end - 1),
            train_return_pct=train_pct, train_trades=train_n,
            test_from=_day(df, te_start), test_to=_day(df, te_end - 1),
            test_return_pct=test_pct, test_trades=test_n,
            fold_robustness_score=fold_robustness(train_pct, test_pct),
        ))

    scores = [f.fold_robustness_score for f in folds]
    robustness = round(float(np.mean(scores)), 2)
    oos = np.cumprod([1 + f.test_return_pct / 100 for f in folds])

    return {
        "symbol": None,
        "strategy": strategy,
        "label": label,
        "n_splits": n_splits,
        "train_ratio": train_ratio,
        "total_candles": len(df),
        "date_from": _day(df, 0),
        "date_to": _day(df, len(df) - 1),
        "avg_train_return_pct": round(float(np.mean(
            [f.train_return_pct for f in folds])), 2),
        "avg_test_return_pct": round(float(np.mean(
            [f.test_return_pct for f in folds])), 2),
        "robustness_score": robustness,
        "verdict": verdict_for(robustness),
        "oos_total_trades": sum(f.test_trades for f in folds),
        "oos_total_return_pct": round(float(oos[-1] - 1) * 100, 2),
        "buy_and_hold_return_pct": buy_and_hold(df),
        "folds": [asdict(f) for f in folds],
        "disclaimer": "Past performance does not guarantee future results. "
                      "Educational use only.",
    }


# Yahoo tickers carry no display name, so the few this project uses are mapped
# by hand. Anything else falls back to the ticker itself.
INSTRUMENT_NAMES = {
    "^NSEBANK": "Bank Nifty Index",
    "^NSEI": "Nifty 50 Index",
}


def comparison_report(symbol: str, period: str, df: pd.DataFrame,
                      benchmark: float, ranked: list) -> dict:
    """
    Build the results document in the shape make_charts.py reads.

    The two had drifted: this wrote {"results": [...]} keyed by display label,
    while the charts read {"ranking": [...]} keyed by strategy id. Regenerating
    the results therefore produced a file that crashed chart generation, and
    only the committed copy — written by an earlier version of this script —
    still worked.
    """
    return {
        "symbol": symbol,
        "instrument": INSTRUMENT_NAMES.get(symbol, symbol),
        "period": period,
        "interval": "1d",
        "candles_analyzed": len(df),
        "date_from": _day(df, 0),
        "date_to": _day(df, len(df) - 1),
        "commission_pct": round(COMMISSION_PCT * 100, 4),
        "slippage_pct": round(SLIPPAGE_PCT * 100, 4),
        "buy_and_hold_return_pct": benchmark,
        "ranking": [
            {
                "rank": i,
                "strategy": key,
                "label": r.strategy,
                **{k: v for k, v in asdict(r).items() if k != "strategy"},
            }
            for i, (key, r) in enumerate(ranked, start=1)
        ],
        "disclaimer": "Past performance does not guarantee future results. "
                      "Educational use only.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest six classic strategies.")
    parser.add_argument("--symbol", default="^NSEBANK", help="Yahoo Finance ticker")
    parser.add_argument("--period", default="2y", help="e.g. 1y, 2y, 5y")
    parser.add_argument("--out", default="results/comparison_local.json")
    parser.add_argument("--walk-forward", metavar="STRATEGY",
                        help="validate one strategy fold by fold, e.g. rsi")
    parser.add_argument("--splits", type=int, default=3,
                        help="walk-forward folds (default 3)")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="share of each fold used in sample (default 0.7)")
    args = parser.parse_args()

    df = load_prices(args.symbol, args.period)

    if args.walk_forward:
        report = walk_forward(df, args.walk_forward, args.splits, args.train_ratio)
        report["symbol"] = args.symbol
        report["period"] = args.period

        print(f"\n{report['label']}  ·  {report['n_splits']} folds  ·  "
              f"{report['date_from']} to {report['date_to']}\n")
        head = f"{'Fold':<7}{'In-sample':>12}{'Out-of-sample':>16}{'Retained':>11}"
        print(head)
        print("-" * len(head))
        for f in report["folds"]:
            print(f"{f['fold']:<7}{f['train_return_pct']:>11}%"
                  f"{f['test_return_pct']:>15}%{f['fold_robustness_score']:>11}")
        print(f"\nRobustness {report['robustness_score']}  ·  {report['verdict']}")

        out = args.out.replace("comparison_local", f"walk_forward_{args.walk_forward}_local")
        with open(out, "w") as f:
            json.dump(report, f, indent=2, allow_nan=False)
        print(f"Saved to {out}")
        return

    benchmark = buy_and_hold(df)

    print(f"\n{args.symbol}  ·  {len(df)} daily candles  ·  "
          f"{df.index[0].date()} to {df.index[-1].date()}")
    print(f"Buy & hold: {benchmark}%\n")

    results = []
    for key, (label, signal_fn) in STRATEGIES.items():
        trades = extract_trades(df, signal_fn(df))
        results.append((key, evaluate(label, trades)))

    results.sort(key=lambda pair: pair[1].total_return_pct, reverse=True)

    header = f"{'Strategy':<34}{'Return':>9}{'Trades':>8}{'Win%':>8}{'PF':>7}{'MaxDD':>9}"
    print(header)
    print("-" * len(header))
    for _, r in results:
        pf = "—" if r.profit_factor is None else f"{r.profit_factor}"
        print(f"{r.strategy:<34}{r.total_return_pct:>8}%{r.total_trades:>8}"
              f"{r.win_rate_pct:>8}{pf:>7}{r.max_drawdown_pct:>8}%")

    beat = [r.strategy for _, r in results if r.total_return_pct > benchmark]
    print(f"\nBeat buy & hold: {', '.join(beat) if beat else 'none'}")

    with open(args.out, "w") as f:
        json.dump(comparison_report(args.symbol, args.period, df, benchmark, results),
                  f, indent=2, allow_nan=False)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()