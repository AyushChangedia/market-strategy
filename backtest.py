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
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

COMMISSION_PCT = 0.001   # 0.1% per trade
SLIPPAGE_PCT = 0.0005    # 0.05% per trade
COST_PER_TRADE = COMMISSION_PCT + SLIPPAGE_PCT

# Daily bars, so a year of history is this many of them. Used to turn a
# per-trade statistic into a per-year one.
TRADING_DAYS_PER_YEAR = 252


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
    # Share of the history actually spent holding something. A strategy in the
    # market a tenth of the time earned its return with a tenth of the
    # exposure, which the return alone does not say.
    exposure_pct: float = 0.0
    # Annualised. A 40% total return means one thing over one year and
    # something much duller over five, and the table never said which.
    cagr_pct: float = 0.0
    # The run of losses a trader would have had to sit through. Max drawdown
    # says how deep the hole was; this says how long they were in it.
    max_losing_streak: int = 0


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

def extract_trades(df: pd.DataFrame, position: pd.Series,
                   fill: str = "close", cost: float = COST_PER_TRADE) -> list:
    """
    Walk the position series and return each completed trade's net return.

    `fill` decides which bar the order is assumed to execute on:

      "close"     the same bar that produced the signal. Every indicator here
                  is computed from that bar's close, so this assumes you saw
                  the close and still traded at it — a market-on-close order
                  gets you near it, but not reliably.
      "next-bar"  the following bar's close, which is the earliest price you
                  could actually have acted on.

    `cost` is what a round trip costs, as a fraction — commission plus
    slippage. It defaults to this project's own assumption, 0.15%, which is
    reasonable for a liquid Indian index future and wrong for almost anything
    else. A reader whose broker charges differently, or who wants to know how
    much of an edge is really just a cost assumption, can say so.

    The defaults are unchanged so published figures stay reproducible; pass
    "next-bar" to see how much of an edge survives a realistic fill.
    """
    if fill not in ("close", "next-bar"):
        raise ValueError(f"fill must be 'close' or 'next-bar', got {fill!r}")

    if cost < 0:
        raise ValueError(f"cost must not be negative, got {cost!r}")

    step = 1 if fill == "next-bar" else 0
    last = len(df) - 1
    price_at = lambda i: df["Close"].iloc[min(i + step, last)]

    trades = []
    entry_price = None

    for i in range(1, len(position)):
        was_flat = position.iloc[i - 1] == 0
        now_long = position.iloc[i] == 1

        if was_flat and now_long:
            entry_price = price_at(i)
        elif not was_flat and not now_long and entry_price is not None:
            exit_price = price_at(i)
            gross = (exit_price - entry_price) / entry_price
            trades.append(gross - cost)   # costs on every round trip
            entry_price = None

    # close any open position at the final bar
    if entry_price is not None:
        gross = (df["Close"].iloc[-1] - entry_price) / entry_price
        trades.append(gross - cost)

    return trades


def trades_per_year(n_trades: int, n_candles: Optional[int]) -> float:
    """
    How many trades this strategy takes in a year, at the pace observed.

    Returns 1 when the span is unknown or unusable, which leaves a Sharpe
    unannualised — a number that is honestly per-trade rather than one that
    has been multiplied by something meaningless.
    """
    if not n_candles or n_candles <= 0:
        return 1.0
    years = n_candles / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return 1.0
    return n_trades / years


def max_losing_streak(trades) -> int:
    """
    The longest run of consecutive losing trades.

    The figure that decides whether a strategy is survivable in practice
    rather than on paper. Maximum drawdown says how deep the hole was; this
    says how many losses in a row a person had to keep taking the next signal
    through. Nine is the point at which most people stop following the rules,
    whatever the backtest says happens afterwards.

    A scratch trade breaks the run. It cost nothing, and a flat result is not
    another loss to sit through.
    """
    longest = 0
    current = 0
    for trade in trades or []:
        if trade < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def cagr(total_return_pct: float, n_candles: Optional[int]) -> float:
    """
    The compound annual growth rate behind a total return.

    A total return cannot be compared across histories of different lengths,
    and this project runs 1y, 2y and 5y periods from the same command. 40%
    over one year and 40% over five are the same cell in the table and very
    different results.

    Returns 0 when the span is unknown, and when the strategy lost everything
    — a total return of -100% has no annual rate, and the root of a negative
    number is not a figure to print.
    """
    if not n_candles or n_candles <= 0:
        return 0.0
    years = n_candles / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return 0.0

    growth = 1 + total_return_pct / 100
    if growth <= 0:
        return 0.0
    return round((growth ** (1 / years) - 1) * 100, 2)


def exposure(position: pd.Series) -> float:
    """
    The share of the history spent holding something, as a percentage.

    The missing half of every return in this report. A strategy that returned
    8% while in the market a tenth of the time and one that returned 8% while
    always in it are not the same strategy — the first left nine tenths of the
    period free and took a tenth of the risk to get there. Comparing them on
    return alone silently prefers the one that was simply exposed for longer.
    """
    if position is None or len(position) == 0:
        return 0.0
    return round(float((position == 1).sum()) / len(position) * 100, 1)


def evaluate(name: str, trades: list, n_candles: Optional[int] = None,
             position: Optional[pd.Series] = None) -> Result:
    """
    Turn a list of trade returns into the metrics that matter.

    `n_candles` is how many daily bars the trades were taken over. It is what
    turns a per-trade Sharpe into a per-year one; without it the ratio is
    reported unannualised rather than annualised by the wrong thing.
    """
    held = 0.0 if position is None else exposure(position)

    if not trades:
        return Result(strategy=name, total_return_pct=0.0, win_rate_pct=0.0,
                      total_trades=0, scratch_trades=0, profit_factor=0.0,
                      sharpe_ratio=0.0, max_drawdown_pct=0.0,
                      expectancy_pct=0.0, exposure_pct=held, cagr_pct=0.0,
                      max_losing_streak=0)

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

    # The curve starts at 1, before the first trade. Without that leading
    # point the running peak starts at the equity *after* trade one, so a
    # strategy whose first trade loses 20% shows no drawdown for it at all —
    # the worst stretch of a losing run is the one the report skips.
    equity = np.concatenate(([1.0], np.cumprod(1 + arr)))
    peak = np.maximum.accumulate(equity)
    max_dd = ((equity - peak) / peak).min() * 100

    # Sharpe: mean return per trade against its own dispersion, scaled to a
    # year by how many trades a year actually contains.
    #
    # It was scaled by sqrt(number of trades), which is not annualisation —
    # it is the sample size. That makes the ratio grow with trade count
    # regardless of quality, so four hundred mediocre trades score higher than
    # twenty good ones, and a strategy looks better the more often it trades.
    # The fix needs to know how long the trades were taken over; with no span
    # given the figure is left unannualised rather than annualised by the
    # wrong quantity.
    sharpe = 0.0
    if len(arr) > 1 and arr.std(ddof=1) > 0:
        per_trade = float(arr.mean() / arr.std(ddof=1))
        sharpe = per_trade * np.sqrt(trades_per_year(len(arr), n_candles))

    total_return_pct = round((equity[-1] - 1) * 100, 2)

    return Result(
        strategy=name,
        total_return_pct=total_return_pct,
        win_rate_pct=round(len(wins) / len(arr) * 100, 1),
        scratch_trades=scratches,
        total_trades=len(arr),
        profit_factor=None if profit_factor is None else round(profit_factor, 2),
        sharpe_ratio=round(sharpe, 2),
        max_drawdown_pct=round(max_dd, 2),
        expectancy_pct=round(arr.mean() * 100, 2),
        exposure_pct=held,
        cagr_pct=cagr(total_return_pct, n_candles),
        max_losing_streak=max_losing_streak(trades),
    )


def buy_and_hold(df: pd.DataFrame) -> float:
    """The benchmark every strategy has to beat to be worth running."""
    return round((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100, 2)


def buy_and_hold_net(df: pd.DataFrame, cost: float = COST_PER_TRADE) -> float:
    """
    The same benchmark after paying to get in and out.

    Every strategy is charged COST_PER_TRADE on each round trip while the
    benchmark was quoted gross, which flattered it by exactly the cost of the
    one round trip it also has to make. The gap is small over two years and
    does not overturn the headline finding, but comparing a net number against
    a gross one is not a fair test, so both are reported.
    """
    gross = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1)
    return round((gross - cost) * 100, 2)


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
                 n_splits: int = 3, train_ratio: float = 0.7,
                 cost: float = COST_PER_TRADE) -> dict:
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
    oos_return = round(float(oos[-1] - 1) * 100, 2)
    net_benchmark = buy_and_hold_net(df, cost)

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
        "oos_total_return_pct": oos_return,
        # Net, because every fold return is net. The comparison the verdict
        # rests on was a strategy charged for each round trip against a
        # benchmark charged for none — the same mismatch the headline report
        # already fixed, left standing here.
        "buy_and_hold_return_pct": buy_and_hold(df),
        "buy_and_hold_net_return_pct": net_benchmark,
        # The question the whole exercise exists to answer, against the
        # benchmark charged the same way the strategy was.
        "oos_beat_benchmark": bool(oos_return > net_benchmark),
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


def walk_forward_path(out: str, strategy: str) -> str:
    """
    Where a walk-forward report goes, given the --out meant for the comparison.

    This was `out.replace("comparison_local", ...)`, which only renames the
    one filename the project happens to ship with. Any other --out matched
    nothing, the replace returned the string unchanged, and the walk-forward
    report was written straight over the comparison results the same flag
    names — silently, and with no way to tell afterwards which run produced
    the file.

    The name is now built from the path rather than found inside it, so the
    two can never land on top of each other.
    """
    path = Path(out)
    stem = path.stem
    # Keep the project's own naming when it is the project's own file, so
    # committed results and the chart generator still line up.
    if "comparison" in stem:
        stem = stem.replace("comparison", f"walk_forward_{strategy}")
    else:
        stem = f"{stem}_walk_forward_{strategy}"
    return str(path.with_name(stem + (path.suffix or ".json")))


def comparison_report(symbol: str, period: str, df: pd.DataFrame,
                      benchmark: float, ranked: list,
                      net_benchmark: float | None = None,
                      cost: float = COST_PER_TRADE,
                      fill: str = "close") -> dict:
    """
    Build the results document in the shape make_charts.py reads.

    The two had drifted: this wrote {"results": [...]} keyed by display label,
    while the charts read {"ranking": [...]} keyed by strategy id. Regenerating
    the results therefore produced a file that crashed chart generation, and
    only the committed copy — written by an earlier version of this script —
    still worked.
    """
    if net_benchmark is None:
        net_benchmark = round(benchmark - cost * 100, 2)

    return {
        "symbol": symbol,
        "instrument": INSTRUMENT_NAMES.get(symbol, symbol),
        "period": period,
        "interval": "1d",
        "candles_analyzed": len(df),
        "date_from": _day(df, 0),
        "date_to": _day(df, len(df) - 1),
        # What the figures below were actually charged, not what the module
        # defaults to. With --cost set, the report used to document an
        # assumption none of its numbers were computed under — the one thing
        # a saved result has to get right, because it outlives the command
        # that produced it.
        "round_trip_cost_pct": round(cost * 100, 4),
        "commission_pct": round(COMMISSION_PCT * 100, 4),
        "slippage_pct": round(SLIPPAGE_PCT * 100, 4),
        # Which bar the orders were assumed to fill on. Also not recoverable
        # from the numbers afterwards.
        "fill": fill,
        "buy_and_hold_return_pct": benchmark,
        "buy_and_hold_net_return_pct": net_benchmark,
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
    parser.add_argument("--cost", type=float, default=COST_PER_TRADE * 100,
                        metavar="PCT",
                        help="round-trip cost in percent (default: %(default)s, "
                             "which is 0.1%% commission plus 0.05%% slippage)")
    parser.add_argument("--fill", choices=("close", "next-bar"), default="close",
                        help="bar the order fills on (default: close)")
    parser.add_argument("--strategy", metavar="NAME", action="append",
                        help="run only this strategy; repeatable")
    parser.add_argument("--list-strategies", action="store_true",
                        help="print the registered strategy ids and exit")
    parser.add_argument("--walk-forward", metavar="STRATEGY",
                        help="validate one strategy fold by fold, e.g. rsi")
    parser.add_argument("--splits", type=int, default=3,
                        help="walk-forward folds (default 3)")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="share of each fold used in sample (default 0.7)")
    args = parser.parse_args()

    if args.list_strategies:
        for key, (label, _) in STRATEGIES.items():
            print(f"{key:<12}{label}")
        return

    chosen = args.strategy or list(STRATEGIES)
    unknown = [k for k in chosen if k not in STRATEGIES]
    if unknown:
        parser.error(f"unknown strategy {unknown[0]!r}; "
                     f"pick from {', '.join(STRATEGIES)}")

    df = load_prices(args.symbol, args.period)

    # Resolved before either branch: --walk-forward reads it too, and having
    # it below that branch made the flag crash outright.
    cost = args.cost / 100
    if cost < 0:
        parser.error("--cost must not be negative")

    if args.walk_forward:
        report = walk_forward(df, args.walk_forward, args.splits,
                              args.train_ratio, cost=cost)
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
        print(f"\nOut-of-sample {report['oos_total_return_pct']}%  ·  "
              f"buy & hold {report['buy_and_hold_net_return_pct']}% after costs  ·  "
              f"{'ahead' if report['oos_beat_benchmark'] else 'behind'}")
        print(f"Robustness {report['robustness_score']}  ·  {report['verdict']}")

        out = walk_forward_path(args.out, args.walk_forward)
        with open(out, "w") as f:
            json.dump(report, f, indent=2, allow_nan=False)
        print(f"Saved to {out}")
        return

    benchmark = buy_and_hold(df)
    net_benchmark = buy_and_hold_net(df, cost)

    print(f"\n{args.symbol}  ·  {len(df)} daily candles  ·  "
          f"{df.index[0].date()} to {df.index[-1].date()}")
    print(f"Buy & hold: {benchmark}%  ({net_benchmark}% after "
          f"{args.cost}% round-trip costs)\n")

    results = []
    for key in chosen:
        label, signal_fn = STRATEGIES[key]
        position = signal_fn(df)
        trades = extract_trades(df, position, fill=args.fill, cost=cost)
        results.append((key, evaluate(label, trades, n_candles=len(df),
                                      position=position)))

    results.sort(key=lambda pair: pair[1].total_return_pct, reverse=True)

    header = (f"{'Strategy':<34}{'Return':>9}{'Trades':>8}{'Win%':>8}"
              f"{'PF':>7}{'MaxDD':>9}{'Held':>8}{'CAGR':>9}")
    print(header)
    print("-" * len(header))
    for _, r in results:
        pf = "—" if r.profit_factor is None else f"{r.profit_factor}"
        print(f"{r.strategy:<34}{r.total_return_pct:>8}%{r.total_trades:>8}"
              f"{r.win_rate_pct:>8}{pf:>7}{r.max_drawdown_pct:>8}%"
              f"{r.exposure_pct:>7}%{r.cagr_pct:>8}%")

    beat = [r.strategy for _, r in results if r.total_return_pct > net_benchmark]
    print(f"\nBeat buy & hold: {', '.join(beat) if beat else 'none'}")

    with open(args.out, "w") as f:
        json.dump(comparison_report(args.symbol, args.period, df, benchmark,
                                    results, net_benchmark, cost=cost,
                                    fill=args.fill),
                  f, indent=2, allow_nan=False)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()