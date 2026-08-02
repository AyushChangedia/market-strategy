"""
Regression tests for the backtest engine.

These cover the three defects fixed alongside them, all of which were silent:
an undefined RSI in a strong rally, a position series that disagreed with the
strategy's own state, and a results file that no strict JSON parser could read.

Everything here is synthetic and offline — no network, no yfinance.
"""

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from backtest import (
    bollinger_signals,
    buy_and_hold,
    evaluate,
    extract_trades,
    rsi_series,
    rsi_signals,
    COST_PER_TRADE,
)


def frame(close) -> pd.DataFrame:
    """Build an OHLC frame from a close series, with sane highs and lows."""
    close = pd.Series(close, dtype="float64")
    return pd.DataFrame({"Close": close, "High": close * 1.01, "Low": close * 0.99})


def rally() -> pd.DataFrame:
    """A dip deep enough to trigger an oversold entry, then an unbroken rally."""
    return frame(np.concatenate([np.linspace(100, 90, 12), np.linspace(90, 140, 28)]))


class TestRsiDefinition:
    def test_no_nan_after_the_warmup_window(self):
        # The rally contains windows with no down-closes at all. Those used to
        # divide by a zero mapped to NaN, leaving RSI undefined for 15 bars.
        close = rally()["Close"]
        loss = (-close.diff().clip(upper=0)).rolling(14).mean()
        assert (loss == 0).sum() > 0, "fixture no longer exercises the zero-loss case"

        rsi = rsi_series(close)
        assert not rsi[14:].isna().any()

    def test_a_window_with_no_losses_is_maximally_overbought(self):
        # Straight line up: every close higher than the last.
        rsi = rsi_series(frame(np.linspace(100, 200, 40))["Close"])
        assert rsi.iloc[-1] == 100.0

    def test_a_flat_window_is_neither_overbought_nor_oversold(self):
        rsi = rsi_series(frame([100.0] * 40)["Close"])
        assert rsi.iloc[-1] == 50.0

    def test_a_vertical_rally_never_stays_long(self):
        # RSI pegged at 100 is above the exit threshold, so the position closes.
        assert rsi_signals(frame(np.linspace(100, 200, 40))).iloc[-1] == 0

    def test_a_flat_market_never_opens_a_position(self):
        assert rsi_signals(frame([100.0] * 40)).sum() == 0

    def test_the_overbought_exit_still_fires(self):
        df = rally()
        position = rsi_signals(df)
        assert position.max() == 1, "expected an oversold entry on the dip"
        assert position.iloc[-1] == 0, "expected an overbought exit in the rally"


class TestUndefinedIndicatorHoldsPosition:
    def test_a_nan_mid_series_does_not_close_the_position(self):
        # The strategy has no opinion while its band is undefined, so the
        # position must persist rather than being silently flattened and
        # re-opened. A quiet stretch then a sharp drop opens a position first.
        close = np.concatenate([np.full(20, 100.0), np.full(20, 90.0)])
        df = frame(close)
        assert bollinger_signals(df).iloc[21] == 1, "fixture must open a position"

        gapped = df.copy()
        gapped.loc[gapped.index[25], "Close"] = np.nan
        position = bollinger_signals(gapped)

        undefined = gapped["Close"].rolling(20).mean().isna().to_numpy()
        mid = undefined.copy()
        mid[:21] = False   # ignore the warmup, where flat is correct
        assert mid.any(), "fixture no longer produces a mid-series NaN"

        # The position is open before the gap and unchanged across it.
        assert position.iloc[20] == 1
        assert set(position.to_numpy()[mid]) == {1}

    def test_warmup_bars_are_still_flat(self):
        df = rally()
        # Holding starts at 0, so preserving it leaves the warmup untouched.
        assert rsi_signals(df).iloc[0] == 0


class TestProfitFactor:
    def test_is_none_when_there_are_no_losing_trades(self):
        result = evaluate("all winners", [0.05, 0.03, 0.02])
        assert result.profit_factor is None

    def test_is_a_ratio_when_there_are_losses(self):
        result = evaluate("mixed", [0.05, -0.02, 0.01])
        assert result.profit_factor == 3.0

    def test_results_survive_a_strict_json_round_trip(self):
        # float("inf") serialises as the bare token Infinity, which json.dump
        # writes but no strict parser accepts.
        result = evaluate("all winners", [0.05, 0.03])
        blob = json.dumps({"results": [asdict(result)]}, allow_nan=False)
        assert '"profit_factor": null' in blob
        assert json.loads(blob)["results"][0]["profit_factor"] is None

    def test_a_non_finite_metric_fails_loudly(self):
        with pytest.raises(ValueError):
            json.dumps({"x": float("inf")}, allow_nan=False)


class TestTradeExtraction:
    def test_charges_costs_on_a_completed_round_trip(self):
        df = frame([100, 100, 110, 110])
        position = pd.Series([0, 1, 1, 0], index=df.index)
        trades = extract_trades(df, position)
        assert len(trades) == 1
        # Entered at 100, exited at 110, less the round-trip cost.
        assert trades[0] == pytest.approx(0.10 - COST_PER_TRADE)

    def test_closes_an_open_position_at_the_final_bar(self):
        df = frame([100, 100, 120])
        position = pd.Series([0, 1, 1], index=df.index)
        trades = extract_trades(df, position)
        assert len(trades) == 1
        assert trades[0] == pytest.approx(0.20 - COST_PER_TRADE)

    def test_a_strategy_that_never_fires_books_nothing(self):
        df = frame([100, 101, 102])
        assert extract_trades(df, pd.Series([0, 0, 0], index=df.index)) == []


class TestEvaluate:
    def test_an_empty_book_is_all_zeroes_not_nan(self):
        result = evaluate("never fired", [])
        assert result.total_trades == 0
        assert result.total_return_pct == 0.0
        assert result.profit_factor == 0.0

    def test_drawdown_is_negative_or_zero(self):
        result = evaluate("choppy", [0.10, -0.05, 0.02, -0.08])
        assert result.max_drawdown_pct <= 0

    def test_expectancy_is_the_mean_trade(self):
        result = evaluate("two trades", [0.10, -0.04])
        assert result.expectancy_pct == pytest.approx(3.0)


def test_buy_and_hold_measures_first_close_to_last():
    assert buy_and_hold(frame([100, 120, 110])) == 10.0
