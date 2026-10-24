"""
Regression tests for the backtest engine.

These cover the three defects fixed alongside them, all of which were silent:
an undefined RSI in a strong rally, a position series that disagreed with the
strategy's own state, and a results file that no strict JSON parser could read.

Everything here is synthetic and offline — no network, no yfinance.
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import backtest

from backtest import (
    INSTRUMENT_NAMES,
    TRADING_DAYS_PER_YEAR,
    trades_per_year,
    STRATEGIES,
    bollinger_signals,
    comparison_report,
    buy_and_hold,
    buy_and_hold_net,
    evaluate,
    extract_trades,
    fold_bounds,
    fold_robustness,
    rsi_series,
    rsi_signals,
    verdict_for,
    walk_forward,
    COST_PER_TRADE,
)


def dated(close) -> pd.DataFrame:
    """An OHLC frame on real trading dates, as walk_forward reports them."""
    df = frame(close)
    df.index = pd.bdate_range("2024-07-23", periods=len(df))
    return df


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


class TestFoldBounds:
    def test_blocks_are_consecutive_and_do_not_overlap(self):
        bounds = fold_bounds(300, n_splits=3, train_ratio=0.7)
        assert len(bounds) == 3
        for (tr_start, tr_end), (te_start, te_end) in bounds:
            assert tr_start < tr_end == te_start < te_end
        # each block begins where the previous one ended
        assert bounds[0][1][1] == bounds[1][0][0]
        assert bounds[1][1][1] == bounds[2][0][0]

    def test_reproduces_the_published_495_candle_layout(self):
        # results/walk_forward_rsi.json was built from 495 candles, 3 splits,
        # 0.7 train. These bounds are what its fold dates imply.
        assert fold_bounds(495, 3, 0.7) == [
            ((0, 115), (115, 165)),
            ((165, 280), (280, 330)),
            ((330, 445), (445, 495)),
        ]

    def test_never_evaluates_on_data_before_it_was_measured(self):
        # Test halves must always follow their own training half.
        for (_, tr_end), (te_start, _) in fold_bounds(400, 4, 0.6):
            assert te_start >= tr_end

    def test_rejects_a_series_too_short_to_split(self):
        with pytest.raises(ValueError):
            fold_bounds(4, n_splits=3)

    def test_rejects_a_nonsensical_train_ratio(self):
        with pytest.raises(ValueError):
            fold_bounds(300, train_ratio=0.0)
        with pytest.raises(ValueError):
            fold_bounds(300, train_ratio=1.0)


class TestFoldRobustness:
    def test_reports_the_share_of_return_that_survived(self):
        # Fold 1 of the published run: 8.18% in sample, 3.11% out.
        assert fold_robustness(8.18, 3.11) == 0.38

    def test_a_fold_that_earned_nothing_in_training_is_consistent(self):
        # Nothing to degrade from, so it does not count against the strategy.
        assert fold_robustness(0.0, 0.0) == 1.0

    def test_losing_everything_out_of_sample_scores_zero(self):
        assert fold_robustness(3.01, 0.0) == 0.0

    def test_is_clamped_to_one_when_the_test_half_did_better(self):
        assert fold_robustness(2.0, 8.0) == 1.0

    def test_a_losing_test_half_scores_zero(self):
        assert fold_robustness(5.0, -3.0) == 0.0


class TestVerdict:
    def test_a_strategy_that_held_up_is_robust(self):
        assert verdict_for(0.85).startswith("ROBUST")

    def test_the_published_score_reads_as_weak(self):
        # 0.46 is the committed robustness_score for RSI.
        assert verdict_for(0.46).startswith("WEAK")

    def test_a_collapse_is_fragile(self):
        assert verdict_for(0.1).startswith("FRAGILE")


class TestWalkForward:
    def test_reports_one_entry_per_fold(self):
        report = walk_forward(dated(np.linspace(100, 130, 300)), "rsi", n_splits=3)
        assert len(report["folds"]) == 3
        assert [f["fold"] for f in report["folds"]] == [1, 2, 3]

    def test_fold_dates_run_forward_without_gaps_in_order(self):
        report = walk_forward(dated(np.linspace(100, 130, 300)), "rsi", n_splits=3)
        for f in report["folds"]:
            assert f["train_from"] <= f["train_to"] < f["test_from"] <= f["test_to"]

    def test_averages_match_the_folds_it_reported(self):
        report = walk_forward(dated(np.linspace(100, 130, 300)), "rsi", n_splits=3)
        folds = report["folds"]
        assert report["avg_train_return_pct"] == pytest.approx(
            np.mean([f["train_return_pct"] for f in folds]), abs=0.01)
        assert report["robustness_score"] == pytest.approx(
            np.mean([f["fold_robustness_score"] for f in folds]), abs=0.01)

    def test_the_verdict_follows_the_score(self):
        report = walk_forward(dated(np.linspace(100, 130, 300)), "rsi", n_splits=3)
        assert report["verdict"] == verdict_for(report["robustness_score"])

    def test_the_report_is_strict_json(self):
        report = walk_forward(dated(np.linspace(100, 130, 300)), "rsi", n_splits=3)
        assert json.loads(json.dumps(report, allow_nan=False))["strategy"] == "rsi"

    def test_rejects_an_unknown_strategy(self):
        with pytest.raises(KeyError):
            walk_forward(dated(np.linspace(100, 130, 300)), "not_a_strategy")

    def test_works_for_every_registered_strategy(self):
        from backtest import STRATEGIES
        df = dated(np.linspace(100, 130, 300))
        for key in STRATEGIES:
            assert walk_forward(df, key, n_splits=3)["strategy"] == key


# The keys make_charts.py reads out of results/comparison.json. If this list
# and the writer ever disagree again, chart generation dies with a KeyError
# rather than a useful message — which is exactly what happened once already.
CHART_TOP_LEVEL = ("ranking", "buy_and_hold_return_pct")
CHART_ROW_KEYS = ("strategy", "label", "total_return_pct", "total_trades",
                  "win_rate_pct", "profit_factor")


def report_for(days: int = 300) -> dict:
    df = dated(np.linspace(100, 130, days))
    ranked = []
    for key, (label, fn) in STRATEGIES.items():
        ranked.append((key, evaluate(label, extract_trades(df, fn(df)))))
    ranked.sort(key=lambda pair: pair[1].total_return_pct, reverse=True)
    return comparison_report("^NSEBANK", "2y", df, buy_and_hold(df), ranked)


class TestComparisonReport:
    def test_carries_every_key_the_charts_read(self):
        report = report_for()
        for key in CHART_TOP_LEVEL:
            assert key in report, f"make_charts.py reads {key!r}"

    def test_every_ranking_row_carries_the_keys_the_charts_read(self):
        for row in report_for()["ranking"]:
            for key in CHART_ROW_KEYS:
                assert key in row, f"make_charts.py reads row[{key!r}]"

    def test_identifies_strategies_by_id_not_display_label(self):
        # The charts map row["strategy"] through SHORT_NAMES, which is keyed by
        # id. Writing the label here is what broke chart generation before.
        ids = {row["strategy"] for row in report_for()["ranking"]}
        assert ids == set(STRATEGIES)

    def test_labels_match_the_strategy_registry(self):
        for row in report_for()["ranking"]:
            assert row["label"] == STRATEGIES[row["strategy"]][0]

    def test_covers_every_registered_strategy_once(self):
        rows = report_for()["ranking"]
        assert len(rows) == len(STRATEGIES)

    def test_ranks_run_from_one_in_descending_return_order(self):
        rows = report_for()["ranking"]
        assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
        returns = [r["total_return_pct"] for r in rows]
        assert returns == sorted(returns, reverse=True)

    def test_records_the_run_metadata(self):
        report = report_for()
        assert report["symbol"] == "^NSEBANK"
        assert report["instrument"] == INSTRUMENT_NAMES["^NSEBANK"]
        assert report["candles_analyzed"] == 300
        assert report["date_from"] <= report["date_to"]

    def test_falls_back_to_the_ticker_for_an_unmapped_symbol(self):
        df = dated(np.linspace(100, 130, 300))
        report = comparison_report("RELIANCE.NS", "2y", df, 1.0, [])
        assert report["instrument"] == "RELIANCE.NS"

    def test_states_the_cost_rates_actually_charged(self):
        report = report_for()
        assert report["commission_pct"] == pytest.approx(0.1)
        assert report["slippage_pct"] == pytest.approx(0.05)

    def test_is_strict_json(self):
        # An undefined profit factor must serialise as null, not Infinity.
        blob = json.dumps(report_for(), allow_nan=False)
        assert json.loads(blob)["ranking"][0]["rank"] == 1


class TestCommittedResults:
    """The published file the README's tables and charts are built from."""

    def test_satisfies_the_same_contract_as_a_fresh_run(self):
        path = Path(__file__).resolve().parents[1] / "results" / "comparison.json"
        committed = json.loads(path.read_text())
        for key in CHART_TOP_LEVEL:
            assert key in committed
        for row in committed["ranking"]:
            for key in CHART_ROW_KEYS:
                assert key in row

    def test_names_only_strategies_that_still_exist(self):
        path = Path(__file__).resolve().parents[1] / "results" / "comparison.json"
        committed = json.loads(path.read_text())
        for row in committed["ranking"]:
            assert row["strategy"] in STRATEGIES


class TestScratchTrades:
    def test_a_flat_trade_is_not_counted_as_a_loss(self):
        result = evaluate("flat", [0.05, 0.0, -0.02])
        assert result.scratch_trades == 1
        # One win out of three; the scratch is neither side.
        assert result.win_rate_pct == pytest.approx(33.3, abs=0.1)

    def test_a_book_of_scratches_has_no_wins_or_losses(self):
        result = evaluate("all flat", [0.0, 0.0])
        assert result.scratch_trades == 2
        assert result.win_rate_pct == 0.0

    def test_scratches_do_not_change_gross_loss(self):
        with_scratch = evaluate("a", [0.05, 0.0, -0.02])
        without = evaluate("b", [0.05, -0.02])
        assert with_scratch.profit_factor == without.profit_factor


class TestSharpeRatio:
    def test_is_positive_for_a_consistently_profitable_book(self):
        assert evaluate("good", [0.04, 0.05, 0.03, 0.06]).sharpe_ratio > 0

    def test_is_negative_for_a_losing_book(self):
        assert evaluate("bad", [-0.04, -0.05, -0.03, -0.06]).sharpe_ratio < 0

    def test_rewards_consistency_over_raw_return(self):
        # Small but non-zero dispersion — identical returns are handled by the
        # zero-variance case below, not here.
        steady = evaluate("steady", [0.030, 0.028, 0.032, 0.029, 0.031])
        lumpy = evaluate("lumpy", [0.20, -0.10, 0.15, -0.08, 0.05])
        assert steady.sharpe_ratio > lumpy.sharpe_ratio

    def test_is_zero_when_there_is_nothing_to_measure(self):
        assert evaluate("empty", []).sharpe_ratio == 0.0
        assert evaluate("one", [0.05]).sharpe_ratio == 0.0

    def test_is_finite_for_identical_returns(self):
        # Zero dispersion would divide by zero.
        assert evaluate("identical", [0.02, 0.02, 0.02]).sharpe_ratio == 0.0


class TestNetBenchmark:
    def test_charges_one_round_trip_against_the_gross_figure(self):
        df = frame([100.0, 109.3])
        assert buy_and_hold(df) == 9.3
        assert buy_and_hold_net(df) == pytest.approx(9.3 - COST_PER_TRADE * 100, abs=0.01)

    def test_is_always_the_lower_of_the_two(self):
        for closes in ([100.0, 130.0], [100.0, 90.0], [100.0, 100.0]):
            df = frame(closes)
            assert buy_and_hold_net(df) < buy_and_hold(df)

    def test_appears_in_the_results_file(self):
        assert "buy_and_hold_net_return_pct" in report_for()


class TestFillMode:
    def _frame_and_position(self):
        df = frame([100.0, 100.0, 110.0, 120.0, 120.0])
        return df, pd.Series([0, 1, 1, 0, 0], index=df.index)

    def test_close_fill_uses_the_signal_bar(self):
        df, pos = self._frame_and_position()
        # Entered at bar 1 (100), exited at bar 3 (120).
        assert extract_trades(df, pos)[0] == pytest.approx(0.20 - COST_PER_TRADE)

    def test_next_bar_fill_uses_the_following_close(self):
        df, pos = self._frame_and_position()
        # Entry slips to bar 2 (110); the exit bar is the last actionable one.
        assert extract_trades(df, pos, fill="next-bar")[0] == pytest.approx(
            (120 - 110) / 110 - COST_PER_TRADE)

    def test_default_is_unchanged(self):
        df, pos = self._frame_and_position()
        assert extract_trades(df, pos) == extract_trades(df, pos, fill="close")

    def test_never_reads_past_the_last_bar(self):
        df = frame([100.0, 100.0, 110.0])
        pos = pd.Series([0, 1, 1], index=df.index)
        assert len(extract_trades(df, pos, fill="next-bar")) == 1

    def test_rejects_an_unknown_fill(self):
        df, pos = self._frame_and_position()
        with pytest.raises(ValueError):
            extract_trades(df, pos, fill="tomorrow")


# ---------------------------------------------------------------------------
# Maximum drawdown
# ---------------------------------------------------------------------------

def test_drawdown_counts_a_losing_first_trade():
    """
    The regression. The equity curve started at the value *after* the first
    trade, so the running peak did too — a strategy whose opening trade lost
    20% reported no drawdown for it at all, which is precisely the stretch a
    reader is looking for.
    """
    assert evaluate("x", [-0.20]).max_drawdown_pct == -20.0


def test_drawdown_of_an_opening_loss_that_is_never_recovered():
    result = evaluate("x", [-0.20, 0.0, 0.0])
    assert result.max_drawdown_pct == -20.0


def test_drawdown_is_measured_from_the_running_peak():
    # Up 30% from 0.9 is a new high, so the 25% that follows is measured
    # against 1.17 rather than against the starting capital.
    result = evaluate("x", [-0.10, 0.30, -0.25])
    assert result.max_drawdown_pct == -25.0


def test_a_strategy_that_never_loses_has_no_drawdown():
    assert evaluate("x", [0.05, 0.05, 0.05]).max_drawdown_pct == 0.0


def test_drawdown_is_the_worst_stretch_not_the_last_one():
    result = evaluate("x", [-0.40, 2.0, -0.10])
    assert result.max_drawdown_pct == -40.0


def test_drawdown_compounds_across_consecutive_losses():
    # Two 10% losses are 19%, not 20%.
    result = evaluate("x", [-0.10, -0.10])
    assert result.max_drawdown_pct == pytest.approx(-19.0, abs=0.01)


def test_drawdown_is_never_positive():
    for trades in ([0.1], [0.1, 0.2], [0.0], [-0.5, 1.0, 1.0]):
        assert evaluate("x", trades).max_drawdown_pct <= 0.0


def test_total_return_is_unchanged_by_the_leading_point():
    # The extra 1.0 is there for the peak, not for the arithmetic.
    result = evaluate("x", [0.10, 0.10])
    assert result.total_return_pct == pytest.approx(21.0, abs=0.01)


# ---------------------------------------------------------------------------
# Sharpe ratio
# ---------------------------------------------------------------------------

def test_sharpe_is_scaled_by_a_year_of_trades():
    # One trade a week for a year, so about 52 of them.
    assert trades_per_year(52, TRADING_DAYS_PER_YEAR) == pytest.approx(52.0)


def test_trades_per_year_accounts_for_a_longer_history():
    # The same 52 trades spread over two years is 26 a year.
    assert trades_per_year(52, TRADING_DAYS_PER_YEAR * 2) == pytest.approx(26.0)


def test_an_unknown_span_leaves_the_ratio_unannualised():
    # Honestly per-trade, rather than multiplied by something meaningless.
    for span in (None, 0, -10):
        assert trades_per_year(100, span) == 1.0


def test_trading_more_often_does_not_by_itself_raise_sharpe():
    """
    The regression. The ratio was scaled by sqrt(number of trades), which is
    the sample size and not a span — so a strategy that traded more scored
    higher whatever the trades were worth.
    """
    rng = np.random.default_rng(7)
    span = TRADING_DAYS_PER_YEAR * 2

    strong = list(rng.normal(0.02, 0.05, 20))
    weak = list(rng.normal(0.004, 0.05, 400))

    assert (
        evaluate("few", strong, n_candles=span).sharpe_ratio
        > evaluate("many", weak, n_candles=span).sharpe_ratio
    )


def test_the_same_edge_over_a_longer_span_scores_lower():
    # Ten trades in one year is a better engine than ten in four.
    trades = [0.03, -0.01] * 5
    quick = evaluate("quick", trades, n_candles=TRADING_DAYS_PER_YEAR)
    slow = evaluate("slow", trades, n_candles=TRADING_DAYS_PER_YEAR * 4)
    assert quick.sharpe_ratio > slow.sharpe_ratio


def test_a_losing_edge_stays_negative_however_it_is_scaled():
    result = evaluate("x", [-0.02, -0.01, -0.03, 0.01], n_candles=TRADING_DAYS_PER_YEAR)
    assert result.sharpe_ratio < 0


def test_sharpe_is_zero_when_there_is_nothing_to_measure():
    assert evaluate("x", [], n_candles=252).sharpe_ratio == 0.0
    assert evaluate("x", [0.05], n_candles=252).sharpe_ratio == 0.0


def test_sharpe_is_zero_when_every_trade_was_identical():
    # No dispersion means no ratio, not a division by zero.
    assert evaluate("x", [0.02, 0.02, 0.02], n_candles=252).sharpe_ratio == 0.0


# ---------------------------------------------------------------------------
# The walk-forward benchmark
# ---------------------------------------------------------------------------

def _trending_frame(n: int = 600, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = pd.Series(
        100 * np.cumprod(1 + rng.normal(0.0004, 0.01, n)),
        index=pd.bdate_range("2023-01-02", periods=n),
    )
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close}
    )


def test_walk_forward_reports_the_benchmark_net_as_well_as_gross():
    """
    Every fold return is net of costs. The benchmark it was being read against
    was gross, which is the same mismatch the headline report already fixed —
    left standing in the one place the robustness verdict comes from.
    """
    report = backtest.walk_forward(_trending_frame(), "rsi")
    assert "buy_and_hold_net_return_pct" in report
    assert report["buy_and_hold_net_return_pct"] < report["buy_and_hold_return_pct"]


def test_the_net_benchmark_is_lower_by_exactly_one_round_trip():
    df = _trending_frame()
    report = backtest.walk_forward(df, "rsi")
    gap = report["buy_and_hold_return_pct"] - report["buy_and_hold_net_return_pct"]
    assert gap == pytest.approx(backtest.COST_PER_TRADE * 100, abs=0.01)


def test_walk_forward_says_whether_it_beat_the_benchmark():
    report = backtest.walk_forward(_trending_frame(), "rsi")
    assert report["oos_beat_benchmark"] == (
        report["oos_total_return_pct"] > report["buy_and_hold_net_return_pct"]
    )


def test_the_verdict_is_judged_against_the_net_benchmark():
    # Not the gross one. In a strong uptrend the difference is small, but a
    # net number read against a gross one is not a fair test at any size.
    report = backtest.walk_forward(_trending_frame(), "rsi")
    assert isinstance(report["oos_beat_benchmark"], bool)


def test_the_walk_forward_report_is_still_strict_json():
    # numpy scalars and NaN both serialise in ways a strict parser rejects.
    report = backtest.walk_forward(_trending_frame(), "rsi")
    json.loads(json.dumps(report, allow_nan=False))


# ---------------------------------------------------------------------------
# Exposure
# ---------------------------------------------------------------------------

def test_exposure_is_the_share_of_bars_held():
    assert backtest.exposure(pd.Series([0, 1, 1, 0])) == 50.0
    assert backtest.exposure(pd.Series([1, 1, 1, 1])) == 100.0
    assert backtest.exposure(pd.Series([0, 0, 0, 0])) == 0.0


def test_exposure_of_nothing_is_nothing_rather_than_a_division_by_zero():
    assert backtest.exposure(pd.Series([], dtype=float)) == 0.0
    assert backtest.exposure(None) == 0.0


def test_exposure_is_rounded_the_way_it_is_printed():
    assert backtest.exposure(pd.Series([1, 0, 0])) == 33.3


def test_two_strategies_with_the_same_return_are_not_the_same_strategy():
    """
    The missing half of every return in this report. One of these earned its
    8% while holding a tenth of the time; the other was always in. Compared on
    return alone the report silently prefers whichever was simply exposed for
    longer.
    """
    rare = evaluate("rare", [0.08], position=pd.Series([1] + [0] * 9))
    always = evaluate("always", [0.08], position=pd.Series([1] * 10))

    assert rare.total_return_pct == always.total_return_pct
    assert rare.exposure_pct == 10.0
    assert always.exposure_pct == 100.0


def test_exposure_is_reported_even_when_nothing_was_traded():
    # A signal series that never fires still has an exposure, and it is zero.
    result = evaluate("x", [], position=pd.Series([0, 0, 0]))
    assert result.total_trades == 0
    assert result.exposure_pct == 0.0


def test_exposure_defaults_to_zero_when_no_position_is_given():
    # evaluate is called from tests and from segment code without one; it
    # must not invent a figure it was not told.
    assert evaluate("x", [0.05]).exposure_pct == 0.0


def test_exposure_is_on_the_serialised_result():
    result = evaluate("x", [0.05], position=pd.Series([1, 1, 0, 0]))
    assert json.loads(json.dumps(asdict(result), allow_nan=False))["exposure_pct"] == 50.0


# ---------------------------------------------------------------------------
# CAGR
# ---------------------------------------------------------------------------

def test_a_years_return_is_its_own_annual_rate():
    assert backtest.cagr(40, TRADING_DAYS_PER_YEAR) == pytest.approx(40.0, abs=0.01)


def test_the_same_return_over_five_years_is_a_much_duller_one():
    """
    40% over one year and 40% over five were the same cell in the table, and
    this project runs 1y, 2y and 5y periods from the same command.
    """
    assert backtest.cagr(40, TRADING_DAYS_PER_YEAR * 5) == pytest.approx(6.96, abs=0.01)


def test_cagr_compounds_rather_than_dividing():
    # 21% over two years is 10% a year, not 10.5%.
    assert backtest.cagr(21, TRADING_DAYS_PER_YEAR * 2) == pytest.approx(10.0, abs=0.01)


def test_a_loss_annualises_to_a_loss():
    assert backtest.cagr(-19, TRADING_DAYS_PER_YEAR * 2) == pytest.approx(-10.0, abs=0.01)


def test_losing_everything_has_no_annual_rate():
    # The root of a negative number is not a figure to print.
    assert backtest.cagr(-100, TRADING_DAYS_PER_YEAR) == 0.0
    assert backtest.cagr(-150, TRADING_DAYS_PER_YEAR) == 0.0


def test_an_unknown_span_gives_no_rate_rather_than_a_wrong_one():
    for span in (None, 0, -5):
        assert backtest.cagr(40, span) == 0.0


def test_evaluate_carries_the_annual_rate():
    result = evaluate("x", [0.21], n_candles=TRADING_DAYS_PER_YEAR * 2)
    assert result.total_return_pct == pytest.approx(21.0, abs=0.01)
    assert result.cagr_pct == pytest.approx(10.0, abs=0.01)


def test_cagr_is_zero_when_there_were_no_trades():
    assert evaluate("x", [], n_candles=TRADING_DAYS_PER_YEAR).cagr_pct == 0.0


def test_cagr_is_on_the_serialised_result():
    result = evaluate("x", [0.21], n_candles=TRADING_DAYS_PER_YEAR * 2)
    assert "cagr_pct" in json.loads(json.dumps(asdict(result), allow_nan=False))


# ---------------------------------------------------------------------------
# The longest losing streak
# ---------------------------------------------------------------------------

def test_the_longest_run_of_losses_is_counted():
    assert backtest.max_losing_streak([-0.01, -0.01, 0.05, -0.01]) == 2


def test_the_longest_run_not_the_last_one():
    assert backtest.max_losing_streak([-0.01] * 5 + [0.10] + [-0.01] * 2) == 5


def test_a_winner_breaks_the_run():
    assert backtest.max_losing_streak([-0.01, 0.01, -0.01, 0.01, -0.01]) == 1


def test_a_scratch_trade_breaks_the_run_too():
    # It cost nothing. A flat result is not another loss to sit through.
    assert backtest.max_losing_streak([-0.01, 0.0, -0.01]) == 1


def test_a_strategy_that_never_loses_has_no_streak():
    assert backtest.max_losing_streak([0.01, 0.02, 0.0]) == 0
    assert backtest.max_losing_streak([]) == 0
    assert backtest.max_losing_streak(None) == 0


def test_every_trade_losing_is_the_whole_run():
    assert backtest.max_losing_streak([-0.01] * 9) == 9


def test_the_streak_is_on_the_result():
    """
    The figure that decides whether a strategy is survivable in practice
    rather than on paper. Drawdown says how deep the hole was; this says how
    many losses in a row somebody had to keep taking the next signal through.
    """
    result = evaluate("x", [-0.02, -0.02, -0.02, 0.10], n_candles=TRADING_DAYS_PER_YEAR)
    assert result.max_losing_streak == 3


def test_the_streak_and_the_loss_count_agree_when_all_losses_are_consecutive():
    trades = [-0.01, -0.02, -0.03, 0.05]
    result = evaluate("x", trades, n_candles=TRADING_DAYS_PER_YEAR)
    losses = sum(1 for t in trades if t < 0)
    assert result.max_losing_streak == losses


# ---------------------------------------------------------------------------
# The cost assumption
# ---------------------------------------------------------------------------

def _two_trade_frame() -> tuple:
    close = pd.Series([100.0, 100.0, 110.0, 110.0, 100.0, 100.0])
    df = pd.DataFrame({"Close": close})
    position = pd.Series([0, 1, 1, 0, 0, 0])
    return df, position


def test_the_default_cost_is_the_projects_own_assumption():
    df, position = _two_trade_frame()
    [trade] = backtest.extract_trades(df, position)
    assert trade == pytest.approx(0.10 - backtest.COST_PER_TRADE, abs=1e-9)


def test_a_higher_cost_eats_more_of_the_trade():
    """
    0.15% is reasonable for a liquid Indian index future and wrong for almost
    anything else. A reader whose broker charges differently — or who wants to
    know how much of an edge is really just this assumption — can say so.
    """
    df, position = _two_trade_frame()
    [trade] = backtest.extract_trades(df, position, cost=0.01)
    assert trade == pytest.approx(0.10 - 0.01, abs=1e-9)


def test_a_cost_of_zero_is_the_gross_trade():
    df, position = _two_trade_frame()
    [trade] = backtest.extract_trades(df, position, cost=0.0)
    assert trade == pytest.approx(0.10, abs=1e-9)


def test_a_negative_cost_is_refused_rather_than_paying_you_to_trade():
    df, position = _two_trade_frame()
    with pytest.raises(ValueError, match="negative"):
        backtest.extract_trades(df, position, cost=-0.01)


def test_the_cost_applies_to_an_open_position_closed_at_the_end():
    df = pd.DataFrame({"Close": pd.Series([100.0, 100.0, 110.0])})
    position = pd.Series([0, 1, 1])
    [trade] = backtest.extract_trades(df, position, cost=0.02)
    assert trade == pytest.approx(0.10 - 0.02, abs=1e-9)


def test_the_benchmark_is_charged_the_same_cost():
    df = pd.DataFrame({"Close": pd.Series([100.0, 110.0])})
    assert backtest.buy_and_hold_net(df, cost=0.02) == pytest.approx(8.0, abs=0.01)
    assert backtest.buy_and_hold_net(df, cost=0.0) == pytest.approx(10.0, abs=0.01)


def test_walk_forward_charges_the_benchmark_what_it_was_told():
    df = _trending_frame()
    cheap = backtest.walk_forward(df, "rsi", cost=0.0)
    dear = backtest.walk_forward(df, "rsi", cost=0.05)
    assert dear["buy_and_hold_net_return_pct"] < cheap["buy_and_hold_net_return_pct"]


def test_a_bigger_cost_can_turn_a_winning_strategy_into_a_losing_one():
    # Which is the point of being able to set it.
    df, position = _two_trade_frame()
    assert backtest.extract_trades(df, position, cost=0.0)[0] > 0
    assert backtest.extract_trades(df, position, cost=0.5)[0] < 0


# ---------------------------------------------------------------------------
# What the saved report says about itself
# ---------------------------------------------------------------------------

def _tiny_report(**kwargs) -> dict:
    df = pd.DataFrame(
        {"Close": pd.Series([100.0, 110.0], index=pd.bdate_range("2024-01-01", periods=2))}
    )
    ranked = [("rsi", evaluate("RSI", [0.05], n_candles=2))]
    return backtest.comparison_report("^NSEBANK", "1y", df, 10.0, ranked, **kwargs)


def test_the_report_states_the_cost_its_figures_were_charged():
    """
    The report documented the module's default whatever --cost said, so a run
    at a different assumption saved a file claiming numbers it had not
    produced. A saved result outlives the command that made it; the
    assumptions are the one thing it has to carry correctly.
    """
    assert _tiny_report(cost=0.004)["round_trip_cost_pct"] == pytest.approx(0.4)
    assert _tiny_report(cost=0.0)["round_trip_cost_pct"] == 0.0


def test_the_report_states_the_fill_assumption_too():
    # Also not recoverable from the numbers afterwards.
    assert _tiny_report(fill="next-bar")["fill"] == "next-bar"
    assert _tiny_report()["fill"] == "close"


def test_the_net_benchmark_follows_the_cost_it_was_given():
    assert _tiny_report(cost=0.0)["buy_and_hold_net_return_pct"] == pytest.approx(10.0)
    assert _tiny_report(cost=0.01)["buy_and_hold_net_return_pct"] == pytest.approx(9.0)


def test_the_defaults_are_unchanged_so_published_figures_still_match():
    report = _tiny_report()
    assert report["round_trip_cost_pct"] == pytest.approx(backtest.COST_PER_TRADE * 100)


def test_the_new_metrics_reach_the_saved_ranking():
    row = _tiny_report()["ranking"][0]
    for field in ("exposure_pct", "cagr_pct", "max_losing_streak", "sharpe_ratio"):
        assert field in row, field


def test_the_report_is_strict_json():
    json.loads(json.dumps(_tiny_report(), allow_nan=False))
