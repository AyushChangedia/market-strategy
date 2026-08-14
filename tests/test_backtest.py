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

from backtest import (
    INSTRUMENT_NAMES,
    STRATEGIES,
    bollinger_signals,
    comparison_report,
    buy_and_hold,
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
