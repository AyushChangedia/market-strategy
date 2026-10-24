"""
The command line, run the way a reader runs it.

Everything else here imports functions. This runs backtest.py as a program,
against a price file on disk, because the parts that only exist in main() —
argument parsing, the cost conversion, which report gets written where — have
never been exercised at all, and every one of them fails silently: a bad flag
produces a file, just not the one you asked for.

No network. The frames are synthetic and load_prices is replaced.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def prices(tmp_path) -> Path:
    """A believable two-year daily series, saved where the stub can find it."""
    rng = np.random.default_rng(11)
    n = 520
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.011, n))
    frame = pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.008,
            "Low": close * 0.992,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=pd.bdate_range("2023-01-02", periods=n),
    )
    path = tmp_path / "prices.csv"
    frame.to_csv(path)
    return path


def run(prices: Path, tmp_path: Path, *args) -> dict:
    """Run the CLI with load_prices stubbed, and return the saved report."""
    out = tmp_path / "report.json"
    script = f"""
import sys, pandas as pd
sys.path.insert(0, {str(ROOT)!r})
import backtest

def fake_load(symbol, period):
    return pd.read_csv({str(prices)!r}, index_col=0, parse_dates=True)

backtest.load_prices = fake_load
sys.argv = ["backtest.py", "--out", {str(out)!r}] + {list(args)!r}
backtest.main()
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=tmp_path
    )
    assert result.returncode == 0, result.stderr
    run.stdout = result.stdout
    if out.exists():
        return json.loads(out.read_text())
    return {}


def test_a_plain_run_writes_a_report(prices, tmp_path):
    report = run(prices, tmp_path)
    assert report["candles_analyzed"] == 520
    assert len(report["ranking"]) >= 1


def test_the_report_is_strict_json_on_disk(prices, tmp_path):
    # Written with allow_nan=False, but a NaN reaching a field would only
    # surface here, at the moment the file is produced.
    run(prices, tmp_path)
    text = (tmp_path / "report.json").read_text()
    assert "NaN" not in text
    assert "Infinity" not in text
    json.loads(text)


def test_the_cost_flag_reaches_the_figures(prices, tmp_path):
    free = run(prices, tmp_path, "--cost", "0")
    dear = run(prices, tmp_path, "--cost", "1")

    assert free["round_trip_cost_pct"] == 0.0
    assert dear["round_trip_cost_pct"] == 1.0
    assert dear["buy_and_hold_net_return_pct"] < free["buy_and_hold_net_return_pct"]


def test_a_costlier_run_cannot_report_a_better_strategy(prices, tmp_path):
    free = run(prices, tmp_path, "--cost", "0")
    dear = run(prices, tmp_path, "--cost", "1")

    by_key = {row["strategy"]: row["total_return_pct"] for row in free["ranking"]}
    for row in dear["ranking"]:
        assert row["total_return_pct"] <= by_key[row["strategy"]] + 1e-9, row["strategy"]


def test_a_negative_cost_is_refused_by_the_parser(prices, tmp_path):
    out = tmp_path / "report.json"
    script = f"""
import sys, pandas as pd
sys.path.insert(0, {str(ROOT)!r})
import backtest
backtest.load_prices = lambda s, p: pd.read_csv({str(prices)!r}, index_col=0, parse_dates=True)
sys.argv = ["backtest.py", "--out", {str(out)!r}, "--cost", "-1"]
backtest.main()
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode != 0
    assert "cost" in result.stderr.lower()


def test_the_fill_mode_is_recorded_and_changes_the_trades(prices, tmp_path):
    same_bar = run(prices, tmp_path, "--fill", "close")
    next_bar = run(prices, tmp_path, "--fill", "next-bar")

    assert same_bar["fill"] == "close"
    assert next_bar["fill"] == "next-bar"

    # The whole point of the flag: acting on a price you could actually have
    # traded at is not the same backtest.
    same = [row["total_return_pct"] for row in same_bar["ranking"]]
    later = [row["total_return_pct"] for row in next_bar["ranking"]]
    assert same != later


def test_selecting_one_strategy_reports_only_that_one(prices, tmp_path):
    report = run(prices, tmp_path, "--strategy", "rsi")
    assert [row["strategy"] for row in report["ranking"]] == ["rsi"]


def test_the_ranking_is_sorted_by_return(prices, tmp_path):
    returns = [row["total_return_pct"] for row in run(prices, tmp_path)["ranking"]]
    assert returns == sorted(returns, reverse=True)


def test_every_row_carries_every_metric(prices, tmp_path):
    expected = {
        "total_return_pct", "win_rate_pct", "total_trades", "scratch_trades",
        "profit_factor", "sharpe_ratio", "max_drawdown_pct", "expectancy_pct",
        "exposure_pct", "cagr_pct", "max_losing_streak",
    }
    for row in run(prices, tmp_path)["ranking"]:
        assert expected <= set(row), expected - set(row)


def test_exposure_is_a_real_share_of_the_history(prices, tmp_path):
    for row in run(prices, tmp_path)["ranking"]:
        assert 0.0 <= row["exposure_pct"] <= 100.0, row["strategy"]


def test_a_strategy_that_never_traded_reports_zeroes_not_nulls(prices, tmp_path):
    for row in run(prices, tmp_path)["ranking"]:
        if row["total_trades"] == 0:
            assert row["total_return_pct"] == 0.0
            assert row["max_drawdown_pct"] == 0.0
            assert row["max_losing_streak"] == 0


def test_walk_forward_writes_beside_the_comparison_not_over_it(prices, tmp_path):
    """
    The path was derived by string-replacing "comparison_local" in --out,
    which only renames the one filename this project happens to ship. Any
    other --out matched nothing, so the walk-forward report was written
    straight over the comparison results the same flag names — silently, with
    no way to tell afterwards which run produced the file.
    """
    comparison = run(prices, tmp_path)
    assert comparison["candles_analyzed"] == 520

    run(prices, tmp_path, "--walk-forward", "rsi")

    # The comparison report is untouched.
    still_there = json.loads((tmp_path / "report.json").read_text())
    assert "ranking" in still_there
    assert still_there["candles_analyzed"] == 520

    written = list(tmp_path.glob("report_walk_forward_rsi*.json"))
    assert written, [p.name for p in tmp_path.iterdir()]

    report = json.loads(written[0].read_text())
    assert report["strategy"] == "rsi"
    assert "oos_beat_benchmark" in report


def test_the_walk_forward_path_is_never_the_one_it_was_given():
    import backtest

    for out in ("results/comparison_local.json", "results/x.json", "out", "/tmp/r.json"):
        assert backtest.walk_forward_path(out, "rsi") != out


def test_the_projects_own_naming_is_preserved():
    # So the committed results and the chart generator still line up.
    import backtest

    assert backtest.walk_forward_path("results/comparison_local.json", "rsi") == (
        "results/walk_forward_rsi_local.json"
    )


def test_each_strategy_gets_its_own_walk_forward_file():
    import backtest

    paths = {backtest.walk_forward_path("out.json", s) for s in ("rsi", "macd", "ema")}
    assert len(paths) == 3
