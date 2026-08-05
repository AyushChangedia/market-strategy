"""Generate the result charts used in the README.

Every number is read from results/*.json, so the charts and the published
metrics cannot drift apart. Regenerating them still needs no data pull — the
committed result files are the input.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG = "#0B0D12"
GRID = "#1B2030"
TEXT = "#E2E8F0"
MUTED = "#64748B"
BLUE = "#38BDF8"
PURPLE = "#7C3AED"
RED = "#F43F5E"
GREY = "#475569"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
    "font.family": "DejaVu Sans",
})

RESULTS = Path(__file__).resolve().parent / "results"

# The JSON carries full labels; the charts need something that fits an axis.
SHORT_NAMES = {
    "rsi": "RSI",
    "macd": "MACD",
    "ema_cross": "EMA 20/50",
    "donchian": "Donchian",
    "bollinger": "Bollinger",
    "supertrend": "Supertrend",
}


def load(filename: str) -> dict:
    path = RESULTS / filename
    if not path.exists():
        raise SystemExit(f"{path} not found — run backtest.py first.")
    with open(path) as handle:
        return json.load(handle)


COMPARISON = load("comparison.json")
RANKING = COMPARISON["ranking"]

STRATEGIES = [SHORT_NAMES.get(r["strategy"], r["label"]) for r in RANKING]
RETURNS = [r["total_return_pct"] for r in RANKING]
TRADES = [r["total_trades"] for r in RANKING]
WIN_RATE = [r["win_rate_pct"] for r in RANKING]
# An undefined profit factor serialises as null; plot it at zero like a
# strategy that never traded, which is how it already appeared.
PROFIT_FACTOR = [r["profit_factor"] or 0.0 for r in RANKING]
BUY_HOLD = COMPARISON["buy_and_hold_return_pct"]


def returns_chart():
    """Every strategy vs buy-and-hold. The main finding."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = [BLUE if r > 0 else RED if r < 0 else GREY for r in RETURNS]
    bars = ax.bar(STRATEGIES, RETURNS, color=colors, width=0.62, zorder=3)

    ax.axhline(BUY_HOLD, color=TEXT, linestyle="--", linewidth=1.6, zorder=4)
    ax.text(len(STRATEGIES) - 0.55, BUY_HOLD + 0.35, f"Buy & hold  {BUY_HOLD}%",
            color=TEXT, fontsize=10, ha="right", fontweight="bold")
    ax.axhline(0, color=GRID, linewidth=1.2, zorder=2)

    for bar, val in zip(bars, RETURNS):
        offset = 0.3 if val >= 0 else -0.75
        ax.text(bar.get_x() + bar.get_width() / 2, val + offset, f"{val}%",
                ha="center", color=TEXT, fontsize=10, zorder=5)

    ax.set_ylim(min(min(RETURNS), 0) * 1.6, max(max(RETURNS), BUY_HOLD) * 1.18)
    ax.set_ylabel("Total return (%)", fontsize=11)
    ax.set_title("Not one strategy beat doing nothing",
                 fontsize=15, fontweight="bold", pad=16, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig("assets/returns.png", dpi=160)
    plt.close(fig)


def winrate_chart():
    """Win rate vs profit factor — the payoff-ratio lesson."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    live = [i for i, t in enumerate(TRADES) if t > 0]
    x = [WIN_RATE[i] for i in live]
    y = [PROFIT_FACTOR[i] for i in live]
    sizes = [TRADES[i] * 42 + 90 for i in live]
    colors = [BLUE if PROFIT_FACTOR[i] > 1 else RED for i in live]

    ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1.5, zorder=2)
    ax.text(max(x) + 3.5, 1.03, "break-even", color=MUTED, fontsize=9.5, ha="right")

    ax.scatter(x, y, s=sizes, c=colors, alpha=0.72,
               edgecolors=TEXT, linewidths=1.1, zorder=3)

    labels = {"RSI": (8, 12), "MACD": (8, -20), "EMA 20/50": (-14, 16),
              "Bollinger": (10, 12), "Supertrend": (10, -20)}
    for i in live:
        name = STRATEGIES[i]
        ax.annotate(name, (WIN_RATE[i], PROFIT_FACTOR[i]),
                    textcoords="offset points", xytext=labels.get(name, (8, 10)),
                    color=TEXT, fontsize=10.5, fontweight="bold")

    ax.set_xlabel("Win rate (%)", fontsize=11)
    ax.set_ylabel("Profit factor", fontsize=11)
    ax.set_title("A high win rate does not mean a profitable strategy",
                 fontsize=15, fontweight="bold", pad=16, loc="left")
    ax.text(0.0, -0.19,
            "Bubble size = number of trades.  Bollinger wins most often and still loses money.",
            transform=ax.transAxes, color=MUTED, fontsize=10)

    ax.set_xlim(min(x) - 7, max(x) + 5)
    ax.set_ylim(min(y) - 0.28, max(y) + 0.35)
    ax.grid(color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig("assets/winrate.png", dpi=160)
    plt.close(fig)


def walkforward_chart():
    """In-sample vs out-of-sample — the overfitting lesson."""
    fig, ax = plt.subplots(figsize=(10, 5))

    report = load("walk_forward_rsi.json")
    rows = report["folds"]

    folds = [f"Fold {r['fold']}" for r in rows] + ["Average"]
    train = [r["train_return_pct"] for r in rows] + [report["avg_train_return_pct"]]
    test = [r["test_return_pct"] for r in rows] + [report["avg_test_return_pct"]]

    idx = np.arange(len(folds))
    width = 0.36

    ax.bar(idx - width / 2, train, width, label="In-sample (trained on)",
           color=PURPLE, zorder=3)
    ax.bar(idx + width / 2, test, width, label="Out-of-sample (unseen)",
           color=GREY, zorder=3)

    for i, (tr, te) in enumerate(zip(train, test)):
        ax.text(i - width / 2, tr + 0.18, f"{tr}%", ha="center",
                color=TEXT, fontsize=9.5, zorder=4)
        ax.text(i + width / 2, te + 0.18, f"{te}%", ha="center",
                color=TEXT, fontsize=9.5, zorder=4)

    ax.set_xticks(idx)
    ax.set_xticklabels(folds)
    ax.set_ylabel("Return (%)", fontsize=11)
    ax.set_ylim(0, max(train + test) * 1.17)
    ax.set_title(f"{report['label'].split()[0]}, the best performer, "
                 "mostly stopped working on unseen data",
                 fontsize=15, fontweight="bold", pad=16, loc="left")
    trades = report["oos_total_trades"]
    ax.text(0.0, -0.16,
            f"Robustness score {report['robustness_score']} — verdict: likely overfitted.  "
            f"The entire out-of-sample period produced {trades} "
            f"trade{'' if trades == 1 else 's'}.",
            transform=ax.transAxes, color=MUTED, fontsize=10)

    leg = ax.legend(frameon=False, loc="upper right", fontsize=10.5)
    for text in leg.get_texts():
        text.set_color(TEXT)

    ax.grid(axis="y", color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig("assets/walkforward.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    returns_chart()
    winrate_chart()
    walkforward_chart()
    print("Charts written to assets/")