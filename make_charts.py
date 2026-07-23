"""Generate the result charts used in the README.

Numbers are hardcoded from the backtest output in results/comparison.json
so the charts can be regenerated without re-running the data pull.
"""

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

STRATEGIES = ["RSI", "MACD", "EMA 20/50", "Donchian", "Bollinger", "Supertrend"]
RETURNS = [5.35, 4.95, 4.39, 0.0, -1.07, -3.24]
TRADES = [6, 18, 3, 0, 7, 8]
WIN_RATE = [50.0, 33.3, 33.3, 0.0, 57.1, 25.0]
PROFIT_FACTOR = [1.91, 1.39, 2.75, 0.0, 0.92, 0.88]
BUY_HOLD = 9.3


def returns_chart():
    """Every strategy vs buy-and-hold. The main finding."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = [BLUE if r > 0 else RED if r < 0 else GREY for r in RETURNS]
    bars = ax.bar(STRATEGIES, RETURNS, color=colors, width=0.62, zorder=3)

    ax.axhline(BUY_HOLD, color=TEXT, linestyle="--", linewidth=1.6, zorder=4)
    ax.text(5.45, BUY_HOLD + 0.35, f"Buy & hold  {BUY_HOLD}%",
            color=TEXT, fontsize=10, ha="right", fontweight="bold")
    ax.axhline(0, color=GRID, linewidth=1.2, zorder=2)

    for bar, val in zip(bars, RETURNS):
        offset = 0.3 if val >= 0 else -0.75
        ax.text(bar.get_x() + bar.get_width() / 2, val + offset, f"{val}%",
                ha="center", color=TEXT, fontsize=10, zorder=5)

    ax.set_ylim(-5.2, 11)
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
    ax.text(60.5, 1.03, "break-even", color=MUTED, fontsize=9.5, ha="right")

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

    ax.set_xlim(18, 62)
    ax.set_ylim(0.6, 3.1)
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

    folds = ["Fold 1", "Fold 2", "Fold 3", "Average"]
    train = [8.18, 0.0, 3.01, 3.73]
    test = [3.11, 0.0, 0.0, 1.04]

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
    ax.set_ylim(0, 9.6)
    ax.set_title("RSI, the best performer, mostly stopped working on unseen data",
                 fontsize=15, fontweight="bold", pad=16, loc="left")
    ax.text(0.0, -0.16,
            "Robustness score 0.46 — verdict: likely overfitted.  The entire out-of-sample "
            "period produced 1 trade.",
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