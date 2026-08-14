"""Generate the README hero banner.

banner.png was the one asset committed without the code that produced it, so it
could not be re-themed, resized, or pointed at a different instrument the way
the three result charts can. It also could not be repaired: when a stray white
dot turned up over the title, the glyph underneath had to be rebuilt by hand
from the same letter elsewhere in the word.

The equity line is drawn from a fixed seed rather than live prices — the banner
is decoration, and a hero image that changes every time the data is refreshed
would produce noise in the diff for no benefit.

Usage:
    python make_banner.py
    python make_banner.py --title "Market Strategy" --out assets/banner.png
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Bbox

# Same palette as make_charts.py, so the hero and the figures below it match.
BG = "#0B0D12"
GRID = "#131622"
MUTED = "#64748B"
PURPLE = "#7C3AED"
BLUE = "#38BDF8"

WIDTH, HEIGHT = 1200, 340
DPI = 100

# Measured from the original: verticals every 150px, horizontals every 60px.
GRID_X = 150
GRID_Y = 60

# All measured off the committed banner: the title's ink spans x 81-987, the
# subtitle's x 77-852. TITLE_X sits left of 81 because TextPath positions the
# pen, not the ink, and the 'M' carries a left side bearing.
TITLE_X = 72
TITLE_BASELINE = 152
TITLE_SIZE = 100
TITLE_END = "#4796F6"

SUBTITLE_X = 76
SUBTITLE_Y = 186
SUBTITLE_SIZE = 13.5
SUBTITLE_TRACKING = 6.3

TITLE = "Market Strategy"
SUBTITLE = "BACKTESTING SIX CLASSIC SYSTEMS ON BANK NIFTY"


def tracked_text(ax, fig, x, y, s, *, tracking, **kw):
    """
    Draw text with letter spacing.

    matplotlib's Text has no tracking property, and the banner's small caps
    rely on it heavily, so each character is placed individually and the pen
    advanced by its measured width plus the gap.
    """
    renderer = fig.canvas.get_renderer()
    pen = x
    for ch in s:
        t = ax.text(pen, y, ch, **kw)
        if ch == " ":
            pen += kw.get("fontsize", 12) * 0.42 + tracking
            t.remove()
            continue
        w = t.get_window_extent(renderer).width
        pen += w + tracking
    return pen


def equity_walk(n: int = 52, seed: int = 11) -> np.ndarray:
    """A plausible-looking equity path. Fixed seed keeps the banner stable."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.12, 1.0, n)
    walk = np.cumsum(steps)
    return walk - walk.min()


def draw(title: str, subtitle: str, out: str) -> None:
    fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axis("off")

    # Everything below is positioned in pixels-from-the-top, the way the
    # original was measured; Y flips that into the axes' y-up space.
    Y = lambda y: HEIGHT - y

    for x in range(GRID_X - 50, WIDTH, GRID_X):
        ax.axvline(x, color=GRID, linewidth=1, zorder=0)
    for y in range(GRID_Y, HEIGHT, GRID_Y):
        ax.axhline(Y(y), color=GRID, linewidth=1, zorder=0)

    # ---- the strategy line, over a band that fades into the background -----
    walk = equity_walk()
    xs = np.linspace(-10, WIDTH + 10, len(walk))
    span = walk.max() - walk.min() or 1
    # The subtitle's baseline sits at y=196, so the line is kept below 200.
    # Amplitude, not position, is what stops it cutting through the text.
    ys = Y(258 - (walk - walk.min()) / span * 56)

    fade = LinearSegmentedColormap.from_list("fade", [BG, "#1B3B51"])
    ax.imshow(
        np.linspace(0, 1, 256).reshape(-1, 1),
        cmap=fade, aspect="auto", extent=[0, WIDTH, 0, Y(190)], zorder=1,
    )
    ax.plot(xs, ys, color=BLUE, linewidth=2.4, solid_joinstyle="round", zorder=3)

    # ---- buy & hold: the benchmark the whole study is measured against -----
    ax.plot(xs, Y(np.linspace(268, 196, len(walk))), color=MUTED,
            linewidth=1.4, linestyle=(0, (5, 4)), zorder=2)
    ax.text(WIDTH - 12, Y(186), "BUY  &  HOLD", color=MUTED, fontsize=9.5,
            ha="right", va="bottom", fontweight="bold", zorder=4)

    # ---- title, filled with a horizontal gradient --------------------------
    # matplotlib cannot fill text with a gradient, so the glyph outlines are
    # taken as a path and used to clip an image drawn across their bounding box.
    font = FontProperties(family="DejaVu Sans", weight="bold", size=TITLE_SIZE)
    path = TextPath((TITLE_X, Y(TITLE_BASELINE)), title, prop=font)
    clip = PathPatch(path, facecolor="none", edgecolor="none", zorder=5)
    ax.add_patch(clip)

    bb = path.get_extents()
    ramp = LinearSegmentedColormap.from_list("title", [PURPLE, TITLE_END])
    grad = ax.imshow(
        np.linspace(0, 1, 512).reshape(1, -1),
        cmap=ramp, aspect="auto", zorder=6,
        extent=[bb.x0, bb.x1, bb.y0, bb.y1],
    )
    grad.set_clip_path(clip)

    tracked_text(fig=fig, ax=ax, x=SUBTITLE_X, y=Y(SUBTITLE_Y), s=subtitle,
                 tracking=SUBTITLE_TRACKING, color="#8A93A6",
                 fontsize=SUBTITLE_SIZE, va="top", ha="left", zorder=5)

    fig.savefig(out, dpi=DPI, facecolor=BG,
                bbox_inches=Bbox([[0, 0], [WIDTH / DPI, HEIGHT / DPI]]))
    plt.close(fig)

    # matplotlib writes RGBA; the committed banner is RGB, and an alpha channel
    # on an opaque image only inflates the file.
    Image.open(out).convert("RGB").save(out)
    print(f"Banner written to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the README banner.")
    parser.add_argument("--title", default=TITLE)
    parser.add_argument("--subtitle", default=SUBTITLE)
    parser.add_argument("--out", default="assets/banner.png")
    args = parser.parse_args()
    draw(args.title, args.subtitle, args.out)


if __name__ == "__main__":
    main()
