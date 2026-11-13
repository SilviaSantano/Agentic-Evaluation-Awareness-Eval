"""The experimental design as one compact figure.

Reads left to right as the equation it is:

    [12 scenarios x 5 framings]  x  [10 epochs]  =  600 runs

The grid is the real 12 x 5 layout, with the three scenario categories as
blocks, so the shape of the design is visible rather than described. Epochs are
a stack because they are repeats of the same cell, which is also why they buy no
statistical power against the clustering -- the figure says that by construction.

Structural, so no colour carries data: one accent for the total, greys for
everything else.

    ~/evals/bin/python3 design_figure.py            # light mode, into plots/
    ~/evals/bin/python3 design_figure.py --dark
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from plots import CUE_ORDER, DARK, LIGHT, SHORT, save  # noqa: E402

EPOCHS = 10
CATEGORIES = ["transparency", "human_in_loop", "self_limiting"]
PER_CATEGORY = 4
BLOCK_GAP = 0.7

N_SCENARIOS = len(CATEGORIES) * PER_CATEGORY
N_FRAMINGS = len(CUE_ORDER)


def column_positions() -> tuple[list[float], list[tuple[str, float, float]]]:
    """X for each scenario column, plus (category, first_x, last_x) per block."""
    xs: list[float] = []
    bands: list[tuple[str, float, float]] = []
    x = 0.0
    for category in CATEGORIES:
        first = x
        for _ in range(PER_CATEGORY):
            xs.append(x)
            x += 1.0
        bands.append((category, first, x - 1.0))
        x += BLOCK_GAP
    return xs, bands


def cell(ax, x: float, y: float, *, fill: str, edge: str, size: float = 0.82,
         lw: float = 0.9, zorder: int = 2) -> None:
    ax.add_patch(FancyBboxPatch(
        (x - size / 2, y - size / 2), size, size,
        boxstyle="round,pad=0,rounding_size=0.14",
        linewidth=lw, facecolor=fill, edgecolor=edge, zorder=zorder,
    ))


def design_figure(pal: dict, out: Path) -> None:
    xs, bands = column_positions()
    grid_right = xs[-1]

    # Layout anchors, in grid units, reading left to right.
    x_times1 = grid_right + 1.5
    x_stack = x_times1 + 1.6
    x_equals = x_stack + 2.3
    x_total = x_equals + 1.0

    fig, ax = plt.subplots(figsize=(10.6, 3.1))
    fig.patch.set_facecolor(pal["surface"])
    ax.set_facecolor(pal["surface"])
    ax.set_aspect("equal")
    ax.axis("off")

    # ---------------- the 12 x 5 grid ----------------
    for x in xs:
        for row in range(N_FRAMINGS):
            cell(ax, x, row, fill=pal["grid"], edge=pal["axis"])

    # framing names down the left
    for row, framing in enumerate(CUE_ORDER):
        ax.text(-0.75, row, SHORT[framing], ha="right", va="center",
                color=pal["ink2"], fontsize=8)

    # category names above their block, with a rule spanning the block
    for category, first, last in bands:
        ax.plot([first - 0.41, last + 0.41], [-0.95, -0.95],
                color=pal["axis"], linewidth=0.9, solid_capstyle="butt")
        ax.text((first + last) / 2, -1.35, category, ha="center", va="center",
                color=pal["muted"], fontsize=8)

    ax.text((xs[0] + grid_right) / 2, N_FRAMINGS - 0.35,
            f"{N_SCENARIOS} scenarios  x  {N_FRAMINGS} framings  =  "
            f"{N_SCENARIOS * N_FRAMINGS} prompts",
            ha="center", va="top", color=pal["ink2"], fontsize=9)

    # ---------------- x epochs ----------------
    ax.text(x_times1, 2.0, "x", ha="center", va="center",
            color=pal["muted"], fontsize=15)

    # A stack, because epochs are repeats of one cell rather than new cells.
    step = 0.13
    for i in range(EPOCHS):
        back = i < EPOCHS - 1
        cell(
            ax,
            x_stack + (EPOCHS - 1 - i) * step,
            1.45 + (EPOCHS - 1 - i) * step,
            fill=pal["surface"] if back else pal["grid"],
            edge=pal["axis"],
            size=1.15, lw=0.8, zorder=2 + i,
        )
    ax.text(x_stack + 0.6, 3.6, f"{EPOCHS} epochs", ha="center", va="top",
            color=pal["ink2"], fontsize=9)
    ax.text(x_stack + 0.6, 4.3, "repeats of the same prompt",
            ha="center", va="top", color=pal["muted"], fontsize=8)

    # ---------------- = total ----------------
    ax.text(x_equals, 2.0, "=", ha="center", va="center",
            color=pal["muted"], fontsize=15)
    ax.text(x_total, 1.9, f"{N_SCENARIOS * N_FRAMINGS * EPOCHS}",
            ha="left", va="center", color=pal["accent"], fontsize=30)
    ax.text(x_total, 3.15, "runs", ha="left", va="center",
            color=pal["ink2"], fontsize=10)
    ax.text(x_total, 4.0, "one subject model", ha="left", va="center",
            color=pal["muted"], fontsize=8)

    ax.set_xlim(-3.1, x_total + 2.4)
    ax.set_ylim(N_FRAMINGS + 0.7, -2.1)

    fig.text(
        0.5, 0.035,
        "The realism arm added later doubles the grid to two arms at 6 epochs: "
        "12 x 5 x 2 x 6 = 720 runs. Not run.",
        color=pal["muted"], fontsize=8, ha="center", va="top",
    )

    save(fig, out / ("design-dark.png" if pal is DARK else "design.png"), pal)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("plots"))
    parser.add_argument("--dark", action="store_true")
    args = parser.parse_args()
    design_figure(DARK if args.dark else LIGHT, args.out)


if __name__ == "__main__":
    main()
