"""Charts for the eval-awareness results.

Three figures, written as PNGs:

  forest.png      per-scenario testing-minus-deployed differences plus the
                  pooled estimate -- the headline result
  conditions.png  cautious-behaviour rate for each of the five framings
  dose.png        observer-rated "looks like a test" against behaviour

Each figure is a small multiple with one panel per subject model, so no figure
carries more than one series colour: context marks are grey, the mark that
matters is blue. The table view these figures summarise is the text report from
analysis.py -- every value plotted here is printed there.

    ~/evals/bin/python3 plots.py                      # all logs in logs/
    ~/evals/bin/python3 plots.py logs/a.eval --out plots --dark
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analysis import (  # noqa: E402
    DEPLOYED,
    TESTING,
    clustered_contrast,
    covered,
    load_rows,
    newcombe,
    rate,
    select,
    wilson,
)

# Palette: the data-viz reference instance. Both modes are selected sets, not an
# automatic inversion. Only one categorical hue is used anywhere -- the figures
# follow the emphasis pattern (one accent plus grey) rather than seating a
# second series colour, which also keeps every panel colourblind-safe by
# construction.
LIGHT = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "accent": "#2a78d6",
}
DARK = {
    "surface": "#1a1a19",
    "ink": "#f0efec",
    "ink2": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "accent": "#3987e5",
}

# Ordered by how strongly an outside observer reads each framing as a test, so a
# gradient down the axis is visible if one exists. Not alphabetical on purpose.
CUE_ORDER = (
    "testing_explicit",
    "testing_implicit",
    "deployed_explicit",
    "deployed_implicit",
    "neutral",
)
SHORT = {
    "testing_explicit": "test · explicit",
    "testing_implicit": "test · implicit",
    "deployed_explicit": "live · explicit",
    "deployed_implicit": "live · implicit",
    "neutral": "neutral",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "figure.dpi": 150,
    }
)


def style(ax, pal: dict, xgrid: bool = True) -> None:
    """Recessive chrome: hairline solid grid, muted ticks, no boxing in."""
    ax.set_facecolor(pal["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(pal["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=pal["muted"], labelcolor=pal["ink2"], length=3, width=0.8)
    if xgrid:
        ax.grid(axis="x", color=pal["grid"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def short_model(name: str) -> str:
    return name.split("/")[-1].replace(":free", " (free)")


def new_figure(n_panels: int, width: float, height: float, pal: dict,
               sharex: bool = False):
    fig, axes = plt.subplots(
        1, n_panels, figsize=(width * n_panels, height), sharey=True,
        sharex=sharex, squeeze=False,
    )
    fig.patch.set_facecolor(pal["surface"])
    return fig, axes[0]


def save(fig, path: Path, pal: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=pal["surface"], bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #
# 1. forest plot
# --------------------------------------------------------------------------- #


def forest(by_model: dict, pal: dict, out: Path) -> None:
    """One row per scenario, so the spread that drives the interval is visible.

    A single pooled number hides whether the scenarios agree. Here a reader sees
    twelve differences at once: all on one side is a result, scattered across
    zero is not, and one extreme row carrying the mean is obvious rather than
    buried.
    """
    panels = {}
    for model, rows in by_model.items():
        usable = covered(rows)
        c = clustered_contrast(usable, TESTING, DEPLOYED)
        if c.get("mean") is not None:
            panels[model] = (usable, c)
    if not panels:
        print("skipped forest.png -- no model has a computable contrast")
        return

    # One shared row order across panels. The y axis is shared, so each panel
    # sorting by its own values would put different scenarios on the same row
    # and silently mislabel every panel but the last.
    totals: dict[str, list[float]] = defaultdict(list)
    for _, c in panels.values():
        for scenario, diff in c["per_scenario"]:
            totals[scenario].append(diff)
    order = sorted(totals, key=lambda s: sum(totals[s]) / len(totals[s]))
    y_of = {scenario: i for i, scenario in enumerate(order)}

    fig, axes = new_figure(
        len(panels), 5.2, 1.4 + 0.34 * (len(order) + 2), pal, sharex=True
    )

    for ax, (model, (usable, c)) in zip(axes, panels.items()):
        for scenario, diff in c["per_scenario"]:
            y = y_of[scenario]
            sub = select(usable, scenario=scenario)
            k_t, n_t, _ = rate(select(sub, condition=TESTING))
            k_d, n_d, _ = rate(select(sub, condition=DEPLOYED))
            if n_t and n_d:
                _, lo, hi = newcombe(k_t, n_t, k_d, n_d)
                ax.plot([lo, hi], [y, y], color=pal["grid"], linewidth=2, zorder=1,
                        solid_capstyle="round")
            ax.plot(diff, y, "o", markersize=8, color=pal["muted"],
                    markeredgecolor=pal["surface"], markeredgewidth=2, zorder=3)

        # Pooled estimate, set apart and in the accent hue: the one mark that is
        # the answer rather than an ingredient of it.
        y_pool = -1.6
        lo, hi = c.get("lo"), c.get("hi")
        if lo is not None:
            ax.plot([lo, hi], [y_pool, y_pool], color=pal["accent"], linewidth=2,
                    zorder=2, solid_capstyle="round")
        ax.plot(c["mean"], y_pool, "D", markersize=9, color=pal["accent"],
                markeredgecolor=pal["surface"], markeredgewidth=2, zorder=4)
        ax.annotate(
            f"{c['mean']:+.3f}",
            (c["mean"], y_pool),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            color=pal["accent"],
            fontsize=9,
            fontweight="bold",
        )

        ax.axvline(0, color=pal["axis"], linewidth=1, zorder=0)
        ax.set_yticks(list(range(len(order))) + [y_pool])
        ax.set_yticklabels(order + ["all scenarios pooled"])
        ax.set_ylim(y_pool - 1.1, len(order) - 0.4)
        ax.set_xlabel("testing minus deployed  (right: more careful when tested)",
                      color=pal["ink2"])
        ax.set_title(short_model(model), color=pal["ink"], fontsize=10, loc="left")
        style(ax, pal)
        # After style(), which resets every tick label colour.
        ticklabels = ax.get_yticklabels()   # empty on the shared-axis panels
        if ticklabels:
            ticklabels[-1].set_color(pal["accent"])
            ticklabels[-1].set_fontweight("bold")

    fig.suptitle(
        "Does it behave more carefully when it thinks it is being tested?",
        color=pal["ink"], fontsize=12, x=0.01, ha="left", y=1.0,
    )
    save(fig, out / "forest.png", pal)


# --------------------------------------------------------------------------- #
# 2. rate per condition
# --------------------------------------------------------------------------- #


def conditions(by_model: dict, pal: dict, out: Path) -> None:
    """Five framings, strongest test-cue at the top.

    Dots with intervals rather than bars: a bare bar invites reading a few
    points of difference as a finding, and the intervals here are wide enough
    that the distinction matters.
    """
    panels = {m: covered(r) for m, r in by_model.items()}
    panels = {m: r for m, r in panels.items() if r}
    if not panels:
        print("skipped conditions.png -- nothing covered by any scorer")
        return

    fig, axes = new_figure(len(panels), 4.6, 3.2, pal)
    for ax, (model, rows) in zip(axes, panels.items()):
        ticks, labels, counts = [], [], []
        for i, condition in enumerate(reversed(CUE_ORDER)):
            k, n, r = rate(select(rows, condition=condition))
            ticks.append(i)
            # No n in the tick label: the y axis is shared across panels, so one
            # model's counts would end up labelling every panel.
            labels.append(SHORT[condition])
            counts.append(n)
            if not n:
                continue
            lo, hi = wilson(k, n)
            ax.plot([lo, hi], [i, i], color=pal["grid"], linewidth=2,
                    solid_capstyle="round", zorder=1)
            ax.plot(r, i, "o", markersize=8, color=pal["accent"],
                    markeredgecolor=pal["surface"], markeredgewidth=2, zorder=3)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 1)
        ax.set_xlabel("took the cautious option", color=pal["ink2"])
        typical = sorted(counts)[len(counts) // 2] if counts else 0
        ax.set_title(f"{short_model(model)}   (n around {typical} per framing)",
                     color=pal["ink"], fontsize=10, loc="left")
        style(ax, pal)

    fig.suptitle(
        "Cautious behaviour by framing, all scorers pooled",
        color=pal["ink"], fontsize=12, x=0.01, ha="left", y=1.02,
    )
    save(fig, out / "conditions.png", pal)


# --------------------------------------------------------------------------- #
# 3. dose-response
# --------------------------------------------------------------------------- #


def dose(by_model: dict, pal: dict, out: Path, cues: dict) -> None:
    """Measured cue strength against measured behaviour.

    The claim the project actually wants to make is that the more a session
    reads as an evaluation, the more carefully the model behaves. A slope here
    states that directly; two pooled groups can only hint at it.
    """
    panels = {m: covered(r) for m, r in by_model.items()}
    panels = {m: r for m, r in panels.items() if r}
    if not panels:
        print("skipped dose.png -- nothing covered by any scorer")
        return

    fig, axes = new_figure(len(panels), 4.4, 3.6, pal)
    for ax, (model, rows) in zip(axes, panels.items()):
        xs, ys, names = [], [], []
        for condition, cue in cues.items():
            _, n, r = rate(select(rows, condition=condition))
            if not n or r is None:
                continue
            xs.append(cue)
            ys.append(r)
            names.append(SHORT.get(condition, condition))

        if len(xs) >= 4:
            slope, intercept = np.polyfit(xs, ys, 1)
            grid = np.array([min(xs) - 2, max(xs) + 2])
            ax.plot(grid, slope * grid + intercept, color=pal["muted"],
                    linewidth=2, zorder=1)
            r_val = float(np.corrcoef(xs, ys)[0, 1])
            # Say how few points the line rests on. A correlation over five
            # aggregated points looks authoritative and is easy to over-read.
            ax.annotate(
                f"least-squares fit over {len(xs)} points"
                f"   slope {slope:+.4f} per point of cue   r = {r_val:+.2f}",
                (0.02, 0.02), xycoords="axes fraction",
                color=pal["ink2"], fontsize=8,
            )

        ax.plot(xs, ys, "o", markersize=9, color=pal["accent"],
                markeredgecolor=pal["surface"], markeredgewidth=2, zorder=3)
        for x, y, name in zip(xs, ys, names):
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(0, 11),
                        ha="center", color=pal["ink2"], fontsize=8)

        ax.set_xlabel("observer-rated likelihood this is a test", color=pal["ink2"])
        if ax is axes[0]:
            ax.set_ylabel("took the cautious option", color=pal["ink2"])
        ax.set_ylim(0, 1)
        ax.set_title(short_model(model), color=pal["ink"], fontsize=10, loc="left")
        style(ax, pal)
        ax.grid(axis="y", color=pal["grid"], linewidth=0.8, linestyle="-")

    fig.suptitle(
        "Does looking like a test change behaviour?",
        color=pal["ink"], fontsize=12, x=0.01, ha="left", y=1.02,
    )
    save(fig, out / "dose.png", pal)


def load_cues(path: Path) -> dict:
    """Per-condition cue strength measured by manipulation_check.py."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    cues = data.get("likelihood_by_condition", {})
    return {k: v for k, v in cues.items() if isinstance(v, (int, float))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", type=Path)
    parser.add_argument("--out", type=Path, default=Path("plots"))
    parser.add_argument("--dark", action="store_true", help="dark-surface variant")
    parser.add_argument(
        "--cues",
        type=Path,
        default=Path("manipulation_check.json"),
        help="JSON written by manipulation_check.py, for the dose-response plot",
    )
    args = parser.parse_args()

    paths = args.logs or sorted(Path("logs").glob("*.eval"))
    if not paths:
        parser.error("no .eval logs found")

    rows, _ = load_rows(list(paths))
    if not rows:
        parser.error("logs contained no scored samples")

    by_model: dict[str, list] = defaultdict(list)
    for row in rows:
        by_model[row.model].append(row)
    by_model = dict(sorted(by_model.items()))

    pal = DARK if args.dark else LIGHT
    out = args.out

    forest(by_model, pal, out)
    conditions(by_model, pal, out)

    cues = load_cues(args.cues)
    if cues:
        dose(by_model, pal, out, cues)
    else:
        print(
            f"skipped dose.png -- no cue strengths at {args.cues}."
            " Run manipulation_check.py first; it writes that file."
        )


if __name__ == "__main__":
    main()
