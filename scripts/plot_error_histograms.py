# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

"""
Distributions of the per-report errors of one variant (default `original`),
pooled over all flights: position error, position error with 0.1/0.3/0.6 s
tolerance and track distance, plus the altitude and ground speed errors.

The errors come from data/<flight>/epu_validation.csv, written by run.py.
Writes histograms, ECDFs and violin plots per variant into the data folder.

    python scripts/plot_error_histograms.py [flight ...] [--variant original] [--bins 60] [--quantile 99]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Per-report error column of epu_validation.csv (without the variant prefix), its label and colour
# (the tolerances in shades of one blue, from no tolerance dark to 0.6 s light)
ERRORS = {
    "position_error": ("position error", "#08306b"),
    "position_error_tol100ms": ("position error, 0.1 s tolerance", "#2171b5"),
    "position_error_tol300ms": ("position error, 0.3 s tolerance", "#4292c6"),
    "position_error_tol600ms": ("position error, 0.6 s tolerance", "#6baed6"),
    "track_distance": ("track distance", "tab:orange"),
}
HISTOGRAMS = ("position_error", "position_error_tol600ms", "track_distance")
# NACp categories whose EPU bound is drawn in the ECDF
SHOWN_NACP = (11, 10, 9, 8)
# The violin figures: file name part, their errors (signed ones shown absolute), unit, axis start
FIGURES = {
    "": (ERRORS, "m", 0.1),
    "altitude": ({"altitude_error": ("geometric altitude error", "tab:green"),
                  "baro_error": ("barometric altitude error", "tab:olive")}, "m", 0.1),
    "groundspeed": ({"groundspeed_error": ("ground speed error", "tab:purple")}, "m/s", 0.001),
}
# The variants of the pipeline, with the outline colour of the filters (as in plot_flight_section.py)
VARIANTS = {"original": "0.8", "mean": "tab:blue", "median": "tab:green", "kalman_smoother": "tab:red"}
# Reports a violin's density is estimated from (a random sample of the larger sets)
_DENSITY_SAMPLE = 50_000

POSTER_STYLE = {"font.size": 14, "xtick.labelsize": 15, "ytick.labelsize": 15,
                "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"}


def save_figure(fig, out):
    """Saves the figure; a file held open by a viewer (Windows) is reported and skipped."""
    try:
        fig.savefig(out)
        print(f"Saved {out}")
    except PermissionError:
        print(f"Skipped {out} - the file is open in another program.")


def save_histogram(values, label, colour, bins, quantile, out):
    import matplotlib.pyplot as plt
    import numpy as np

    p95 = np.percentile(values, 95)
    x_max = max(np.percentile(values, quantile), p95 * 1.1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=bins, range=(0, x_max), color=colour)
    ax.axvline(p95, color="black", linestyle="--", linewidth=1.5)
    ax.annotate(f"P95 = {p95:.1f} m", (p95, 1), xycoords=("data", "axes fraction"),
                xytext=(6, -4), textcoords="offset points", va="top")
    ax.set(xlim=(0, x_max), xlabel=f"{label} [m]", ylabel="reports")
    save_figure(fig, out)
    plt.close(fig)


def save_ecdf(errors, out):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (label, colour), values in errors:
        values = np.sort(values)
        p95 = np.percentile(values, 95)
        ax.plot(values, np.arange(1, values.size + 1) / values.size, color=colour, linewidth=2,
                label=f"{label} (P95 {p95:.1f} m)")
        ax.plot(p95, 0.95, "o", color=colour)
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1)
    log_error_axis(ax, errors)
    ax.set(ylim=(0, 1.01), ylabel="share of reports")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=12, frameon=False)
    save_figure(fig, out)
    plt.close(fig)


def epu_bounds():
    """The (NACp, EPU bound in m) of the SHOWN_NACP categories."""
    from qualifly_lib.accuracy_comparison import NACP_TABLE

    return [(nacp, epu) for nacp, epu, _ in NACP_TABLE if nacp in SHOWN_NACP]


def figure_bounds(figure):
    """
    The bounds of a FIGURES figure as (name, (category, bound) pairs) rows: the EPU
    of the NACp, the VEPU of the NACp and the GVA, or the NACv.
    """
    from qualifly_lib.accuracy_comparison import GVA_TABLE, NACP_TABLE, NACV_TABLE

    if figure == "altitude":
        return [("NACp (VEPU)", [(nacp, vepu) for nacp, _, vepu in NACP_TABLE if vepu is not None]),
                ("GVA", GVA_TABLE)]
    return [("NACv", NACV_TABLE)] if figure == "groundspeed" else [("NACp", epu_bounds())]


def save_violins(errors, out, x_max=None, bounds=None, unit="m", x_min=0.1, overlays=()):
    """
    One horizontal violin per error with its P95: on the log error axis, or with
    x_max on a linear axis from 0 to x_max, the density of the errors in that range
    and the share beyond it at the right edge. bounds: rows as of figure_bounds,
    default the EPU of the NACp; x_min: start of the log axis. overlays:
    (name, colour, values per row or None) drawn as outlines over grey violins.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    bounds = [("NACp", epu_bounds())] if bounds is None else bounds
    fig, ax = plt.subplots(figsize=(7, 0.9 * len(errors) + 1.5))
    for row, ((label, colour), values) in enumerate(errors):
        colour = VARIANTS["original"] if overlays else colour
        draw_violin(ax, values, row, x_max, facecolor=colour, edgecolor=colour, alpha=0.85)
        if x_max is not None:
            share = (values > x_max).mean()
            ax.annotate(f"{'<0.1%' if 0 < share < 0.001 else f'{share:.1%}'} > {x_max:g} {unit}", (1, row),
                        xycoords=("axes fraction", "data"), ha="right", va="center", fontsize=10, color="0.3",
                        bbox=dict(facecolor="white", edgecolor="none"))
        for _, line_colour, rows in overlays:
            if rows[row] is not None:
                draw_violin(ax, rows[row], row, x_max, facecolor="none", edgecolor=line_colour, linewidth=1.5, alpha=1)
                p95 = np.percentile(rows[row], 95)
                ax.plot([p95, p95], [row - 0.42, row + 0.42], color=line_colour, linewidth=1.5)
        p95 = np.percentile(values, 95)
        ax.plot([p95, p95], [row - 0.42, row + 0.42], color="black", linewidth=2)
        ax.annotate(f"P95 {p95:.1f} {unit}", (p95, row), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=11, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
    if x_max is None:
        log_error_axis(ax, errors, bounds, unit, x_min)
    else:
        ax.set(xlim=(0, x_max), xlabel=f"error [{unit}]")
        draw_bounds(ax, bounds, short=True)
    ax.set_yticks(range(len(errors)), [label.replace(", ", "\n") for (label, _), _ in errors], fontsize=12)
    ax.set_ylim(len(errors) - 0.5, -0.5)
    if overlays:
        handles = [Patch(color=VARIANTS["original"], label="original (P95 black)"),
                   *(Line2D([], [], color=colour, linewidth=1.5, label=name) for name, colour, _ in overlays)]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=11)
    save_figure(fig, out)
    plt.close(fig)


def draw_violin(ax, values, row, x_max, **style):
    """
    The violin of values in a row: the density of log10(values) shown on the log
    axis, or with x_max that of the values up to x_max (the far outliers would
    stretch it over their range and flatten it).
    """
    import numpy as np

    # zero errors have no log
    shown = np.log10(values[values > 0]) if x_max is None else values[values <= x_max]
    # the density from a random sample: the shape of 50k reports equals that of all
    if shown.size > _DENSITY_SAMPLE:
        shown = np.random.default_rng(0).choice(shown, _DENSITY_SAMPLE, replace=False)
    parts = ax.violinplot(shown, positions=[row], orientation="horizontal",
                          widths=0.85, showextrema=False, points=400)
    for body in parts["bodies"]:
        if x_max is None:
            vertices = body.get_paths()[0].vertices
            vertices[:, 0] = 10 ** vertices[:, 0]
        body.set(**style)


def log_error_axis(ax, errors, bounds=None, unit="m", x_min=0.1):
    """Log error axis from x_min up to twice the 99.9th percentile, with the bounds (default the NACp EPU)."""
    import numpy as np

    # single far outliers would stretch the axis over decades
    upper = max(np.percentile(values, 99.9) for _, values in errors) * 2
    ax.set(xscale="log", xlim=(x_min, upper), xlabel=f"error [{unit}]")
    draw_bounds(ax, [("NACp", epu_bounds())] if bounds is None else bounds)


def draw_bounds(ax, rows, short=False):
    """
    The bounds within the x axis as vertical lines, a row of labels above the axis
    per (name, (category, bound) pairs) of rows. One row labels each line
    "name category"; with `short` (close lines on a linear axis) or several rows
    a line has its category, and the row its name at the left.
    """
    low, high = ax.get_xlim()
    named_rows = short or len(rows) > 1
    for row, (name, bounds) in enumerate(rows):
        style = dict(xycoords=("data", "axes fraction"), xytext=(0, 4 + 16 * row), textcoords="offset points",
                     ha="center", va="bottom", color="0.4", fontsize=11)
        for category, bound in bounds:
            if low <= bound <= high:
                ax.axvline(bound, color="0.6", linewidth=1, zorder=0)
                ax.annotate(str(category) if named_rows else f"{name} {category}", (bound, 1), **style)
        if named_rows:
            ax.annotate(name, (0, 1), **{**style, "xycoords": "axes fraction", "ha": "right"})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("flights", nargs="*", help="Flight ids (default: all flights of the data folder)")
    parser.add_argument("--variant", default="original", help="original, mean, median or kalman_smoother")
    parser.add_argument("--bins", type=int, default=60, help="Number of histogram bins (default 60)")
    parser.add_argument("--quantile", type=float, default=99,
                        help="Histogram x axis ends at this percentile (default 99)")
    parser.add_argument("--linear-max", type=float, default=80,
                        help="End of the linear violin axis in m (default 80)")
    parser.add_argument("--format", default="pdf", help="File format, e.g. pdf, png, svg (default pdf)")
    parser.add_argument("--data", type=Path, default=ROOT / "data", help="Data folder (default data/)")
    args = parser.parse_args()
    if args.bins < 1 or not 0 < args.quantile <= 100:
        parser.error("--bins must be >= 1 and --quantile in (0, 100]")

    import matplotlib.pyplot as plt
    import pandas as pd

    files = ([args.data / flight / "epu_validation.csv" for flight in args.flights]
             or sorted(args.data.glob("*/epu_validation.csv")))
    missing = [f for f in files if not f.is_file()]
    if missing or not files:
        sys.exit(f"No epu_validation.csv: {missing or args.data}")

    names = [name for errors, *_ in FIGURES.values() for name in errors]
    columns = [f"{variant}_{name}" for variant in {args.variant, *VARIANTS} for name in names]
    frames = [pd.read_csv(f, usecols=lambda c: c in columns) for f in files]
    table = pd.concat(frames, ignore_index=True).reindex(columns=columns)

    plt.rcParams.update(POSTER_STYLE)
    shown = error_rows(table, args.variant, ERRORS)
    if not shown:
        sys.exit(f"No {args.variant} errors in {len(files)} flights.")
    for name, ((label, colour), values) in shown.items():
        if name in HISTOGRAMS:
            out = args.data / f"hist_{args.variant}_{name}.{args.format}"
            save_histogram(values, label, colour, args.bins, args.quantile, out)
    save_ecdf(list(shown.values()), args.data / f"ecdf_{args.variant}.{args.format}")
    save_violins(list(shown.values()), args.data / f"violins_linear_{args.variant}.{args.format}", args.linear_max)

    for figure, (errors, unit, x_min) in FIGURES.items():
        stem = f"violins_{figure}_" if figure else "violins_"
        style = dict(bounds=figure_bounds(figure), unit=unit, x_min=x_min)
        per_variant = {variant: error_rows(table, variant, errors) for variant in VARIANTS}
        for variant, rows in per_variant.items():
            if rows:
                save_violins(list(rows.values()), args.data / f"{stem}{variant}.{args.format}", **style)
        base = per_variant["original"]
        if base:
            overlays = [(variant, colour, [per_variant[variant].get(name, (None, None))[1] for name in base])
                        for variant, colour in VARIANTS.items() if variant != "original"]
            save_violins(list(base.values()), args.data / f"{stem}filters.{args.format}", overlays=overlays, **style)


def error_rows(table, variant, errors):
    """{error: ((label, colour), absolute values)} of the errors of a variant that have values."""
    rows = {}
    for name, (label, colour) in errors.items():
        values = table[f"{variant}_{name}"].dropna().abs().to_numpy()
        if values.size:
            rows[name] = ((label, colour), values)
        else:
            print(f"No {variant} {label} in the data - skipped")
    return rows


if __name__ == "__main__":
    main()
