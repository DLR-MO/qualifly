# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

"""
Maps of a section of one flight on a white background (no base map): source
track, target reports and every filter variant, each target point linked to the
source position at its report time, plus a time-ordered route map and a stack
of one map per variant.

The times are times of day in UTC. Source and target come from config.yaml and
are read from data/<flight>/*.csv, written by run.py. Writes
data/<flight>/section_<start>_<end>.pdf and the _assignment/_routes/n_section
variants.

    python scripts/plot_flight_section.py <flight_id> 12:00:00 12:00:40
"""

import argparse
import math
import sys
from datetime import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Filter variants with their colour and marker
VARIANTS = {"mean": ("tab:blue", "s"), "median": ("tab:green", "^"), "kalman_smoother": ("tab:red", "D")}
# The markers as text symbols, before the panel labels of the stacked maps
SYMBOLS = {"o": "○", "s": "■", "^": "▲", "D": "◆"}


def read_section(path, start, end, lat, lon):
    """
    The positions of a pipeline CSV (timestamp index) between start and end,
    one row per timestamp like the pipeline compares them (OSN repeats a fix in several rows).
    """
    import pandas as pd

    df = pd.read_csv(path, index_col="timestamp")
    df.index = pd.to_datetime(df.index, utc=True, format="ISO8601")
    df = df.sort_index().between_time(start, end)[[lat, lon]].dropna()
    return df.groupby(level=0).median()


def add_scale_bar(ax, meters_per_degree):
    """A scale bar of about a fifth of the plot width, rounded to 1, 2 or 5 x 10^n meters."""
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    target = (x_max - x_min) * meters_per_degree / 5
    power = 10 ** math.floor(math.log10(target))
    length = max(m for m in (1, 2, 5) if m * power <= target) * power
    ax.add_artist(AnchoredSizeBar(ax.transData, length / meters_per_degree, f"{length:g} m", "lower left",
                                  frameon=False, size_vertical=(y_max - y_min) / 100, fontproperties={"size": 14}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("flight", help="Flight id")
    parser.add_argument("start", type=time.fromisoformat, help="Start time of day (UTC), e.g. 12:00:00 or 12:00:05.5")
    parser.add_argument("end", type=time.fromisoformat, help="End time of day (UTC), e.g. 12:00:40")
    parser.add_argument("--data", type=Path, default=ROOT / "data", help="Data folder (default data/)")
    parser.add_argument("--out", type=Path, help="Output PDF (default <data>/<flight>/section_<start>_<end>.pdf)")
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import pandas as pd

    from plot_error_histograms import save_figure

    plt.rcParams.update({"font.size": 14, "savefig.bbox": "tight"})

    from run import load_config
    from qualifly_lib.error_calculation import interp_with_gaps
    from qualifly_lib.data_fetching.data_fetcher import load_config as load_columns

    settings = load_config()
    config = load_columns(settings["source"], settings["target"])
    source, target = config["source"], config["target"]
    flight_dir = args.data / args.flight

    def section(name, columns):
        path = flight_dir / f"{name}.csv"
        if not path.is_file():
            sys.exit(f"{path} missing.")
        return read_section(path, args.start, args.end, columns.LAT, columns.LON).rename(
            columns={columns.LAT: "lat", columns.LON: "lon"})

    track = section(source.IDENTIFIER, source)
    # the source fixes: a GNSS fix held in several rows counts once, as in the pipeline
    track = track[(track.diff() != 0).any(axis=1)]
    reports = section(target.IDENTIFIER, target)
    if reports.empty:
        sys.exit(f"No {target.IDENTIFIER} positions of {args.flight} between {args.start} and {args.end}.")
    variants = {variant: section(variant, target).reindex(reports.index).dropna() for variant in VARIANTS}
    # the source at every report time, as the pipeline compares them
    reference = pd.DataFrame({c: interp_with_gaps(track.index.asi8 / 1e9, track[c].to_numpy(),
                                                  reports.index.asi8 / 1e9, settings.get("max_gap_s", 2.0))
                              for c in ("lat", "lon")}, index=reports.index).dropna()

    stamp = f"{args.start:%H%M%S}_{args.end:%H%M%S}"
    out = args.out or flight_dir / f"section_{stamp}.pdf"
    names = (source.IDENTIFIER, target.IDENTIFIER)
    for path, draw in ((out, lambda ax: draw_variants(ax, track, reports, variants, reference, names)),
                       (out.with_stem(f"{out.stem}_assignment"),
                        lambda ax: draw_assignment(ax, track, reports, reference, names)),
                       (out.with_stem(f"{out.stem}_routes"),
                        lambda ax: draw_routes(ax, track, reports, variants, names))):
        fig, ax = plt.subplots(figsize=(12, 6))
        draw(ax)
        finish_map(ax, reports["lat"].mean())
        save_figure(fig, path)
        plt.close(fig)
    save_small_multiples(track, reports, variants, reference, names, out.with_name(f"n_{out.name}"))


def save_small_multiples(track, reports, variants, reference, names, out):
    """
    One map per variant (original and every filter) stacked with the same section
    and scale: the source track, the variant's points and each point linked to the
    source position at its report time, and the points in their time order (dotted).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from plot_error_histograms import save_figure
    from qualifly_lib.error_calculation import _EARTH_RADIUS

    panels = {"original": reports, **variants}
    shown = pd_concat([track, reference, *panels.values()])
    lon_min, lon_max = shown["lon"].min(), shown["lon"].max()
    lat_min, lat_max = shown["lat"].min(), shown["lat"].max()
    margin_lon, margin_lat = (lon_max - lon_min) * 0.03, (lat_max - lat_min) * 0.1
    cos_lat = np.cos(np.radians(reports["lat"].mean()))
    # each panel as high as its section is in meters relative to its width
    ratio = (lat_max - lat_min + 2 * margin_lat) / ((lon_max - lon_min + 2 * margin_lon) * cos_lat)
    width = 12
    fig, axes = plt.subplots(len(panels), 1, figsize=(width, len(panels) * width * ratio + 0.8),
                             gridspec_kw={"hspace": 0.06})
    for ax, (name, points) in zip(axes, panels.items()):
        colour, marker = VARIANTS.get(name, ("black", "o"))
        ax.plot(track["lon"], track["lat"], color="0.6", linewidth=1.5, zorder=1, label=f"{names[0]} ({len(track)})")
        # the time order, dotted to tell it from the error links
        ax.plot(points["lon"], points["lat"], color=colour, linewidth=1, linestyle=":", zorder=2)
        linked = points.loc[points.index.intersection(reference.index)]
        target = reference.loc[linked.index]
        ax.plot(np.stack([linked["lon"], target["lon"]]), np.stack([linked["lat"], target["lat"]]),
                color=colour, linewidth=1, zorder=2)
        ax.plot(target["lon"], target["lat"], "x", color="tab:purple", markersize=5, zorder=3,
                label=f"{names[0]} at report time")
        style = dict(markerfacecolor="none", markeredgewidth=1.2) if name == "original" else {}
        ax.plot(points["lon"], points["lat"], marker, color=colour, markersize=6, zorder=4, **style)
        label = f"{names[1]} report" if name == "original" else name
        ax.text(0.005, 0.95, f"{SYMBOLS[marker]} {label} ({len(points)})", transform=ax.transAxes, va="top", fontsize=13,
                color=colour, fontweight="bold")
        ax.set(xlim=(lon_min - margin_lon, lon_max + margin_lon), ylim=(lat_min - margin_lat, lat_max + margin_lat),
               xticks=[], yticks=[], aspect=1 / cos_lat)
    add_scale_bar(axes[-1], _EARTH_RADIUS * math.radians(1) * cos_lat)
    axes[-1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=12)
    save_figure(fig, out)
    plt.close(fig)


def pd_concat(frames):
    import pandas as pd

    return pd.concat(frames)


def draw_variants(ax, track, reports, variants, reference, names):
    """
    The source track, the reports and every filter variant, each filtered point
    linked to the source position at its report time: the error the pipeline scores.
    """
    import numpy as np

    ax.plot(track["lon"], track["lat"], color="0.3", linewidth=1.5, zorder=1, label=f"{names[0]} ({len(track)})")
    draw_reports(ax, reports, names[1])
    for variant, filtered in variants.items():
        colour, marker = VARIANTS[variant]
        linked = filtered.loc[filtered.index.intersection(reference.index)]
        target = reference.loc[linked.index]
        ax.plot(np.stack([linked["lon"], target["lon"]]), np.stack([linked["lat"], target["lat"]]),
                color=colour, linewidth=0.8, alpha=0.7, zorder=2)
        ax.plot(filtered["lon"], filtered["lat"], marker, color=colour, markersize=6, zorder=4,
                label=f"{variant} ({len(filtered)})")
    draw_reference(ax, reference, names[0])


def draw_routes(ax, track, reports, variants, names):
    """The source track, the reports and every filter variant, each connected in time order (its flight route)."""
    ax.plot(track["lon"], track["lat"], color="0.3", linewidth=1.5, zorder=1, label=f"{names[0]} ({len(track)})")
    ax.plot(reports["lon"], reports["lat"], color="black", linewidth=0.8, zorder=2)
    draw_reports(ax, reports, names[1], markersize=6)
    for variant, filtered in variants.items():
        colour, marker = VARIANTS[variant]
        ax.plot(filtered["lon"], filtered["lat"], marker=marker, color=colour, markersize=3.5, linewidth=1,
                zorder=4, label=f"{variant} ({len(filtered)})")


def draw_assignment(ax, track, reports, reference, names):
    """
    The source fixes and the reports in their time order, each report linked to
    the source position at its report time: a report stamped too late or too
    early gets a long link and a jump in the order.
    """
    import numpy as np

    ax.plot(track["lon"], track["lat"], ".", color="0.5", markersize=3, zorder=1, label=f"{names[0]} ({len(track)})")
    ax.plot(reports["lon"], reports["lat"], color="black", linewidth=0.8, zorder=2)
    draw_reports(ax, reports, names[1])
    linked = reports.loc[reference.index]
    ax.plot(np.stack([linked["lon"], reference["lon"]]), np.stack([linked["lat"], reference["lat"]]),
            color="tab:purple", linewidth=1.2, linestyle="--", zorder=2)
    draw_reference(ax, reference, names[0])


def draw_reference(ax, reference, name):
    ax.plot(reference["lon"], reference["lat"], "x", color="tab:purple", markersize=7, zorder=4,
            label=f"{name} at report time")


def draw_reports(ax, reports, name, markersize=9):
    ax.plot(reports["lon"], reports["lat"], "o", markersize=markersize, markerfacecolor="none", markeredgecolor="black",
            markeredgewidth=1.2, zorder=3, label=f"{name} report ({len(reports)})")


def finish_map(ax, lat):
    """A map: equal meters on both axes, only the frame, a scale bar and the legend below."""
    import numpy as np

    from qualifly_lib.error_calculation import _EARTH_RADIUS

    cos_lat = np.cos(np.radians(lat))
    ax.set_aspect(1 / cos_lat)
    ax.set(xticks=[], yticks=[])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=12)
    add_scale_bar(ax, _EARTH_RADIUS * math.radians(1) * cos_lat)


if __name__ == "__main__":
    main()
