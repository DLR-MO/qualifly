# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

"""
Barometric altitude over a time section of one flight: source, target and every
filter variant (mean, median, kalman_smoother), built as in the pipeline.

The times are times of day in UTC. Source and target come from config.yaml and
are read from data/<flight>/<identifier>.csv, written by run.py.
Writes data/<flight>/baro_<start>_<end>.pdf.

    python scripts/plot_baro_section.py <flight_id> 12:00:00 12:00:40
"""

import argparse
import sys
from datetime import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Filter variants with their colour and marker (as in plot_flight_section.py)
VARIANTS = {"mean": ("tab:blue", "s"), "median": ("tab:green", "^"), "kalman_smoother": ("tab:red", "D")}


def read_flight(path):
    """The numeric columns of a pipeline CSV, one row per timestamp like the pipeline compares them."""
    import pandas as pd

    df = pd.read_csv(path, index_col="timestamp")
    df.index = pd.to_datetime(df.index, utc=True, format="ISO8601")
    return df.sort_index().select_dtypes("number").groupby(level=0).median()


def baro_variants(reports, target):
    """The barometric altitude of the filter variants, made as in the pipeline (flight_processor)."""
    from qualifly_lib.filtering import median_filter, use_filtering
    from qualifly_lib.flight_processor import _at_reports, _target_grid

    reports = reports.dropna(subset=[target.LAT, target.LON, target.BAROALT])
    grid = _target_grid(reports, target)
    variants = {name: _at_reports(df, grid, reports, target) for name, df in use_filtering(grid, target).items()}
    variants["median"] = median_filter(reports, target)
    return {name: variants[name][target.BAROALT] for name in VARIANTS if name in variants}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("flight", help="Flight id")
    parser.add_argument("start", type=time.fromisoformat, help="Start time of day (UTC), e.g. 12:00:00")
    parser.add_argument("end", type=time.fromisoformat, help="End time of day (UTC), e.g. 12:00:40")
    parser.add_argument("--data", type=Path, default=ROOT / "data", help="Data folder (default data/)")
    parser.add_argument("--out", type=Path, help="Output PDF (default <data>/<flight>/baro_<start>_<end>.pdf)")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    from run import load_config
    from qualifly_lib.data_fetching.data_fetcher import load_config as load_columns

    plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.bbox": "tight"})
    settings = load_config()
    config = load_columns(settings["source"], settings["target"])
    source, target = config["source"], config["target"]
    flight_dir = args.data / args.flight
    paths = [flight_dir / f"{config[side].IDENTIFIER}.csv" for side in ("source", "target")]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        sys.exit(f"Missing {missing}.")
    track, reports = (read_flight(p) for p in paths)
    for df, columns in ((track, source), (reports, target)):
        if getattr(columns, "BAROALT", None) not in df.columns:
            sys.exit(f"No barometric altitude in the {columns.IDENTIFIER} data of {args.flight}.")

    def section(series):
        return series.dropna().between_time(args.start, args.end)

    shown = section(reports[target.BAROALT])
    if shown.empty:
        sys.exit(f"No {target.IDENTIFIER} barometric altitude between {args.start} and {args.end}.")

    fig, ax = plt.subplots(figsize=(10, 5))
    reference = section(track[source.BAROALT])
    ax.plot(reference.index, reference, color="0.3", linewidth=1.5, label=source.IDENTIFIER)
    ax.plot(shown.index, shown, "o", markersize=5, markerfacecolor="none", markeredgecolor="black",
            markeredgewidth=0.9, zorder=5, label=f"{target.IDENTIFIER} report")
    for name, values in baro_variants(reports, target).items():
        colour, marker = VARIANTS[name]
        ax.plot(section(values), marker, color=colour, markersize=3, linestyle="-", linewidth=0.8, label=name)

    # no time axis: the section is about the shape, not the clock time
    ax.set(xticks=[], ylabel="barometric altitude [m]")
    ax.spines["bottom"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=5, frameon=False)

    out = args.out or flight_dir / f"baro_{args.start:%H%M%S}_{args.end:%H%M%S}.pdf"
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
