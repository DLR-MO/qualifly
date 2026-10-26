# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

"""
Heatmaps of the horizontal position difference target - source (e.g. OSN -
onboard records) in east/north meters on [-200, +200] m, per flight and over all
flights.

Every report is compared with the source interpolated to its time, as in
flight_processor.preprocess_data. Source and target come from config.yaml and
are read from data/<flight>/<identifier>.csv, written by run.py. Writes
data/<flight>/position_offset_heatmap.pdf and data/position_offset_heatmap.pdf.

    python scripts/plot_offset_heatmap.py [--range 200] [--bin 1]
"""

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FILE_NAME = "position_offset_heatmap.pdf"


def aligned_offsets(flight_dir, config, tmp_dir):
    """
    East/north offsets in meters of every target report to the source at the
    report time, or None when the pipeline has nothing to compare.
    preprocess_data writes its CSVs to `tmp_dir`, the data folder stays untouched.
    """
    import numpy as np
    import pandas as pd

    from qualifly_lib.error_calculation import _EARTH_RADIUS
    from qualifly_lib.flight_processor import preprocess_data

    source, target = config["source"], config["target"]
    (Path(tmp_dir) / flight_dir.name).mkdir(exist_ok=True)
    _, _, src, tgt = preprocess_data(pd.read_csv(flight_dir / f"{source.IDENTIFIER}.csv"),
                                     pd.read_csv(flight_dir / f"{target.IDENTIFIER}.csv"),
                                     config, tmp_dir, flight_dir.name)
    if src is None:
        return None
    lat = np.radians(src[source.LAT].to_numpy())
    east = _EARTH_RADIUS * np.radians(tgt[target.LON].to_numpy() - src[source.LON].to_numpy()) * np.cos(lat)
    north = _EARTH_RADIUS * np.radians(tgt[target.LAT].to_numpy() - src[source.LAT].to_numpy())
    return east, north


def save_heatmap(east, north, limit, bin_m, out):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots(figsize=(7, 6))
    edges = np.arange(-limit, limit + bin_m / 2, bin_m)
    *_, image = ax.hist2d(east, north, bins=[edges, edges], norm=LogNorm(), cmap="viridis")
    fig.colorbar(image, ax=ax, label="reports per bin")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.plot(np.median(east), np.median(north), "r+", markersize=12, label="median")
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal",
           xlabel="east offset [m]", ylabel="north offset [m]")
    ax.legend(loc="upper right")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--range", type=float, default=200, help="Half width of the plot in meters (default 200)")
    parser.add_argument("--bin", type=float, default=1, help="Bin size in meters (default 1)")
    parser.add_argument("--data", type=Path, default=ROOT / "data", help="Data folder (default data/)")
    args = parser.parse_args()
    if args.range <= 0 or args.bin <= 0:
        parser.error("--range and --bin must be positive")

    import numpy as np

    from run import load_config
    from qualifly_lib.data_fetching.data_fetcher import load_config as load_columns

    settings = load_config()
    config = load_columns(settings["source"], settings["target"])
    config["max_gap_s"] = settings.get("max_gap_s", 2.0)
    source, target = config["source"].IDENTIFIER, config["target"].IDENTIFIER

    flight_dirs = sorted(d for d in args.data.iterdir()
                         if (d / f"{source}.csv").is_file() and (d / f"{target}.csv").is_file())
    if not flight_dirs:
        sys.exit(f"No flight folders with {source}.csv and {target}.csv in {args.data}.")

    all_east, all_north = [], []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for flight_dir in flight_dirs:
            offsets = aligned_offsets(flight_dir, config, tmp_dir)
            if offsets is None:
                print(f"{flight_dir.name}: nothing to compare - skipped")
                continue
            save_heatmap(*offsets, args.range, args.bin, flight_dir / FILE_NAME)
            all_east.append(offsets[0])
            all_north.append(offsets[1])

    if not all_east:
        sys.exit("No flight had comparable reports.")
    out = args.data / FILE_NAME
    save_heatmap(np.concatenate(all_east), np.concatenate(all_north), args.range, args.bin, out)
    print(f"Saved {len(all_east)} flight heatmaps and {out}")


if __name__ == "__main__":
    main()
