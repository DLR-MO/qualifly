# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

"""
How late each target report (e.g. OSN) is stamped relative to the source: report
time minus the time the source track passed the closest point on its path. The
closest point is searched within --window seconds of the report, so another pass
over the same place is not matched. Positive = stamped late. Unlike the
pipeline's per-flight `time_offset_s`, this is per report.

Source and target come from config.yaml and are read from
data/<flight>/<identifier>.csv, written by run.py. Writes
data/<flight>/report_delay[_<start>_<end>].pdf, or data/report_delay_all.pdf
without a flight id.

    python scripts/plot_report_delay.py <flight_id> [12:00:00 12:00:40] [--window 30]
    python scripts/plot_report_delay.py
"""

import argparse
import sys
from datetime import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def report_delays(track, reports, window_s):
    """
    Per report (lat/lon frames, datetime index): its delay in seconds, its
    distance in meters to the closest point of the track within +-window_s, and
    whether that point is the outer end of the window's track (the real match
    lies outside the window). NaN where the window holds no track segment.
    """
    import numpy as np
    import pandas as pd

    from qualifly_lib.error_calculation import _latlon_to_local_xy

    lat0 = np.radians(track["lat"].mean())
    track_xy = _latlon_to_local_xy(track["lat"], track["lon"], lat0)
    report_xy = _latlon_to_local_xy(reports["lat"], reports["lon"], lat0)
    t_track, t_reports = track.index.asi8 / 1e9, reports.index.asi8 / 1e9

    delays, distances = np.full(len(reports), np.nan), np.full(len(reports), np.nan)
    at_edge = np.zeros(len(reports), dtype=bool)
    for k, (point, t) in enumerate(zip(report_xy, t_reports)):
        first, last = np.searchsorted(t_track, [t - window_s, t + window_s])
        if last - first < 2:
            continue
        start, step = track_xy[first:last - 1], np.diff(track_xy[first:last], axis=0)
        # position of the closest point on each segment: 0 = its start, 1 = its end
        along = np.clip(((point - start) * step).sum(axis=1) / np.maximum((step ** 2).sum(axis=1), 1e-12), 0, 1)
        gaps = np.hypot(*(point - start - along[:, None] * step).T)
        best = int(np.argmin(gaps))
        passed = t_track[first + best] + along[best] * (t_track[first + best + 1] - t_track[first + best])
        delays[k], distances[k] = t - passed, gaps[best]
        at_edge[k] = (best == 0 and along[0] == 0) or (best == len(step) - 1 and along[-1] == 1)
    return pd.DataFrame({"delay_s": delays, "distance_m": distances, "at_edge": at_edge}, index=reports.index)


def flight_delays(flight_dir, source, target, start, end, window_s):
    """report_delays of one flight folder between start and end, None without positions to compare."""
    from plot_flight_section import read_section

    paths = [flight_dir / f"{columns.IDENTIFIER}.csv" for columns in (source, target)]
    if not all(path.is_file() for path in paths):
        return None
    track, reports = (read_section(path, start, end, columns.LAT, columns.LON).rename(
        columns={columns.LAT: "lat", columns.LON: "lon"}) for path, columns in zip(paths, (source, target)))
    # the source fixes: a GNSS fix held in several rows counts once, as in the pipeline
    track = track[(track.diff() != 0).any(axis=1)]
    if reports.empty or len(track) < 2:
        return None
    return report_delays(track, reports, window_s).dropna()


def save_timeline(delays, window_s, target_name, out):
    """The delay of every report of one flight over the report time, coloured by its distance to the track."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np

    at_edge = delays["at_edge"]
    matched = delays[~at_edge]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axhline(0, color="0.6", linewidth=1)
    points = ax.scatter(matched.index, matched["delay_s"], c=matched["distance_m"], s=18, cmap="viridis",
                        vmin=0, vmax=max(np.percentile(matched["distance_m"], 95), 1))
    fig.colorbar(points, ax=ax, label="distance to track [m]", extend="max")
    if at_edge.any():
        ax.scatter(delays.index[at_edge], delays["delay_s"][at_edge], marker="^", s=30, facecolors="none",
                   edgecolors="tab:red", label=f"no match within ±{window_s:g} s ({at_edge.sum()})")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=11, frameon=False)
    # linear within +-1 s, where most reports are, logarithmic beyond for the far outliers
    ax.set_yscale("symlog", linthresh=1)
    ax.set_yticks([t for t in (-30, -10, -3, -1, 0, 1, 3, 10, 30) if abs(t) <= window_s])
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.set(xlabel="report time [UTC]", ylabel="report delay [s]",
           title=f"{len(matched)} {target_name} reports, median delay {matched['delay_s'].median():.2f} s")
    fig.savefig(out)
    plt.close(fig)


def save_pooled(delays, n_flights, window_s, out):
    """Histogram of the delays of all reports of all flights, with median and P95."""
    import matplotlib.pyplot as plt
    import numpy as np

    values = delays.loc[~delays["at_edge"], "delay_s"].to_numpy()
    low, high = min(np.percentile(values, 0.5), 0), np.percentile(values, 99.5)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values, bins=np.arange(low, high + 0.01, 0.01), color="tab:blue")
    for value, name, style in ((np.median(values), "median", ":"), (np.percentile(values, 95), "P95", "--")):
        ax.axvline(value, color="black", linestyle=style, linewidth=1.5, label=f"{name} {value:.2f} s")
    outside = int(((values < low) | (values > high)).sum())
    ax.set(xlim=(low, high), xlabel="report delay [s]", ylabel="reports")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False,
              title=f"{values.size} reports, {n_flights} flights",
              title_fontsize=11, fontsize=12)
    print(f"{outside} reports outside [{low:.2f}, {high:.2f}] s not shown, "
          f"{int(delays['at_edge'].sum())} without a match within ±{window_s:g} s")
    fig.savefig(out)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("flight", nargs="?", help="Flight id (default: all flights pooled into one histogram)")
    parser.add_argument("start", nargs="?", type=time.fromisoformat, help="Start time of day (UTC), default: whole flight")
    parser.add_argument("end", nargs="?", type=time.fromisoformat, help="End time of day (UTC)")
    parser.add_argument("--window", type=float, default=30, help="Search window around the report time in s (default 30)")
    parser.add_argument("--data", type=Path, default=ROOT / "data", help="Data folder (default data/)")
    parser.add_argument("--out", type=Path, help="Output PDF (default <data>/<flight>/report_delay[_<start>_<end>].pdf, "
                                                 "all flights: <data>/report_delay_all.pdf)")
    args = parser.parse_args()
    if (args.start is None) != (args.end is None):
        parser.error("give both start and end, or neither")
    if args.window <= 0:
        parser.error("--window must be positive")

    import matplotlib.pyplot as plt
    import pandas as pd

    from run import load_config
    from qualifly_lib.data_fetching.data_fetcher import load_config as load_columns

    settings = load_config()
    config = load_columns(settings["source"], settings["target"])
    source, target = config["source"], config["target"]
    start, end = args.start or time.min, args.end or time.max
    plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.bbox": "tight"})

    if args.flight is None:
        delays = [flight_delays(d, source, target, start, end, args.window)
                  for d in sorted(args.data.iterdir()) if d.is_dir()]
        delays = [d for d in delays if d is not None and not d.empty]
        if not delays:
            sys.exit(f"No flights with positions to compare in {args.data}.")
        out = args.out or args.data / "report_delay_all.pdf"
        save_pooled(pd.concat(delays), len(delays), args.window, out)
    else:
        delays = flight_delays(args.data / args.flight, source, target, start, end, args.window)
        if delays is None or delays.empty:
            sys.exit(f"No positions of {args.flight} to compare between {start} and {end}.")
        stamp = "" if args.start is None else f"_{args.start:%H%M%S}_{args.end:%H%M%S}"
        out = args.out or args.data / args.flight / f"report_delay{stamp}.pdf"
        save_timeline(delays, args.window, target.IDENTIFIER, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
