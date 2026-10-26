# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
import time

from qualifly_lib.utils import (SCORE_NAMES, atomic_path, open_in_browser, log_failed_ids, missing_columns,
                                msl_to_ellipsoid_height)

# Sample period of the grid the filters run on
_GRID_STEP = "1s"

_RESULT_FILES = ("scores.csv", "epu_validation.csv", "time_series.svg", "trajectory_map.html", "time_series.html")


def process_flight(collection_id, source_identifier, target_identifier, base_path="data", interactive=False,
                   fetch_options=None, max_gap_s=2.0, track_window_s=2.0, rescore=False):
    """
    Processes a single flight; skips recalculation if data and results are cached,
    unless `rescore` is set (the cached downloads are reused either way).
    `fetch_options` is the data_fetching section of config.yaml; `max_gap_s` and
    `track_window_s` are the settings of the same name (see docs/scores.md).
    """
    print(f"\n--- Processing Flight: {collection_id} ---")

    if isinstance(max_gap_s, bool) or not isinstance(max_gap_s, (int, float)) or max_gap_s < 0:
        raise ValueError(f"max_gap_s must be a non-negative number of seconds, got {max_gap_s!r}")

    if isinstance(track_window_s, bool) or not isinstance(track_window_s, (int, float)) or not track_window_s > 0:
        raise ValueError(f"track_window_s must be a positive number of seconds, got {track_window_s!r}")

    fetch_options = fetch_options or {}
    source_geo_alt_is_msl = _geo_alt_is_msl(source_identifier, fetch_options.get(source_identifier))
    target_geo_alt_is_msl = _geo_alt_is_msl(target_identifier, fetch_options.get(target_identifier))

    # Setup paths and read source and target
    os.makedirs(f"{base_path}/{collection_id}", exist_ok=True)

    # Skip recalculation if cached data and results are complete
    if not rescore and _has_cached_results(base_path, collection_id, source_identifier, target_identifier):
        print("Cached data and results found. Skipping recalculation.")
        _sync_summary(base_path, collection_id, source_identifier, target_identifier)
        print_results(base_path, collection_id)
        if interactive:
            _open_cached_results(base_path, collection_id)
        return

    # Heavy imports are only needed when actually processing
    import pandas as pd
    import urllib3
    from qualifly_lib.data_fetching.data_fetcher import (
        backend_caches_data,
        use_local_or_fetch_data,
        load_config,
        get_flight_info,
    )
    from qualifly_lib.error_calculation import calculate_errors
    from qualifly_lib.filtering import median_filter, use_filtering
    from qualifly_lib.accuracy_comparison import validate_nacp
    from qualifly_lib.visualization import (plot_trajectory_map, plot_time_series_overview,
                                            plot_time_series_overview_interactive)

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    config = load_config(source_identifier, target_identifier)
    config["max_gap_s"] = max_gap_s
    config["track_window_s"] = track_window_s
    config["source_geo_alt_is_msl"] = source_geo_alt_is_msl
    config["target_geo_alt_is_msl"] = target_geo_alt_is_msl

    # Fetch Data
    source_df = use_local_or_fetch_data(base_path, collection_id, source_identifier,
                                        fetch_options.get(source_identifier))
    if source_df is None:
        # A missing source must not trigger the target download
        print(f"Skipping flight {collection_id} - no {source_identifier} data, "
              f"so the {target_identifier} fetch is skipped as well.")
        _sync_summary(base_path, collection_id, source_identifier, target_identifier)
        return

    # Flight identity from the source backend, only looked up when the target needs a fetch
    if backend_caches_data(target_identifier) and _needs_fetch(base_path, collection_id, target_identifier):
        flight_info = get_flight_info(source_identifier, collection_id, fetch_options.get(source_identifier))
    else:
        flight_info = None
    target_df = use_local_or_fetch_data(base_path, collection_id, target_identifier,
                                        fetch_options.get(target_identifier), flight_info=flight_info)
    if target_df is None:
        print(f"Skipping flight {collection_id} - data incomplete.")
        _sync_summary(base_path, collection_id, source_identifier, target_identifier)
        return

    # Skip if required columns are missing (e.g. different parameter names in the source)
    missing = missing_columns(source_df, config["source"]) + missing_columns(target_df, config["target"])
    if missing:
        print(f"Missing required columns {missing} for flight {collection_id} - skipping flight. "
              f"The data source may use different parameter names or not provide them.")
        log_failed_ids(f"{base_path}/failed_flights.txt", collection_id)
        _sync_summary(base_path, collection_id, source_identifier, target_identifier)
        return

    # Preprocessing
    start = time.perf_counter()
    source_df, target_df, aligned_source_df, aligned_target_df = preprocess_data(
        source_df, target_df, config, base_path, collection_id
    )
    if aligned_source_df is None:
        _sync_summary(base_path, collection_id, source_identifier, target_identifier)
        return

    # Filters run on a regular 1 s grid, the median on the reports; each variant is scored at the report times
    grid = _target_grid(target_df, config["target"])
    grid_results = use_filtering(grid, config["target"]) if len(grid) > 1 else {}
    median = median_filter(target_df, config["target"])
    filtered_results = {**grid_results, "median": median}
    all_target_dfs = {
        "original": aligned_target_df,
        **{name: _at_reports(df, grid, aligned_target_df, config["target"]) for name, df in grid_results.items()},
        "median": median.loc[aligned_target_df.index],
    }
    end = time.perf_counter()
    print(f"Elapsed time for preprocessing: {end - start:.2f} seconds")

    # Error Calculation
    start = time.perf_counter()
    score, errors = calculate_errors(
        aligned_source_df, all_target_dfs, config, source_track=source_df)
    end = time.perf_counter()
    print(f"Elapsed time for error calculation: {end - start:.2f} seconds")
    scores_df = pd.DataFrame.from_dict(score, orient="index")
    scores_df.index.name = "variant"
    with atomic_path(f"{base_path}/{collection_id}/scores.csv") as tmp:
        scores_df.to_csv(tmp)
    if not errors:
        print("No errors calculated - skipping variant-specific results.")
        _sync_summary(base_path, collection_id, source_identifier, target_identifier)
        return
    for variant in all_target_dfs:
        all_target_dfs[variant] = all_target_dfs[variant].join(errors[variant])
        if variant != "original":
            with atomic_path(f"{base_path}/{collection_id}/{variant}.csv") as tmp:
                all_target_dfs[variant].to_csv(tmp)

    # NACp Validation
    start = time.perf_counter()
    validate_nacp(base_path, collection_id, all_target_dfs, config)
    end = time.perf_counter()
    print(f"Elapsed time for NACp validation: {end - start:.2f} seconds")

    # Update consolidated summary (summary.pq / scores.pq)
    _sync_summary(base_path, collection_id, source_identifier, target_identifier)

    # Results summary
    print_results(base_path, collection_id, scores=score)

    # Plotting: the source at its real position samples
    start = time.perf_counter()
    source_df = source_df.dropna(subset=[config["source"].LAT, config["source"].LON])
    path = f"{base_path}/{collection_id}/time_series.svg"
    fig_map = plot_trajectory_map(
        source_df=source_df,
        target_df=target_df,
        config=config,
        variant_dfs={name: df for name, df in all_target_dfs.items() if name != "original"}
    )
    fig_ts = plot_time_series_overview(
        source_df=source_df,
        target_df=target_df,
        config=config,
        filtered_results=filtered_results,
        score=score
    )
    end = time.perf_counter()
    print(f"Elapsed time for visualization: {end - start:.2f} seconds")
    map_html = f"{base_path}/{collection_id}/trajectory_map.html"
    ts_html = f"{base_path}/{collection_id}/time_series.html"
    with atomic_path(map_html) as tmp:
        fig_map.write_html(tmp)
    with atomic_path(path) as tmp:
        # format pinned: the temporary file has no .svg suffix
        fig_ts.savefig(tmp, format="svg", bbox_inches="tight")

    fig_ts_interactive = plot_time_series_overview_interactive(
        source_df, target_df, config, filtered_results, collection_id, score
    )
    with atomic_path(ts_html) as tmp:
        fig_ts_interactive.write_html(tmp)

    if interactive:
        open_in_browser(map_html)
        open_in_browser(ts_html)


def _sync_summary(base_path, collection_id, source_identifier, target_identifier):
    """
    Upserts this flight's rows in summary.pq / scores.pq, also for skipped flights.
    Never interrupts the pipeline.
    """
    try:
        from qualifly_lib.summary import upsert_flight
        updated = upsert_flight(base_path, collection_id, source_identifier, target_identifier)
        if updated:
            print(f"Summary updated for {collection_id}.")
    except Exception as e:
        print(f"Warning: summary update failed for {collection_id}: {e}")


def _geo_alt_is_msl(identifier, options):
    """
    The validated data_fetching.{identifier}.geo_alt_is_msl flag (default false).
    """
    value = (options or {}).get("geo_alt_is_msl", False)
    if not isinstance(value, bool):
        raise ValueError(
            f"data_fetching.{identifier}.geo_alt_is_msl must be true or false, got {value!r}"
        )
    return value


def find_stored_data(base_path):
    """
    Returns (flight folders, summary/log files) the pipeline stored in `base_path`.
    Only empty folders or folders holding a file the pipeline writes count, so
    e.g. the input folders of local backends are kept.
    """
    from qualifly_lib.data_fetching.data_fetcher import available_identifiers
    from qualifly_lib.summary import SUMMARY_FILE, SCORES_FILE

    base = Path(base_path)
    if not base.is_dir():
        return [], []

    written = {"scores.csv"}
    for identifier in available_identifiers():
        written |= {f"{identifier}.csv", f"{identifier}_not_found.txt"}

    flight_dirs = []
    for folder in sorted(entry for entry in base.iterdir() if entry.is_dir()):
        names = {entry.name for entry in folder.iterdir()}
        if not names or names & written:
            flight_dirs.append(folder)

    stored_files = [SUMMARY_FILE, SCORES_FILE, "summary.csv", "failed_flights.txt", "failed_nacp.txt"]
    return flight_dirs, [base / name for name in stored_files if (base / name).is_file()]


def print_results(base_path, collection_id, scores=None):
    """
    Prints the scores and the NACp summary of every variant. Loads the scores
    from {base_path}/{collection_id}/scores.csv if not given.
    """
    print(f"\n--- Results for flight {collection_id} ---")

    if scores is None:
        scores = _load_scores_csv(Path(base_path) / collection_id / "scores.csv")
    if not scores:
        print("No scores available.")
        return

    print(f"  {'variant':<18}" + "".join(f"{m:>{_width(m)}}" for m in SCORE_NAMES) + "    [m, s, m/s]")
    for variant, values in scores.items():
        row = "".join(f"{_fmt(values.get(m)):>{_width(m)}}" for m in SCORE_NAMES)
        print(f"  {variant:<18}{row}")

    _print_nacp_summary(Path(base_path) / collection_id / "epu_validation.csv", list(scores))


def _print_nacp_summary(filepath, variants):
    """
    Prints the share of reports fulfilling the NACp, the altitude claims
    (VEPU, GVA) and the velocity claim (NACv).
    """
    import pandas as pd
    from qualifly_lib.accuracy_comparison import (
        ERROR_KINDS, NACP_REQUIRED_SHARE, nacp_fulfilment, velocity_fulfilment, vertical_fulfilment
    )

    if not Path(filepath).exists():
        return
    df = pd.read_csv(filepath)
    if df.empty:
        return
    rows = (("position", nacp_fulfilment, ERROR_KINDS),
            ("altitude", vertical_fulfilment, {"vepu": "VEPU", "gva": "GVA"}),
            ("velocity", velocity_fulfilment, {"nacv": "NACv"}))
    print(f"Claims fulfilled (achieved >= claimed, needs >= {NACP_REQUIRED_SHARE:.0f}% of the reports):")
    for label, fulfilment, kinds in rows:
        for variant in variants:
            shares = fulfilment(df, variant)
            if all(share is None for share in shares.values()):
                continue
            print(f"  {label:<9}{variant:<18}" + "  |  ".join(
                f"{kinds[kind]} {_fmt(share, 1)}%" for kind, share in shares.items()))


def _load_scores_csv(filepath):
    import csv

    if not Path(filepath).exists():
        return None
    with open(filepath, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    scores = {}
    for row in rows:
        variant = row.get("variant")
        if not variant:
            continue
        values = {}
        for key, raw in row.items():
            if key == "variant":
                continue
            values[key] = None if raw in (None, "", "None") else float(raw)
        scores[variant] = values
    return scores


def _width(name):
    """Column width of a score in the printed table."""
    return max(10, len(name) + 2)


def _fmt(value, digits=2):
    if value is None or value != value:  # None or NaN
        return "n/a"
    return f"{float(value):.{digits}f}"


def _has_cached_results(base_path, collection_id, source_identifier, target_identifier):
    """
    True if both data source CSVs and all result files are present in {base_path}/{collection_id}/.
    Settings are not tracked: delete scores.csv to recalculate with other settings.
    """
    flight_dir = Path(base_path) / collection_id
    data_files = [f"{source_identifier}.csv", f"{target_identifier}.csv"]
    return all((flight_dir / name).exists() for name in (*data_files, *_RESULT_FILES))


def _needs_fetch(base_path, collection_id, identifier):
    """
    True when neither the cached CSV nor the '_not_found' marker of the backend exists.
    """
    flight_dir = Path(base_path) / collection_id
    return not (flight_dir / f"{identifier}.csv").exists() \
        and not (flight_dir / f"{identifier}_not_found.txt").exists()


def _open_cached_results(base_path, collection_id):
    flight_dir = Path(base_path) / collection_id
    for name in _RESULT_FILES:
        if name.endswith(".html"):
            open_in_browser(flight_dir / name)


def _indexed_by_time(df, time_column):
    """
    The rows sorted by a "timestamp" index of UTC datetimes made from the time
    column (unix epoch seconds or datetime strings).
    """
    import pandas as pd

    unit = "s" if pd.api.types.is_numeric_dtype(df[time_column]) else None
    timestamps = pd.to_datetime(df[time_column], utc=True, unit=unit, errors="coerce")
    return df.set_index(timestamps.rename("timestamp")).sort_index()


def _seconds(index, t0):
    return (index - t0).total_seconds().to_numpy()


def _interp_at(series, index, max_gap_s):
    """
    Interpolates series linearly onto the times of `index`, NaN across gaps wider than max_gap_s.
    """
    from qualifly_lib.error_calculation import interp_with_gaps

    known = series.dropna()
    t0 = index[0]
    return interp_with_gaps(_seconds(known.index, t0), known.to_numpy(dtype=float), _seconds(index, t0), max_gap_s)


def preprocess_data(source_df, target_df, config, base_path, collection_id):
    """
    Interpolates the source (without held positions) to the target's report times.
    Returns (source_track, target_reports, source_at_reports, target_compared);
    the last two are None when nothing can be compared.
    """
    import pandas as pd

    source = config["source"]
    target = config["target"]

    source_df = _indexed_by_time(source_df, source.TIME)
    target_df = _indexed_by_time(target_df, target.TIME)

    # Optional per-backend preparation hook (e.g. raw barometric altitudes)
    prepare = getattr(source, "prepare_data", None)
    if callable(prepare):
        source_df = prepare(source_df)

    # Save with new timstamp
    with atomic_path(f"{base_path}/{collection_id}/{source.IDENTIFIER}.csv") as tmp:
        source_df.to_csv(tmp)
    with atomic_path(f"{base_path}/{collection_id}/{target.IDENTIFIER}.csv") as tmp:
        target_df.to_csv(tmp)

    # MSL to ellipsoid height after the save, so the cached CSVs keep the delivered values
    if config.get("source_geo_alt_is_msl"):
        source_df = msl_to_ellipsoid_height(source_df, source.LAT, source.LON, source.GEOALT, source.IDENTIFIER)
    if config.get("target_geo_alt_is_msl"):
        target_df = msl_to_ellipsoid_height(target_df, target.LAT, target.LON, target.GEOALT, target.IDENTIFIER)

    try:
        # Source: the reference track at its real sample times, one row per timestamp
        track = _drop_held_positions(source_df.select_dtypes("number").groupby(level=0).median(), source)
        # Target: its position reports, one row per timestamp
        reports = target_df.select_dtypes("number").dropna(subset=[target.LAT, target.LON])
        reports = reports.groupby(level=0).median()

        if track.empty or len(reports) < 2:
            print("No overlapping timestamps between datasets thus cannot compute correlation.")
            print(f"source range:  {source_df.index.min()} to {source_df.index.max()}")
            print(f"target range:  {target_df.index.min()} to {target_df.index.max()}")
            return track, reports, None, None

        # Source: interpolated at the report times, not across gaps wider than max_gap_s
        max_gap_s = config.get("max_gap_s", 2.0)
        source_aligned = pd.DataFrame(
            {c: _interp_at(track[c], reports.index, max_gap_s) for c in track.columns}, index=reports.index,
        ).dropna(subset=[source.LAT, source.LON])
        target_aligned = reports.loc[source_aligned.index]
    except KeyError as e:
        print(f"An error occurred while processing parameters: {e}. Skipping flight - "
              "the data may use different parameter names. See failed_flights.txt.")
        log_failed_ids(f"{base_path}/failed_flights.txt", collection_id)
        return source_df, target_df, None, None

    if source_aligned.empty:
        print("No comparable timestamps between the datasets (the target reports lie "
              "outside the source or in source gaps wider than max_gap_s).")
        return track, reports, None, None

    return track, reports, source_aligned, target_aligned


def _drop_held_positions(track, source):
    """
    Sets lat, lon and geometric altitude to NaN where lat/lon repeat the previous
    position row (a GNSS fix held in rows of a higher rate).
    """
    position = track[[source.LAT, source.LON]].dropna()
    held = position.index[(position.diff() == 0).all(axis=1)]
    columns = [c for c in (source.LAT, source.LON, source.GEOALT) if c in track.columns]
    track = track.copy()
    track.loc[held, columns] = float("nan")
    return track


def _target_grid(reports, target):
    """
    The reports on a 1 s grid: the filtered columns (filtering.filtered_columns)
    interpolated, other columns forward-filled. No NaN values, as the Kalman
    smoother would propagate them.
    """
    import pandas as pd

    from qualifly_lib.filtering import filtered_columns

    grid = pd.date_range(reports.index[0].ceil(_GRID_STEP), reports.index[-1].floor(_GRID_STEP),
                         freq=_GRID_STEP, name="timestamp")
    on_grid = reports.ffill().bfill().reindex(grid, method="ffill")
    for column in filtered_columns(target, reports):
        if reports[column].notna().any():
            on_grid[column] = pd.Series(_interp_at(reports[column], grid, float("inf")), index=grid).ffill().bfill()
    return on_grid


def _at_reports(filtered, grid, reports, target):
    """
    A filter variant at the report times: the filtered grid interpolated to the
    reports. Reports before the first or after the last grid second (under 1 s)
    take the interpolated correction of the filter instead. A report missing a
    value keeps it missing.
    """
    import numpy as np

    from qualifly_lib.filtering import filtered_columns

    variant = reports.copy()
    t0 = reports.index[0]
    t = _seconds(reports.index, t0)
    for column in filtered_columns(target, reports):
        values = filtered[column].dropna() if column in filtered.columns else ()
        if len(values) == 0:
            variant[column] = np.nan
            continue
        ts = _seconds(values.index, t0)
        correction = (values - grid[column].loc[values.index]).to_numpy()
        inside = (t >= ts[0]) & (t <= ts[-1])
        at_reports = np.where(inside, np.interp(t, ts, values.to_numpy()),
                              reports[column].to_numpy(dtype=float) + np.interp(t, ts, correction))
        variant[column] = np.where(reports[column].notna(), at_reports, np.nan)
    return variant
