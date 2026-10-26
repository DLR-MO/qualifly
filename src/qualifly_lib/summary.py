# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

"""
Per-flight result summaries: summary.pq (one row per flight) and scores.pq
(one row per flight and variant), upserted whenever a flight is processed.
"""

from __future__ import annotations

import importlib
import threading
from pathlib import Path

import pandas as pd

from qualifly_lib.accuracy_comparison import (
    ERROR_KINDS,
    NACP_REQUIRED_SHARE,
    nacp_fulfilment,
    velocity_fulfilment,
    vertical_fulfilment,
)
from qualifly_lib.flight_processor import _load_scores_csv
from qualifly_lib.utils import SCORE_NAMES, TIME_TOLERANCES, _find_source_column, atomic_path

# Serializes the read-modify-write of summary.pq / scores.pq for parallel flights
_UPSERT_LOCK = threading.Lock()

# The scores where lower is better: the signed ones are no error
_SIGNED_METRICS = {"time_offset_s", "alt_bias", "baro_bias", "gs_bias"}

# Suffix of the NACp fulfilment columns per error kind (see accuracy_comparison.ERROR_KINDS)
_NACP_SUFFIXES = {
    "position_error": "", "position_error_synced": "_synced",
    **{f"position_error_{name}": f"_{name}" for name in TIME_TOLERANCES},
    "track_distance": "_track",
}

# Columns of the per-variant claim fulfilment (percent of the reports) in scores.pq
FULFILMENT_COLUMNS = {
    **{f"nacp_fulfilled{suffix}": ERROR_KINDS[kind] for kind, suffix in _NACP_SUFFIXES.items()},
    # the altitude error against the VEPU of the claimed NACp and the claimed GVA
    "vepu_fulfilled": "altitude VEPU",
    "gva_fulfilled": "altitude GVA",
    # the ground speed error against the claimed NACv
    "nacv_fulfilled": "velocity NACv",
}

# Flight-level columns of summary.pq; the scores and claim fulfilment of each
# variant are in scores.pq
SUMMARY_COLUMNS = [
    "collection_id",
    "processed_at",
    "n_aligned",
    "overlap_s",
    "coverage",
    "max_report_gap_s",
    "p95_report_gap_s",
    "n_dropped",
    # per identifier: {id}_start, {id}_end, {id}_start_lat, {id}_start_lon,
    #                 {id}_end_lat, {id}_end_lon
]

SUMMARY_FILE = "summary.pq"
SCORES_FILE = "scores.pq"


def upsert_flight(base_path, collection_id, source_identifier, target_identifier):
    """
    Upserts the summary row and variant rows of one flight from its cached
    result files; flights without scores get a row with NaN metrics.
    Returns False if the flight directory does not exist.
    """
    base = Path(base_path)
    flight_dir = base / collection_id
    if not flight_dir.is_dir():
        return False
    processed_at = pd.Timestamp.now(tz="UTC")

    row = {"collection_id": collection_id, "processed_at": processed_at}
    validation_path = flight_dir / "epu_validation.csv"
    validation = pd.read_csv(validation_path) if validation_path.exists() else None
    if validation is not None:
        row["n_aligned"] = len(validation)
    row.update(_coverage_stats(flight_dir, source_identifier))
    row.update(_coverage_stats(flight_dir, target_identifier))
    row.update(_overlap_stats(flight_dir, source_identifier, target_identifier, validation))

    _upsert(base / SUMMARY_FILE, collection_id, pd.DataFrame([row]))

    # a flight without scores (any more) drops its variant rows
    scores = _load_scores_csv(flight_dir / "scores.csv") or {}
    scores_rows = [
        {
            "collection_id": collection_id,
            "variant": variant,
            "processed_at": processed_at,
            **{metric: scores[variant].get(metric) for metric in SCORE_NAMES},
            **_fulfilment(validation, variant),
        }
        for variant in scores
    ]
    _upsert(base / SCORES_FILE, collection_id, pd.DataFrame(scores_rows))
    return True


def export_summary(base_path, out_path=None):
    """
    Exports summary.pq joined with scores.pq (one row per flight and variant)
    to a CSV file (default {base_path}/summary.csv).
    Returns (path, number of flights in the summary, number of them with scores).
    """
    base = Path(base_path)
    pq_path = base / SUMMARY_FILE
    if not pq_path.exists():
        raise FileNotFoundError(f"No summary found at {pq_path}. Process a flight first.")

    df = pd.read_parquet(pq_path)
    n_flights = len(df)
    n_scored = 0
    if (base / SCORES_FILE).exists():
        scores = pd.read_parquet(base / SCORES_FILE).drop(columns="processed_at")
        df = df.merge(scores, on="collection_id", how="left")
        if "variant" in df.columns:
            n_scored = int(df.loc[df["variant"].notna(), "collection_id"].nunique())
    out = Path(out_path) if out_path else base / "summary.csv"
    with atomic_path(out) as tmp:
        df.to_csv(tmp, index=False)
    return out, n_flights, n_scored


def analyze_summary(base_path, source_identifier="local_file_in", target_identifier="local_public_ref"):
    """
    Prints a concise analysis of summary.pq and scores.pq: the scores and claim
    fulfilment per variant, and the data coverage.
    """
    base = Path(base_path)
    pq_path = base / SUMMARY_FILE
    if not pq_path.exists():
        raise FileNotFoundError(f"No summary found at {pq_path}. Process a flight first.")

    df = pd.read_parquet(pq_path)
    print("=" * 70)
    print("FLIGHT SUMMARY ANALYSIS")
    print("=" * 70)
    print(f"Flights: {len(df)}")

    if (base / SCORES_FILE).exists():
        _print_variant_tables(pd.read_parquet(base / SCORES_FILE))

    n_aligned = _numeric(df, "n_aligned")
    print(f"\nAligned timestamps per flight: median {n_aligned.median():.0f}  "
          f"(min {n_aligned.min():.0f}, max {n_aligned.max():.0f})")
    overlap, coverage = _numeric(df, "overlap_s"), _numeric(df, "coverage") * 100
    max_gap, dropped = _numeric(df, "max_report_gap_s"), _numeric(df, "n_dropped")
    print(f"Overlap per flight:            median {overlap.median():.0f} s  "
          f"| covered by reports: median {coverage.median():.1f}% (min {coverage.min():.1f}%)")
    print(f"Largest report gap per flight: median {max_gap.median():.1f} s (max {max_gap.max():.1f} s)  "
          f"| reports not compared: median {dropped.median():.0f} (max {dropped.max():.0f})")

    gap = _start_gap(df, source_identifier, target_identifier)
    valid_gap = gap.dropna()
    if not valid_gap.empty:
        n_after = (valid_gap > 0).sum()
        print(f"Start gap ({target_identifier} - {source_identifier}):  median {valid_gap.median():.0f} s  "
              f"| {target_identifier} starts after {source_identifier}: {n_after}/{len(valid_gap)}")


def _print_variant_tables(scores_df):
    """
    Prints per metric the median over the flights of every variant (with the
    flights where it beats original) and per claim the mean fulfilment, the
    number of flights carrying that claim and how many reach the required share.
    """
    variants = ["original", *sorted(set(scores_df["variant"]) - {"original"})]

    def per_variant(column):
        values = scores_df.assign(value=pd.to_numeric(scores_df[column], errors="coerce"))
        return values.pivot(index="collection_id", columns="variant", values="value").reindex(columns=variants)

    n_flights = scores_df["collection_id"].nunique()
    header = f"  {'':<20}" + "".join(f"{variant:>22}" for variant in variants)
    print(f"\nFlights with scores: {n_flights}")
    print("Scores per variant, median over flights [m, s, m/s] (flights better than original / compared):")
    print(header)
    for metric in SCORE_NAMES:
        if metric not in scores_df.columns:  # scores written before the metric existed
            continue
        values = per_variant(metric)
        label, digits = metric, 1
        if metric == "time_offset_s":  # its amount: signed offsets cancel in the median
            values, label, digits = values.abs(), "|time_offset_s|", 2
        cells = []
        for variant in variants:
            cell = f"{values[variant].median():.{digits}f}"
            if variant != "original" and metric not in _SIGNED_METRICS:
                compared = values[variant].notna() & values["original"].notna()
                better = (values[variant] < values["original"])[compared].sum()
                cell += f" ({better}/{compared.sum()})"
            cells.append(f"{cell:>22}")
        print(f"  {label:<20}" + "".join(cells))

    print(f"\nClaim coverage: {_claim_coverage(scores_df, n_flights)}")
    print(f"Claim fulfilment per variant, mean over flights with a claim [%] "
          f"(a flight fulfils a claim at >= {NACP_REQUIRED_SHARE:.0f}% of its reports; "
          f"cells: mean (met/with claim)):")
    print(header)
    for column, label in FULFILMENT_COLUMNS.items():
        if column not in scores_df.columns:
            continue
        values = per_variant(column)
        if values.isna().to_numpy().all():  # no flight with such a claim
            continue
        cells = []
        for variant in variants:
            series = values[variant]
            n_claims = int(series.notna().sum())
            if n_claims:
                met = int((series >= NACP_REQUIRED_SHARE).sum())
                cell = f"{series.mean():.1f} ({met}/{n_claims})"
            else:
                cell = "n/a"
            cells.append(f"{cell:>22}")
        print(f"  {label:<20}" + "".join(cells))


def _claim_coverage(scores_df, n_flights):
    """The flights carrying each claim, e.g. "NACp 58/97, VEPU (NACp 9-11) 58/97, ..."."""
    labels = {
        "nacp_fulfilled": "NACp",
        "vepu_fulfilled": "VEPU (NACp 9-11)",
        "gva_fulfilled": "GVA",
        "nacv_fulfilled": "NACv",
    }
    parts = []
    for column, label in labels.items():
        if column in scores_df.columns:
            n_claims = int(scores_df.loc[scores_df[column].notna(), "collection_id"].nunique())
            parts.append(f"{label} {n_claims}/{n_flights}")
    return ", ".join(parts) if parts else "none"


def _start_gap(df, source_identifier, target_identifier):
    columns = [f"{target_identifier}_start", f"{source_identifier}_start"]
    if not all(column in df.columns for column in columns):
        return pd.Series(dtype=float)
    start, source_start = (pd.to_datetime(df[column], errors="coerce", utc=True) for column in columns)
    return (start - source_start).dt.total_seconds()


def _numeric(df, column):
    """A summary column as numbers, all NaN if the summary has no such column."""
    if column not in df.columns:
        return pd.Series(float("nan"), index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def _upsert(pq_path, collection_id, new_rows):
    """
    Replaces the rows of collection_id with new_rows (none removes them).
    Thread-safe and crash-safe: the read-modify-write runs under a lock and the
    file is written to a temporary path first (see atomic_path).
    """
    with _UPSERT_LOCK:
        frames = [new_rows] if not new_rows.empty else []
        if pq_path.exists():
            existing = pd.read_parquet(pq_path)
            frames.insert(0, existing[existing["collection_id"] != collection_id])
        if frames:
            with atomic_path(pq_path) as tmp:
                pd.concat(frames, ignore_index=True).to_parquet(tmp, index=False)


def _fulfilment(validation, variant):
    """
    Share of reports fulfilling the NACp, altitude and velocity claims for one
    variant (FULFILMENT_COLUMNS) from an epu_validation.csv frame; empty without
    one.
    """
    if validation is None:
        return {}
    return {
        **{f"nacp_fulfilled{_NACP_SUFFIXES[kind]}": share for kind, share in nacp_fulfilment(validation, variant).items()},
        **{f"{kind}_fulfilled": share for kind, share in vertical_fulfilment(validation, variant).items()},
        **{f"{kind}_fulfilled": share for kind, share in velocity_fulfilment(validation, variant).items()},
    }


def _coverage_stats(flight_dir, identifier):
    """
    Start/end statistics from cached {identifier}.csv: first and last
    timestamp plus lat/lon at those points.
    """
    path = flight_dir / f"{identifier}.csv"
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    if "timestamp" not in df.columns or df.empty:
        return {}

    timestamps = _parse_timestamps(df["timestamp"])
    config = _config_class(identifier)
    lat_col = _find_source_column(df, "LAT", config, None)
    lon_col = _find_source_column(df, "LON", config, None)

    stats = {
        f"{identifier}_start": timestamps.iloc[0],
        f"{identifier}_end": timestamps.iloc[-1],
    }

    if lat_col and lon_col:
        located = df.dropna(subset=[lat_col, lon_col])
        if not located.empty:
            first, last = located.iloc[0], located.iloc[-1]
            stats.update({
                f"{identifier}_start_lat": first[lat_col], f"{identifier}_start_lon": first[lon_col],
                f"{identifier}_end_lat": last[lat_col], f"{identifier}_end_lon": last[lon_col],
            })
    return stats


def _overlap_stats(flight_dir, source_identifier, target_identifier, compared):
    """
    How far the compared reports (epu_validation.csv, None without) cover the time both
    sides have positions: overlap_s, coverage, max/p95_report_gap_s and n_dropped (see docs/scores.md).
    """
    source = _position_times(flight_dir, source_identifier)
    reports = _position_times(flight_dir, target_identifier)
    if source is None or reports is None:
        return {}
    start, end = max(source[0], reports[0]), min(source[-1], reports[-1])
    overlap_s = max((end - start).total_seconds(), 0.0)
    gaps = reports[(reports >= start) & (reports <= end)].to_series().diff().dt.total_seconds()
    stats = {"overlap_s": overlap_s, "max_report_gap_s": gaps.max(), "p95_report_gap_s": gaps.quantile(0.95)}

    if compared is not None:
        stats["n_dropped"] = len(reports) - len(compared)
        if overlap_s > 0 and "timestamp" in compared.columns:
            seconds = _parse_timestamps(compared["timestamp"]).dt.floor("s").nunique()
            stats["coverage"] = min(seconds / overlap_s, 1.0)
    return stats


def _position_times(flight_dir, identifier):
    """Sorted unique timestamps of the rows with lat/lon in the cached {identifier}.csv, None without."""
    path = flight_dir / f"{identifier}.csv"
    if not path.exists():
        return None
    header = pd.read_csv(path, nrows=0)
    config = _config_class(identifier)
    columns = ["timestamp", *(_find_source_column(header, attribute, config, None) for attribute in ("LAT", "LON"))]
    if None in columns or "timestamp" not in header.columns:
        return None
    df = pd.read_csv(path, usecols=columns).dropna()
    if df.empty:
        return None
    return pd.DatetimeIndex(_parse_timestamps(df["timestamp"])).unique().sort_values()


def _config_class(identifier):
    """Config class of a data source (e.g. local_file_in), None if unknown."""
    try:
        module = importlib.import_module(f"qualifly_lib.data_fetching.{identifier}")
        return getattr(module, "Config", None)
    except (ImportError, AttributeError):
        return None


def _parse_timestamps(series):
    try:
        return pd.to_datetime(series, utc=True, format="ISO8601")
    except (ValueError, TypeError):
        return pd.to_datetime(series, utc=True, format="mixed")
