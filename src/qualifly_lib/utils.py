# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import threading
from contextlib import contextmanager

# Serializes the shared log files and the summary roll-up within one run
_IO_LOCK = threading.Lock()


@contextmanager
def atomic_path(path):
    """
    Yields a temporary path to write to; on success it replaces `path` in one
    step (os.replace), so an interrupted or parallel run never sees a partial
    file. The temporary file is removed when the writer raises.
    """
    path = os.fspath(path)
    tmp = f"{path}.tmp"
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# Time tolerances (s) of the pos_p95_tol* scores (see docs/scores.md)
TIME_TOLERANCES = {"tol100ms": 0.1, "tol300ms": 0.3, "tol600ms": 0.6}

# Scores of every variant, in display order (see docs/scores.md)
SCORE_NAMES = (
    "pos_rmse", "pos_p95", "time_offset_s", "pos_rmse_synced", "pos_p95_synced",
    *(f"pos_p95_{name}" for name in TIME_TOLERANCES),
    "pos_track_mean", "pos_track_p95", "pos_track_max",
    "alt_bias", "alt_rmse", "alt_nmad", "alt_p95", "baro_bias", "baro_nmad",
    "gs_bias", "gs_rmse", "gs_nmad", "gs_p95",
)

M_TO_FT = 3.28084

# knots -> m/s, for the ground speed of a source reporting in kt
KT_TO_MS = 1852 / 3600

# Canonical internal column names; source names are configured per backend in
# config.yaml (data_fetching.{id}.columns / .sum_columns / .baro_params)
COLUMN_KEYS: dict[str, str] = {
    "TIME": "time",
    "LAT": "lat",
    "LON": "lon",
    "GEOALT": "geo_alt",
    "BAROALT": "baro_alt",
    "POSITIONNAC": "position_nac",
    "GVA": "gva",
    "GROUNDSPEED": "ground_speed",
    "NACV": "nacv",
}

# Canonical name -> Config attribute, to resolve a sum_columns key
COLUMN_ATTRIBUTES = {canonical: attribute for attribute, canonical in COLUMN_KEYS.items()}

# Generic source column spellings accepted besides the configured name; data-set
# specific names belong in config.yaml
SOURCE_ALIASES: dict[str, tuple[str, ...]] = {
    "TIME": ("time", "timestamp", "Time"),
    "LAT": ("lat", "latitude"),
    "LON": ("lon", "longitude"),
    "GEOALT": ("geo_alt", "geoaltitude", "altitude"),
    "BAROALT": ("baro_alt", "baroaltitude"),
    "POSITIONNAC": ("position_nac", "positionnac"),
    "GVA": ("gva", "geometricverticalaccuracy"),
    "GROUNDSPEED": ("ground_speed", "groundspeed", "velocity", "gs"),
    "NACV": ("nacv", "nac_v"),
}

REQUIRED_ATTRIBUTES = ("TIME", "LAT", "LON", "GEOALT")


def prepare_baro_altitude(df, target="baro_alt", baro_params=()):
    """
    Converts the raw barometric altitude columns in `baro_params` (ft -> m) and
    merges them into `target` as their row-wise mean. No-op if `target` exists
    or none of the columns is present.
    """
    if target in df.columns:
        return df
    baro_params = [col for col in baro_params if col in df.columns]
    if not baro_params:
        return df
    df = df.copy()
    df[baro_params] = df[baro_params] / M_TO_FT
    df[target] = df[baro_params].mean(axis=1)
    return df


def source_names(value):
    """
    The alternative names of one configured source parameter: `value` is a
    single name or a list of names (data-set renamings over time), of which the
    first one present is used (see pick_source_name).
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(name) for name in value]


def pick_source_name(value, present):
    """
    The first of the configured alternative names contained in `present` (a
    column index or any name collection), or None when none of them exists.
    """
    for name in source_names(value):
        if name in present:
            return name
    return None


def sum_source_groups(sources):
    """
    The components summed onto one `sum_columns` column as a list of
    alternative-name groups: a name or a flat list of names is one component
    (its first present name is used). To add several parameters to one column,
    give one group per component, e.g. [["fineA", "fineAo"], "fineB"].
    """
    groups = source_names(sources)
    if isinstance(sources, (list, tuple)) and any(
        isinstance(item, (list, tuple)) for item in sources
    ):
        return [source_names(item) for item in sources]
    return [groups] if groups else []


def add_sum_columns(df, target, sources):
    """
    Adds the source parameters of `sources` onto `target` (options["sum_columns"]).
    Each component contributes its last known value, so a slowly refreshed word
    combines with the samples of its fine part. A component is one name or a list
    of alternative names (first one present wins); several components need one
    group each. The summed sources and unused alternatives are dropped, so a
    second pass over the result cannot add them twice.
    """
    target = str(target)
    if target not in df.columns:
        return df
    groups = sum_source_groups(sources)
    summed = [
        name for name in (pick_source_name(group, df.columns) for group in groups)
        if name is not None and name != target
    ]
    if not summed:
        return df
    df = df.copy()
    total = df[target].ffill()
    for name in summed:
        total = total + df[name].ffill()
    df[target] = total
    unused = [name for group in groups for name in group]
    return df.drop(columns=[name for name in unused if name in df.columns and name != target])


def to_epoch_seconds(series):
    """
    Normalizes a time column (epoch seconds or datetimes) to epoch seconds.
    ISO 8601 is parsed with the format pinned, since format inference turns
    values with another fractional-second precision into NaT.
    """
    import pandas as pd

    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    # dtype kind also catches the pyarrow timestamps pyopensky returns
    if series.dtype.kind == "M":
        datetimes = pd.to_datetime(series, utc=True, errors="coerce")
    else:
        numeric = pd.to_numeric(series, errors="coerce")
        if int(numeric.notna().sum()) == int(series.notna().sum()):
            return numeric.astype(float)
        datetimes = pd.to_datetime(series, utc=True, errors="coerce", format="ISO8601")
        if int(datetimes.notna().sum()) < int(series.notna().sum()):
            datetimes = pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    return (datetimes - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().astype(float)


def _find_source_column(df, attribute, config, options):
    """
    First present source column for a Config attribute (the configured name or
    its alternatives first).
    """
    columns = (options or {}).get("columns") or {}
    candidates = (
        *source_names(columns.get(COLUMN_KEYS.get(attribute))),
        str(getattr(config, attribute)) if hasattr(config, attribute) else None,
        *SOURCE_ALIASES.get(attribute, ()),
    )
    for candidate in candidates:
        if candidate is not None and candidate in df.columns:
            return candidate
    return None


def map_to_config_columns(df, config, options=None):
    """
    Renames the source columns onto the `config` names, sums
    `options["sum_columns"]`, builds the barometric altitude and normalizes the
    time column to epoch seconds. `geo_alt_is_ft` and `ground_speed_is_kt`
    convert those source columns before the rename, so a later pass over the
    converted data does not convert them twice.
    """
    options = options or {}
    df = df.copy()
    if hasattr(config, "GEOALT") and options.get("geo_alt_is_ft"):
        geo_alt_source = pick_source_name(
            (options.get("columns") or {}).get(COLUMN_KEYS["GEOALT"]), df.columns)
        geo_alt_target = str(config.GEOALT)
        # a source already named like the canonical column is the meters schema
        if geo_alt_source is not None and geo_alt_source != geo_alt_target:
            df[geo_alt_source] = df[geo_alt_source] / M_TO_FT
    if hasattr(config, "GROUNDSPEED") and options.get("ground_speed_is_kt"):
        gs_source = pick_source_name(
            (options.get("columns") or {}).get(COLUMN_KEYS["GROUNDSPEED"]), df.columns)
        gs_target = str(config.GROUNDSPEED)
        # a source already named like the canonical column is the m/s schema
        if gs_source is not None and gs_source != gs_target:
            df[gs_source] = df[gs_source] * KT_TO_MS
    rename = {}
    for attribute in ("TIME", "LAT", "LON", "GEOALT", "POSITIONNAC", "GVA", "GROUNDSPEED", "NACV"):
        if not hasattr(config, attribute):
            continue
        source_column = _find_source_column(df, attribute, config, options)
        if source_column is None:
            continue
        target = str(getattr(config, attribute))
        if target in df.columns and source_column != target:
            print(
                f"Warning: ignoring column '{source_column}' for {attribute} - "
                f"'{target}' is already present."
            )
            continue
        rename[source_column] = target
    df = df.rename(columns=rename)

    for canonical, sources in ((options or {}).get("sum_columns") or {}).items():
        attribute = COLUMN_ATTRIBUTES.get(canonical)
        target = str(getattr(config, attribute)) if attribute and hasattr(config, attribute) \
            else str(canonical)
        df = add_sum_columns(df, target, sources)

    if hasattr(config, "BAROALT"):
        target = str(config.BAROALT)
        baro_params = (options or {}).get("baro_params") or ()
        metric_column = _find_source_column(df, "BAROALT", config, options)
        if metric_column is not None and metric_column not in baro_params:
            # Ready metric column: do not convert the raw parameters a second time
            df = df.rename(columns={metric_column: target})
        else:
            df = prepare_baro_altitude(df, target=target, baro_params=baro_params)

    if hasattr(config, "TIME") and str(config.TIME) in df.columns:
        df[str(config.TIME)] = to_epoch_seconds(df[str(config.TIME)])
        df = prefer_position_time(df, str(config.TIME))
    return df


def prefer_position_time(df, time_col="time"):
    """
    Uses the OSN position-fix time `lastposupdate` as row time where present.
    No-op without that column.
    """
    if "lastposupdate" in df.columns:
        df[time_col] = to_epoch_seconds(df["lastposupdate"]).fillna(to_epoch_seconds(df[time_col]))
    return df


def missing_columns(df, config):
    """Returns the required Config columns missing from df after mapping."""
    return [
        str(getattr(config, attribute))
        for attribute in REQUIRED_ATTRIBUTES
        if not hasattr(config, attribute) or str(getattr(config, attribute)) not in df.columns
    ]


def geoid_undulation(lat, lon):
    """EGM96 geoid undulation N (m) at lat/lon (deg): h_ellipsoid = H_msl + N."""
    try:
        import egm96
    except ImportError as e:
        raise ImportError("geo_alt_is_msl needs the 'egm96' package: uv sync") from e

    return egm96.undulation(lat, lon)


def msl_to_ellipsoid_height(df, lat_col, lon_col, alt_col, label=None):
    """
    Returns a copy of `df` with the MSL heights in `alt_col` converted to WGS-84
    ellipsoidal heights; rows without a position use the neighbouring one.
    """
    import numpy as np

    if any(column not in df.columns for column in (lat_col, lon_col, alt_col)):
        return df
    position = df[[lat_col, lon_col]].ffill().bfill()
    lat = position[lat_col].to_numpy(dtype=float)
    lon = position[lon_col].to_numpy(dtype=float)
    located = np.isfinite(lat) & np.isfinite(lon)
    undulation = np.full(len(df), np.nan)
    if located.any():
        undulation[located] = geoid_undulation(lat[located], lon[located])
    undulation[~np.isfinite(undulation)] = np.nan

    if label and located.any():
        print(f"Converting {label} geometric altitude from MSL to ellipsoidal height "
              f"(EGM96, median undulation {np.nanmedian(undulation):+.1f} m).")
    df = df.copy()
    df[alt_col] = df[alt_col].to_numpy(dtype=float) + undulation
    return df


def filter_rows(df, filters):
    """
    Keeps the rows matching every key/value pair of `filters` (the per-backend
    `filter` of the config); keys without a column are ignored.
    """
    if not filters:
        return df
    import pandas as pd

    mask = pd.Series(True, index=df.index)
    for key, value in filters.items():
        if key not in df.columns:
            print(f"Warning: filter key '{key}' has no matching column - ignoring it.")
            continue
        mask &= df[key] == value
    return df[mask].copy()


def log_failed_ids(filepath, collection_id):
    """
    Log collection_id in a file saved as filepath if it is not already there.
    The read-check-append runs under a lock, so parallel flights never
    duplicate an id or interleave their lines.
    """
    with _IO_LOCK:
        existing_ids = set()
        if os.path.exists(filepath):
            with open(filepath, "r") as f_read:
                existing_ids = {line.strip() for line in f_read}
        if str(collection_id) not in existing_ids:
            with open(filepath, "a") as f_write:
                f_write.write(f"{collection_id}\n")
    return None


def mark_not_found(base_path, collection_id, identifier, reason):
    """
    Writes {identifier}_not_found.txt so reruns skip re-fetching the data.
    Delete the marker to refetch.
    """
    filepath = os.path.join(base_path, collection_id, f"{identifier}_not_found.txt")
    with atomic_path(filepath) as tmp:
        with open(tmp, "w", encoding="utf-8") as f_write:
            f_write.write(f"{reason}\n")
    return None


def open_in_browser(file_path, timeout=30):
    """
    Opens an HTML file in the default browser, served via a one-shot local HTTP server.
    Map tiles (e.g. OpenStreetMap) do not load from file:// URLs.
    """
    import webbrowser
    from http.server import BaseHTTPRequestHandler, HTTPServer

    with open(file_path, "rb") as f:
        html = f.read()

    class OneShotRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            buffer_size = 1024 * 1024
            for i in range(0, len(html), buffer_size):
                self.wfile.write(html[i:i + buffer_size])

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), OneShotRequestHandler)
    server.timeout = timeout
    webbrowser.open(f"http://127.0.0.1:{server.server_port}")
    server.handle_request()
    server.server_close()
