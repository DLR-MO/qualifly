# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

"""
Shared core of the local file data fetchers `local_file_in` and `local_public_ref`.
Format: see README section "Local data format".
"""

from pathlib import Path

from qualifly_lib.utils import (
    filter_rows,
    map_to_config_columns,
    missing_columns,
)

FILE_SUFFIXES = (".parquet", ".pq", ".csv")
META_SUFFIX = ".meta"


def resolve_flight_file(path, collection_id):
    """Returns {path}/{collection_id}.{parquet|pq|csv} or None if none exists."""
    base = Path(path)
    for suffix in FILE_SUFFIXES:
        candidate = base / f"{collection_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def resolve_flight_meta(path, collection_id):
    """Returns {path}/{collection_id}.meta (YAML flight metadata) or None."""
    candidate = Path(path) / f"{collection_id}{META_SUFFIX}"
    return candidate if candidate.is_file() else None


def get_flight_info(collection_id, options=None):
    """Returns (icao_hex, start, stop) from the optional {path}/{collection_id}.meta sidecar, else None."""
    path = (options or {}).get("path")
    meta_file = resolve_flight_meta(path, collection_id) if path else None
    return read_flight_info(meta_file) if meta_file else None


def read_flight_info(meta_file):
    """Reads icao_hex/start/stop from a .meta YAML file; None if any is missing or unparseable."""
    import yaml

    meta = yaml.safe_load(Path(meta_file).read_text(encoding="utf-8")) or {}
    icao_hex = meta.get("icao_hex")
    start, stop = _epoch_seconds(meta.get("start")), _epoch_seconds(meta.get("stop"))
    if not icao_hex or start is None or stop is None:
        return None
    return str(icao_hex), start, stop


def _epoch_seconds(value):
    """Converts a .meta timestamp (epoch seconds or datetime, naive = UTC) to int unix seconds; None if invalid."""
    import pandas as pd

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if timestamp is pd.NaT:
        return None
    if timestamp.tz is None:
        timestamp = timestamp.tz_localize("UTC")
    return int(timestamp.timestamp())


def read_dataframe(path):
    """Reads a local flight file; parquet/.pq is the primary format, CSV the fallback."""
    import pandas as pd

    path = Path(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def fetch_local_file(collection_id, options, config):
    """
    Reads one local flight file from options["path"], applies options["filter"] and maps it to the
    column names of `config`. Returns None when no file exists.
    """
    options = options or {}
    identifier = str(config.IDENTIFIER)

    path = options.get("path")
    if not path:
        raise ValueError(
            f"'{identifier}' requires 'path' in config.yaml "
            f"(data_fetching.{identifier}.path)"
        )

    file = resolve_flight_file(path, collection_id)
    if file is None:
        print(
            f"No {identifier} file for {collection_id} in {path} "
            f"(expected {collection_id} with suffix {' or '.join(FILE_SUFFIXES)})"
        )
        return None

    print(f"Loading {identifier} data of {collection_id} from {file}...")
    df = read_dataframe(file)

    if options.get("filter"):
        df = filter_rows(df, options["filter"])
        if df.empty:
            print(f"All rows of {collection_id} were removed by the {identifier} filter.")
            return None

    df = map_to_config_columns(df, config, options)

    missing = missing_columns(df, config)
    if missing:
        print(
            f"Warning: {identifier} file {file} is missing required columns {missing} "
            f"(after name mapping) - the flight will be skipped by the pipeline."
        )
    return df
