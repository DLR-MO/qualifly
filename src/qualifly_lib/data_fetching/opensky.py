# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import threading

import pandas as pd
from enum import StrEnum

from qualifly_lib.utils import (
    atomic_path,
    filter_rows,
    mark_not_found,
    prefer_position_time,
    to_epoch_seconds,
)

# one trino query at a time per process.
_TRINO_LOCK = threading.Lock()


# Predefined parameters for comparison
class Config(StrEnum):
    IDENTIFIER = "opensky"
    TIME = "time"
    LAT = "lat"
    LON = "lon"
    GEOALT = "geoaltitude"
    BAROALT = "baroaltitude"
    POSITIONNAC = "positionnac"
    GVA = "geometricverticalaccuracy"
    GROUNDSPEED = "velocity"
    NACV = "nacv"


def _query_with_retry(query_fn, label, attempts=3):
    """
    Runs an OpenSky Trino query, with retry.
    Returns None when every attempt fails.
    """
    for attempt in range(1, attempts + 1):
        try:
            with _TRINO_LOCK:
                return query_fn()
        except AttributeError as exc:
            print(f"OpenSky {label} attempt {attempt}/{attempts} failed: {exc}")
    print(f"OpenSky {label} unavailable after {attempts} attempts.")
    return None


def fetch_data(collection_id, base_path, options=None, flight_info=None):
    """
    Backend entry point (see data_fetcher.use_local_or_fetch_data).
    Needs `flight_info` (icao_hex, start, stop) from the source backend; `options["filter"]` filters the flightlist.
    """
    if flight_info is None:
        print(f"No flight information for {collection_id} - the opensky backend "
              f"needs the flight identity of the source data (e.g. a .meta sidecar).")
        mark_not_found(base_path, collection_id, Config.IDENTIFIER,
                       "flight information not available")
        return None

    icao_hex, start, end = flight_info
    return fetch_opensky_data(icao_hex, start, end, collection_id, base_path,
                              filters=(options or {}).get("filter") or {})


def merge_nacp(opensky_df, raw_df):
    """
    Attaches the NACp, NACv and GVA of the latest airborne operational status
    message (at most 15 s earlier) to each state vector; leading rows take the
    first values. Skips gracefully when the raw data carries none of the claims.
    """
    if raw_df is None or raw_df.empty:
        return opensky_df
    time = str(Config.TIME)
    claims = [column for column in (Config.POSITIONNAC, Config.GVA, Config.NACV)
              if column in raw_df.columns]
    if not claims:
        return opensky_df
    if "subtypecode" in raw_df.columns:
        # Surface messages (subtype 1) use another layout; their decoded claims
        # are meaningless for the airborne flight (e.g. NACp 3, GVA -1).
        raw_df = raw_df[raw_df["subtypecode"] == 0]
        if raw_df.empty:
            return opensky_df
    raw_df = raw_df.assign(**{time: to_epoch_seconds(raw_df["mintime"])}).sort_values(time)
    # pyopensky returns pyarrow columns; a claim missing from the whole flight is
    # then of the pyarrow null type, and the masked fill below aborts the process
    # in pyarrow (no Python exception). The claims are integers.
    raw_df = raw_df.astype(dict.fromkeys(claims, "Int64"))
    merged = pd.merge_asof(
        opensky_df,
        raw_df[[time, *claims]],
        on=time,
        direction="backward",
        tolerance=15.0,
    )
    leading = merged[time] < raw_df[time].iloc[0]
    merged.loc[leading, claims] = merged.loc[leading, claims].fillna(raw_df[claims].iloc[0])
    return merged


def best_matching_flight(flight_list, start, stop):
    """
    The flightlist row overlapping [start, stop] the most; None when no row
    overlaps.
    """
    first = to_epoch_seconds(flight_list["firstseen"])
    last = to_epoch_seconds(flight_list["lastseen"])
    start, stop = float(start), float(stop)
    overlap = last.clip(upper=stop) - first.clip(lower=start)
    if not (overlap > 0).any():
        return None
    return flight_list.loc[overlap.idxmax()]


def fetch_opensky_data(icao_hex, start, stop, collection_id, base_path, filters=None):
    """
    Retrieves opensky parameters from state_vector table and raw data for NACp values.
    `filters` restricts matching flightlist columns (e.g. {callsign: DLR12}).
    Saves the data in a CSV file within {base_path}/{collection_id}/opensky.csv
    """
    from pyopensky.trino import Trino
    from pyopensky.schema import OperationalStatusData4

    # Initialize Clients
    client = Trino()

    # Searches for flight with start < lastseen < stop
    flight_list = _query_with_retry(
        lambda: client.flightlist(
            start=start,
            stop=stop,
            icao24=icao_hex),
        "flightlist")

    # If opensky data has more data throughout the end of the flight than the flight list
    if flight_list is None:
        flight_list = _query_with_retry(
            lambda: client.flightlist(
                start=start,
                icao24=icao_hex),
            "flightlist (fallthrough)")
        if flight_list is None:
            mark_not_found(base_path, collection_id, Config.IDENTIFIER,
                           "flight not found in OpenSky")
            return None

    if filters:
        matched = {key: value for key, value in filters.items() if key in flight_list.columns}
        if matched:
            flight_list = filter_rows(flight_list, matched)
            if flight_list.empty:
                print(f"No OpenSky flight of the timeframe matches the filter {matched}.")
                return None

    flight = best_matching_flight(flight_list, start, stop)
    if flight is None:
        mark_not_found(base_path, collection_id, Config.IDENTIFIER,
                       "no OpenSky flight overlaps the source timeframe")
        return None

    # Get data
    opensky_df = _query_with_retry(
        lambda: client.history(
            start=flight.firstseen,
            stop=flight.lastseen,
            icao24=icao_hex,
        ),
        "history")

    if opensky_df is None or opensky_df.empty:
        mark_not_found(base_path, collection_id, Config.IDENTIFIER,
                       "no data found in OpenSky for the flight timeframe")
        return None

    # Position-fix time (lastposupdate) as row time
    opensky_df[Config.TIME] = to_epoch_seconds(opensky_df[Config.TIME])
    opensky_df = prefer_position_time(opensky_df, str(Config.TIME))
    opensky_df = opensky_df.sort_values(Config.TIME).reset_index(drop=True)

    # Get raw data with nac score, the vertical accuracy claim and the NACv
    # velocity accuracy claim of the operational status message
    raw_df = _query_with_retry(
        lambda: client.rawdata(
            start=flight.firstseen,
            stop=flight.lastseen,
            icao24=icao_hex,
            Table=OperationalStatusData4,
            extra_columns=(OperationalStatusData4.positionnac,
                           OperationalStatusData4.geometricverticalaccuracy,
                           OperationalStatusData4.nacv,
                           OperationalStatusData4.subtypecode),
        ),
        "rawdata")

    opensky_df = merge_nacp(opensky_df, raw_df)

    with atomic_path(f"{base_path}/{collection_id}/opensky.csv") as tmp:
        opensky_df.to_csv(tmp, index=False)

    return opensky_df
