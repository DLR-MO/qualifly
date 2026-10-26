# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

"""
Filter variants of the target positions, barometric altitude and ground speed.

`mean` and `kalman_smoother` run on a 1 s report grid, `median` on the reports
themselves (see docs/scores.md for the method and known flaws).
"""

import threading

import numpy as np
import pandas as pd

from qualifly_lib.error_calculation import _EARTH_RADIUS, _latlon_to_local_xy
from qualifly_lib.utils import M_TO_FT

_ROLLING_WINDOW = 5
# True track angle [deg] and vertical rate [m/s] of the OSN state vectors; the
# ground speed column is resolved from the target config (the OSN `velocity`)
_OTHER_VELOCITY_COLUMNS = ("heading", "vertrate")
_MPS_TO_KT = 3600 / 1852
# traffic's Kalman filters keep their state in a dict shared by all instances
# (TrackVariable.tracked_variables): smoothers run in parallel threads of one
# process would overwrite each other. run.py --jobs uses processes, each with
# its own state; the lock guards library use from threads.
_KALMAN_LOCK = threading.Lock()


def _speed_column(target):
    """The ground speed column of the target data (the OSN state-vector velocity)."""
    return str(getattr(target, "GROUNDSPEED", "velocity"))


def filtered_columns(target, df):
    """
    The columns the filters smooth: lat, lon, geometric altitude and, where in
    df, the barometric altitude and the ground speed.
    """
    optional = (getattr(target, "BAROALT", None), _speed_column(target))
    return [target.LAT, target.LON, target.GEOALT, *(column for column in optional if column in df.columns)]


def use_filtering(df, target):
    """
    Applies the filters that run on the 1 s report grid to one dataframe.
    Returns a dict with filtered dataframes.
    """
    if len(df) == 0:
        raise ValueError("Got an empty DataFrame — cannot filter anything.")
    columns = filtered_columns(target, df)

    # the seconds at the edges, without a full window, keep their values
    rolling = df[columns].rolling(_ROLLING_WINDOW, center=True)
    results = {"mean": df.assign(**rolling.mean().fillna(df[columns]))}

    velocity = _speed_column(target)
    if not all(column in df.columns and df[column].notna().all()
               for column in (velocity, *_OTHER_VELOCITY_COLUMNS)):
        print(f"Skipping the kalman_smoother variant - "
              f"{', '.join((velocity, *_OTHER_VELOCITY_COLUMNS))} are not all in the data.")
        return results
    try:
        results["kalman_smoother"] = _kalman_smoother(df, target)
    except np.linalg.LinAlgError:
        # it derives its noise from the data: a constant column (e.g. vertrate in level flight) has none
        print("Skipping the kalman_smoother variant - traffic's KalmanSmoother6D failed on a constant column.")
    return results


def median_filter(reports, target):
    """
    Centred rolling median over _ROLLING_WINDOW reports, on the reports
    themselves, so a spike next to a gap is not turned into a ramp. Each column
    skips its own missing values; the edge reports keep their values.
    """
    return reports.assign(**{
        column: reports[column].dropna().rolling(_ROLLING_WINDOW, center=True).median()
        .reindex(reports.index).fillna(reports[column])
        for column in filtered_columns(target, reports)
    })


def _kalman_smoother(df, target):
    """
    traffic's KalmanSmoother6D on local x/y meters (its units: ft, kt, ft/min).
    The east velocity is scaled to the rate of x, and back for the smoothed
    ground speed. The barometric altitude takes a second run in place of the
    geometric one and stays unfiltered where that fails.
    """
    smoothed, lat0 = _kalman_run(df, target, target.GEOALT)
    smoothed_speed = smoothed["groundspeed"].to_numpy() / _MPS_TO_KT
    smoothed_track = np.radians(smoothed["track"].to_numpy())
    smoothed_lat = smoothed["y"].to_numpy() / _EARTH_RADIUS
    result = df.assign(**{
        target.LAT: np.degrees(smoothed_lat),
        target.LON: np.degrees(smoothed["x"].to_numpy() / (_EARTH_RADIUS * np.cos(lat0))),
        target.GEOALT: smoothed["altitude"].to_numpy() / M_TO_FT,
        _speed_column(target): np.hypot(smoothed_speed * np.sin(smoothed_track) * np.cos(smoothed_lat) / np.cos(lat0),
                                        smoothed_speed * np.cos(smoothed_track)),
    })
    baro = getattr(target, "BAROALT", None)
    if baro in df.columns and df[baro].notna().all():
        try:
            result[baro] = _kalman_run(df, target, baro)[0]["altitude"].to_numpy() / M_TO_FT
        except np.linalg.LinAlgError:
            print("kalman_smoother keeps the barometric altitude unfiltered - traffic's KalmanSmoother6D failed on it.")
    return result


def _kalman_run(df, target, altitude):
    """traffic's KalmanSmoother6D output (its units) with `altitude` [m] as the altitude, and the latitude of x/y."""
    from traffic.algorithms.filters.kalman import KalmanSmoother6D

    lat = df[target.LAT].to_numpy(dtype=float)
    lat0 = np.radians(lat.mean())
    xy = _latlon_to_local_xy(lat, df[target.LON], lat0)
    speed = df[_speed_column(target)].to_numpy(dtype=float)
    track = np.radians(df["heading"].to_numpy(dtype=float))
    east = speed * np.sin(track) * np.cos(lat0) / np.cos(np.radians(lat))
    north = speed * np.cos(track)

    features = pd.DataFrame({
        "x": xy[:, 0],
        "y": xy[:, 1],
        "altitude": df[altitude].to_numpy(dtype=float) * M_TO_FT,
        "groundspeed": np.hypot(east, north) * _MPS_TO_KT,
        "track": np.degrees(np.arctan2(east, north)) % 360,
        "vertical_rate": df["vertrate"].to_numpy(dtype=float) * M_TO_FT * 60,
    })
    with _KALMAN_LOCK:
        smoothed = KalmanSmoother6D().apply(features)
    return smoothed, lat0
