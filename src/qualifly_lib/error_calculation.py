# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

import numpy as np

from qualifly_lib.utils import SCORE_NAMES, TIME_TOLERANCES

_EARTH_RADIUS = 6371000  # Earth radius in meters
_NMAD_SCALE_FACTOR = 1.4826  # scales MAD to be comparable to a std-dev under a normal-error assumption

# Search window around a report for the track distance (seconds): the maximum
# ADS-B position latency (14 CFR 91.227)
DEFAULT_TRACK_WINDOW_S = 2.0
# Largest gap of a reference column still interpolated across (seconds)
DEFAULT_MAX_GAP_S = 2.0

# Range (+-) and resolution of the time offset search (seconds)
_OFFSET_RANGE_S = 1.0
_OFFSET_STEP_S = 0.005


def _index_seconds(index):
    """Index as float seconds (UTC epoch); non-datetime indexes as 1 s-spaced positions."""
    import pandas as pd

    if len(index) == 0:
        return np.zeros(0)
    if pd.api.types.is_datetime64_any_dtype(index):
        return np.asarray(pd.DatetimeIndex(index).asi8, dtype=float) / 1e9
    return np.arange(len(index), dtype=float)


def interp_with_gaps(t_samples, values, t_query, max_gap_s):
    """
    Linearly interpolates `values` at the ascending `t_samples` onto `t_query`;
    NaN outside the samples and across gaps wider than max_gap_s.
    """
    t_query = np.asarray(t_query, dtype=float)
    if len(t_samples) == 0:
        return np.full(len(t_query), np.nan)
    after = np.searchsorted(t_samples, t_query)  # first sample at or after t
    before = np.maximum(after - 1, 0)
    after = np.minimum(after, len(t_samples) - 1)
    inside = (t_query >= t_samples[0]) & (t_query <= t_samples[-1])
    on_sample = t_query == t_samples[after]
    bridged = t_samples[after] - t_samples[before] <= max_gap_s
    return np.where(inside & (on_sample | bridged), np.interp(t_query, t_samples, values), np.nan)


def _track_distance(points, t_points, track, t_track, window_s):
    """
    Shortest distance from each point to the polyline `track` (local meters),
    searching only the part flown within +-window_s of the point's time, so a
    point is not matched to another pass. NaN where the window holds no segment.
    """
    distances = np.full(len(points), np.nan)
    for k, (point, t) in enumerate(zip(points, t_points)):
        piece = _track_piece(track, t_track, t - window_s, t + window_s)
        if len(piece) > 1:
            distances[k] = _distance_to_polyline(point, piece)
    return distances


def _track_piece(track, t_track, start, end):
    """The part of `track` flown between start and end, its ends interpolated."""
    start, end = max(start, t_track[0]), min(end, t_track[-1])
    if start >= end:
        return track[:0]
    inner = slice(np.searchsorted(t_track, start, side="right"), np.searchsorted(t_track, end))
    times = np.concatenate([[start], t_track[inner], [end]])
    return np.column_stack([np.interp(times, t_track, track[:, 0]), np.interp(times, t_track, track[:, 1])])


def _distance_to_polyline(point, nodes):
    """Shortest distance from `point` to the polyline through `nodes`."""
    start, step = nodes[:-1], np.diff(nodes, axis=0)
    # position of the closest point on each segment: 0 = its start, 1 = its end
    along = ((point - start) * step).sum(axis=1) / np.maximum((step ** 2).sum(axis=1), 1e-12)
    closest = start + np.clip(along, 0.0, 1.0)[:, None] * step
    return np.hypot(*(point - closest).T).min()


def calculate_errors(source_df, all_target_dfs, columns_config, source_track=None):
    """
    Scores every target variant against the source at the report times (see
    docs/scores.md).

    Returns (scores, all_errors): the SCORE_NAMES per variant and the per-report
    errors of each variant.
    """
    if source_df.empty or any(df.empty for df in all_target_dfs.values()):
        print("Something went wrong. No data found.")
        return {variant: dict.fromkeys(SCORE_NAMES) for variant in all_target_dfs}, {}

    track = source_df if source_track is None else source_track
    scores = {}
    all_errors = {}
    for variant, df in all_target_dfs.items():
        scores[variant], all_errors[variant] = _calculate_errors_per_df(
            source_df, df, columns_config, track, label=variant)
    return scores, all_errors


def _calculate_errors_per_df(source_df, target_df, config, track_df, label=None):
    """
    Scores of one target variant (see calculate_errors), NaN without valid data;
    `track_df` is the reference track. Returns (scores, per-report errors).
    """
    import pandas as pd

    source = config["source"]
    target = config["target"]
    window_s = config.get("track_window_s", DEFAULT_TRACK_WINDOW_S)

    # Position error via haversine in meters
    position_errors = haversine_distance(
        source_df[source.LAT], source_df[source.LON],
        target_df[target.LAT], target_df[target.LON]
    )

    track_xy_df = track_df.dropna(subset=[source.LAT, source.LON])
    points_df = target_df.dropna(subset=[target.LAT, target.LON])
    track_distances = pd.Series(np.nan, index=target_df.index)
    synced_errors = pd.Series(np.nan, index=target_df.index)
    tolerance_errors = {name: pd.Series(np.nan, index=target_df.index) for name in TIME_TOLERANCES}
    time_offset = np.nan
    if not points_df.empty and len(track_xy_df) > 1:
        t_points, t_track = _index_seconds(points_df.index), _index_seconds(track_xy_df.index)
        # One reference latitude for both curves, or their east coordinates drift apart
        lat0 = np.radians(track_xy_df[source.LAT].to_numpy(dtype=float).mean())
        points_xy = _latlon_to_local_xy(points_df[target.LAT], points_df[target.LON], lat0)
        track_xy = _latlon_to_local_xy(track_xy_df[source.LAT], track_xy_df[source.LON], lat0)
        track_distances[points_df.index] = _track_distance(points_xy, t_points, track_xy, t_track, window_s)
        time_offset, synced_errors[points_df.index] = _time_offset(
            points_df[target.LAT], points_df[target.LON], t_points,
            track_xy_df[source.LAT], track_xy_df[source.LON], t_track, _OFFSET_RANGE_S,
        )
        # A time error around the report time forgiven up to each tolerance
        for name, tolerance_s in TIME_TOLERANCES.items():
            tolerance_errors[name][points_df.index] = _track_distance(
                points_xy, t_points, track_xy, t_track, tolerance_s)

    # Altitude errors at the report time
    max_gap_s = config.get("max_gap_s", DEFAULT_MAX_GAP_S)
    altitude_errors = _altitude_errors(track_df, target_df, source.GEOALT, target.GEOALT, max_gap_s)
    speed_errors = _altitude_errors(track_df, target_df, getattr(source, "GROUNDSPEED", None),
                                    getattr(target, "GROUNDSPEED", None), max_gap_s)
    baro_errors = _altitude_errors(track_df, target_df, getattr(source, "BAROALT", None),
                                   getattr(target, "BAROALT", None), max_gap_s)

    if label is not None:
        print(f"Deviation calculated for {label} data")
        if not position_errors.notna().any():
            print(f"Warning: no valid positions for {label} data - its metrics are NaN.")

    errors = pd.DataFrame({
        "position_error": position_errors,
        "position_error_synced": synced_errors,
        **{f"position_error_{name}": distances for name, distances in tolerance_errors.items()},
        "track_distance": track_distances,
        "altitude_error": altitude_errors,
        "baro_error": baro_errors,
        "groundspeed_error": speed_errors,
    })
    return {
        "pos_rmse": _rms(errors["position_error"]),
        "pos_p95": _p95(errors["position_error"]),
        "time_offset_s": time_offset,
        "pos_rmse_synced": _rms(errors["position_error_synced"]),
        "pos_p95_synced": _p95(errors["position_error_synced"]),
        **{f"pos_p95_{name}": _p95(errors[f"position_error_{name}"]) for name in TIME_TOLERANCES},
        "pos_track_mean": float(errors["track_distance"].mean()),
        "pos_track_p95": _p95(errors["track_distance"]),
        "pos_track_max": float(errors["track_distance"].max()),
        "alt_bias": float(altitude_errors.median()),
        "alt_rmse": _rms(altitude_errors),
        "alt_nmad": _normalized_mad(altitude_errors),
        "alt_p95": _p95(altitude_errors.abs()),
        "baro_bias": float(baro_errors.median()),
        "baro_nmad": _normalized_mad(baro_errors),
        "gs_bias": float(speed_errors.median()),
        "gs_rmse": _rms(speed_errors),
        "gs_nmad": _normalized_mad(speed_errors),
        "gs_p95": _p95(speed_errors.abs()),
    }, errors


def _finite(errors):
    errors = np.asarray(errors, dtype=float)
    return errors[np.isfinite(errors)]


def _rms(errors):
    errors = _finite(errors)
    return float(np.sqrt(np.mean(errors ** 2))) if errors.size else np.nan


def _p95(errors):
    errors = _finite(errors)
    return float(np.percentile(errors, 95)) if errors.size else np.nan


def _offset_scan(range_s):
    """The time offsets tried: -range_s to +range_s in _OFFSET_STEP_S steps."""
    return np.round(np.arange(-range_s, range_s + _OFFSET_STEP_S / 2, _OFFSET_STEP_S), 6)


def _time_offset(lat, lon, t_points, track_lat, track_lon, t_track, range_s):
    """
    Constant time offset of the reports (positive = stamped late): the shift
    within +-range_s minimizing the median distance to the reference at
    (report time - shift). Returns (offset, per-report distances), NaN without
    reports whose whole range lies inside the track.
    """
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    track_lat, track_lon = np.asarray(track_lat, dtype=float), np.asarray(track_lon, dtype=float)

    def distances(shift):
        return haversine_distance(np.interp(t_points - shift, t_track, track_lat),
                                  np.interp(t_points - shift, t_track, track_lon), lat, lon)

    inside = (t_points - range_s >= t_track[0]) & (t_points + range_s <= t_track[-1])
    if not inside.any():
        return np.nan, np.full(len(t_points), np.nan)
    shifts = _offset_scan(range_s)
    offset = shifts[int(np.argmin([np.median(distances(shift)[inside]) for shift in shifts]))]
    return float(offset), distances(offset)


def _altitude_errors(track_df, target_df, source_column, target_column, max_gap_s):
    """
    Reference minus reported value per report (altitude or ground speed), the
    reference interpolated at the report time and not across gaps wider than
    max_gap_s. All NaN when a column is missing.
    """
    import pandas as pd

    errors = pd.Series(np.nan, index=target_df.index)
    if source_column not in track_df.columns or target_column not in target_df.columns:
        return errors
    reference = track_df[source_column].dropna()
    at_reports = interp_with_gaps(_index_seconds(reference.index), reference.to_numpy(dtype=float),
                                  _index_seconds(target_df.index), max_gap_s)
    return pd.Series(at_reports - target_df[target_column].to_numpy(dtype=float), index=target_df.index)


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Returns distance in meters between two lat/lon points.
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return 2 * _EARTH_RADIUS * np.arcsin(np.sqrt(a))


def _latlon_to_local_xy(lat, lon, lat0):
    """
    Projects lat/lon (degrees) onto a local equirectangular plane in meters,
    referenced to the latitude `lat0` (radians), so Euclidean distances keep
    their real-world scale. Returns an (N, 2) array.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    x = _EARTH_RADIUS * np.radians(lon) * np.cos(lat0)
    y = _EARTH_RADIUS * np.radians(lat)
    return np.column_stack([x, y])


def _normalized_mad(errors):
    """
    Normalized Median Absolute Deviation (NMAD) of an error array:
        NMAD = 1.4826 * median(|e_i - median(e)|)
    """
    errors = _finite(errors)
    if not errors.size:
        return np.nan
    return float(_NMAD_SCALE_FACTOR * np.median(np.abs(errors - np.median(errors))))
