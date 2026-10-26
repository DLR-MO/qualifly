# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from qualifly_lib.filtering import median_filter, use_filtering

TARGET = SimpleNamespace(LAT="lat", LON="lon", GEOALT="geo_alt", BAROALT="baro_alt")
_DEG_PER_M = 180.0 / np.pi / 6_371_000.0
_SPEED = 150.0       # m/s, heading east
_CLIMB = 10.0        # m/s
_LAT = 50.0


def _grid(noise_m=0.0, periods=300, seed=0, velocity_noise=0.0):
    """A 1 s grid of reports flying east and climbing, with position and velocity noise."""
    seconds = np.arange(periods, dtype=float)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_m, (periods, 3)) if noise_m else np.zeros((periods, 3))
    jitter = rng.normal(0.0, velocity_noise, (periods, 3)) if velocity_noise else np.zeros((periods, 3))
    return pd.DataFrame({
        "lat": _LAT + noise[:, 0] * _DEG_PER_M,
        "lon": 10.0 + (_SPEED * seconds + noise[:, 1]) * _DEG_PER_M / np.cos(np.radians(_LAT)),
        "geo_alt": 1000.0 + _CLIMB * seconds + noise[:, 2],
        "baro_alt": 900.0 + _CLIMB * seconds + noise[:, 2],
        "velocity": _SPEED + jitter[:, 0], "heading": 90.0 + jitter[:, 1], "vertrate": _CLIMB + jitter[:, 2],
    }, index=pd.date_range("2024-01-01", periods=periods, freq="1s", tz="UTC", name="timestamp"))


def _east_error(filtered, truth):
    """Along-track (east) error of each row in meters."""
    return ((filtered["lon"] - truth["lon"]) * np.cos(np.radians(_LAT)) / _DEG_PER_M).to_numpy()


def test_mean_keeps_a_straight_track_and_its_edges():
    grid = _grid()

    mean = use_filtering(grid, TARGET)["mean"]

    for column in ("lat", "lon", "geo_alt"):
        assert mean[column].to_numpy() == pytest.approx(grid[column].to_numpy())


def test_median_removes_a_spike_and_keeps_a_straight_track():
    grid = _grid()
    spiked = grid.copy()
    spiked.iloc[100, spiked.columns.get_loc("lat")] += 500.0 * _DEG_PER_M

    median = median_filter(spiked, TARGET)

    for column in ("lat", "lon", "geo_alt"):
        assert median[column].to_numpy() == pytest.approx(grid[column].to_numpy())


def test_median_removes_two_spiked_reports_between_gaps():
    # a window of reports, not of seconds: the gaps around the spike do not matter
    reports = _grid().iloc[[*range(90), 96, 97, *range(110, 200)]]
    spiked = reports.copy()
    spiked.iloc[[90, 91], spiked.columns.get_loc("lat")] += 6.0

    median = median_filter(spiked, TARGET)

    assert median["lat"].to_numpy() == pytest.approx(reports["lat"].to_numpy())


def test_median_skips_a_missing_altitude():
    reports = _grid()
    reports.iloc[50, reports.columns.get_loc("geo_alt")] = np.nan
    reports.iloc[51, reports.columns.get_loc("geo_alt")] += 500.0

    median = median_filter(reports, TARGET)

    assert np.isnan(median["geo_alt"].iloc[50])
    assert median["geo_alt"].iloc[51] == pytest.approx(_grid()["geo_alt"].iloc[51], abs=10.0)


def test_kalman_smoother_runs_half_a_second_ahead():
    # traffic's KalmanSmoother6D: 75 m ahead along the track at 150 m/s, but
    # in the right place across it and in altitude (i.e. in the right units)
    truth = _grid()

    smoothed = use_filtering(_grid(noise_m=10.0, velocity_noise=0.5), TARGET)["kalman_smoother"]

    along = _east_error(smoothed, truth)[20:-20]
    assert along.mean() == pytest.approx(0.5 * _SPEED, abs=10.0)
    across = ((smoothed["lat"] - truth["lat"]) / _DEG_PER_M).iloc[20:-20]
    assert across.abs().mean() < 5.0
    assert (smoothed["geo_alt"] - truth["geo_alt"]).abs().iloc[20:-20].mean() < 5.0


def test_kalman_smoother_needs_the_velocities():
    grid = _grid().drop(columns="vertrate")

    assert set(use_filtering(grid, TARGET)) == {"mean"}


def test_kalman_smoother_is_skipped_on_constant_velocities():
    # traffic derives its noise from the data: constant columns leave it singular
    assert set(use_filtering(_grid(noise_m=10.0), TARGET)) == {"mean"}


def test_mean_and_median_smooth_the_ground_speed():
    grid = _grid()
    spiked = grid.copy()
    spiked.iloc[150, spiked.columns.get_loc("velocity")] = _SPEED + 50.0

    mean = use_filtering(spiked, TARGET)["mean"]["velocity"]
    median = median_filter(spiked, TARGET)["velocity"]

    assert mean.iloc[148:153].tolist() == pytest.approx([_SPEED + 10.0] * 5)
    assert median.tolist() == pytest.approx(grid["velocity"].tolist())


def test_kalman_smoother_returns_the_ground_speed_in_m_per_s():
    smoothed = use_filtering(_grid(noise_m=10.0, velocity_noise=0.5), TARGET)["kalman_smoother"]

    assert smoothed["velocity"].iloc[20:-20].mean() == pytest.approx(_SPEED, abs=1.0)


def test_every_filter_smooths_the_barometric_altitude():
    truth = _grid()
    spiked = _grid(noise_m=10.0, velocity_noise=0.5)
    spiked.iloc[150, spiked.columns.get_loc("baro_alt")] += 500.0

    variants = {**use_filtering(spiked, TARGET), "median": median_filter(spiked, TARGET)}

    assert set(variants) == {"mean", "median", "kalman_smoother"}
    for name, variant in variants.items():
        error = (variant["baro_alt"] - truth["baro_alt"]).abs().iloc[145:156]
        assert error.max() < (150.0 if name == "mean" else 30.0), name


def test_kalman_smoother_runs_in_parallel_threads():
    # traffic's filters share their state between instances; library use from threads must not mix them
    from concurrent.futures import ThreadPoolExecutor

    grids = [_grid(noise_m=10.0, periods=200 + 50 * k, seed=k, velocity_noise=0.5) for k in range(4)]

    with ThreadPoolExecutor(4) as pool:
        smoothed = list(pool.map(lambda grid: use_filtering(grid, TARGET)["kalman_smoother"], grids))

    assert [len(s) for s in smoothed] == [len(g) for g in grids]
