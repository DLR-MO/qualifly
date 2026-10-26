# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from qualifly_lib.error_calculation import calculate_errors, haversine_distance
from qualifly_lib.utils import SCORE_NAMES


def test_haversine_distance_is_zero_for_identical_points():
    assert haversine_distance(52.0, 13.0, 52.0, 13.0) == pytest.approx(0.0, abs=1e-6)


def test_haversine_distance_matches_a_known_city_distance():
    # Berlin (52.5200N, 13.4050E) to Munich (48.1351N, 11.5820E) is roughly 500 km
    distance_m = haversine_distance(52.5200, 13.4050, 48.1351, 11.5820)
    assert distance_m == pytest.approx(504_000, rel=0.02)


def test_haversine_distance_is_symmetric():
    d1 = haversine_distance(52.0, 13.0, 48.0, 11.0)
    d2 = haversine_distance(48.0, 11.0, 52.0, 13.0)
    assert d1 == pytest.approx(d2)


def _config(**settings):
    fields = SimpleNamespace(LAT="lat", LON="lon", GEOALT="alt")
    return {"source": fields, "target": fields, **settings}


def _trajectory(lat, lon, alt, index=None):
    return pd.DataFrame({"lat": lat, "lon": lon, "alt": alt}, index=index)


def _east_line(north_m=0.0, east_m=0.0, periods=40, step_s=1.0, start_s=0.0, speed=10.0, lat=50.0, lon0=10.0):
    """Eastward track at `speed` m/s, shifted `north_m` across and `east_m` along the track."""
    seconds = start_s + np.arange(periods) * step_s
    index = pd.Timestamp("2022-01-01", tz="UTC") + pd.to_timedelta(seconds, unit="s")
    deg_per_m = 180.0 / np.pi / 6_371_000.0
    lats = lat + np.broadcast_to(north_m, periods) * deg_per_m
    lons = lon0 + (seconds * speed + east_m) * deg_per_m / np.cos(np.radians(lat))
    return _trajectory(lats, lons, [1000.0] * periods, index=index)


POSITION_METRICS = ("pos_rmse", "pos_p95", "pos_track_mean", "pos_track_p95", "pos_track_max")


def test_calculate_errors_scores_an_identical_trajectory_as_zero():
    source_df = _east_line()
    target_df = source_df.copy()

    scores, errors = calculate_errors(source_df, {"original": target_df}, _config())

    for metric in (*POSITION_METRICS, "pos_p95_synced", "alt_rmse", "alt_nmad"):
        assert scores["original"][metric] == pytest.approx(0.0, abs=1e-6)
    assert errors["original"]["position_error"].tolist() == pytest.approx([0.0] * 40, abs=1e-6)


def test_position_metrics_report_a_cross_track_offset_in_meters():
    # at 50 deg latitude, per-curve reference latitudes would add a spurious east-west offset
    source_df = _east_line()
    target_df = _east_line(north_m=50.0)

    scores, _ = calculate_errors(source_df, {"original": target_df}, _config())

    for metric in (*POSITION_METRICS, "pos_p95_synced"):
        assert scores["original"][metric] == pytest.approx(50.0, abs=0.05)
    # no time shift brings a track beside the source closer
    assert scores["original"]["time_offset_s"] == pytest.approx(0.0)


def test_track_distance_ignores_a_time_lag_within_the_window_only():
    # the target reports where the aircraft was 3 s ago: on the track, 30 m off at every instant
    track_df = _east_line(periods=60, start_s=-10.0)
    source_df = _east_line()
    target_df = _east_line(east_m=-30.0)

    wide, _ = calculate_errors(
        source_df, {"original": target_df}, _config(track_window_s=5.0), source_track=track_df)
    tight, _ = calculate_errors(
        source_df, {"original": target_df}, _config(track_window_s=0.2), source_track=track_df)

    for scores in (wide, tight):
        assert scores["original"]["pos_rmse"] == pytest.approx(30.0, abs=0.05)
        assert scores["original"]["pos_p95"] == pytest.approx(30.0, abs=0.05)
    for metric in ("pos_track_mean", "pos_track_p95", "pos_track_max"):
        assert wide["original"][metric] == pytest.approx(0.0, abs=0.05)
        # only the track within t +- 0.2 s counts: 2 m of the 30 m lag are forgiven
        assert tight["original"][metric] == pytest.approx(28.0, abs=0.05)


@pytest.mark.parametrize("track_step_s", [0.1, 5.0])
def test_track_distance_does_not_depend_on_the_sample_rate_of_the_track(track_step_s):
    # reports on the flown line, between the samples of the track
    track_df = _east_line(periods=int(60 / track_step_s), step_s=track_step_s, start_s=-10.0)
    reports_df = _east_line(start_s=0.05)

    scores, _ = calculate_errors(reports_df, {"original": reports_df}, _config(), source_track=track_df)

    assert scores["original"]["pos_track_max"] == pytest.approx(0.0, abs=0.01)


def test_pos_p95_synced_forgives_a_constant_lag_but_not_a_distance_beside_the_track():
    source_df = _east_line()
    lagging_df = _east_line(east_m=-8.0)      # on the track, 0.8 s late: 8 m off at every instant
    beside_df = _east_line(north_m=5.0)       # 5 m beside the track at every instant

    scores, _ = calculate_errors(
        source_df, {"lagging": lagging_df, "beside": beside_df}, _config(),
        source_track=_east_line(periods=60, start_s=-10.0),
    )

    # the lag shows in time_offset_s and pos_rmse, but not in the synced scores
    assert scores["lagging"]["pos_rmse"] > scores["beside"]["pos_rmse"]
    assert scores["lagging"]["time_offset_s"] == pytest.approx(0.8)
    assert scores["lagging"]["pos_rmse_synced"] < scores["beside"]["pos_rmse_synced"]
    assert scores["lagging"]["pos_p95_synced"] < scores["beside"]["pos_p95_synced"]


def test_pos_rmse_counts_a_few_bad_seconds_that_pos_p95_ignores():
    # 2 of 40 seconds are 400 m off: fewer than the 5 % the percentile sees
    source_df = _east_line()
    offsets = np.zeros(40)
    offsets[[10, 11]] = 400.0
    target_df = _east_line(north_m=offsets)

    scores, _ = calculate_errors(source_df, {"original": target_df}, _config())

    assert scores["original"]["pos_p95"] == pytest.approx(0.0, abs=25.0)
    assert scores["original"]["pos_rmse"] == pytest.approx(400.0 * np.sqrt(2 / 40), abs=0.5)


def test_calculate_errors_scores_an_all_nan_variant_without_crashing():
    # e.g. the Kalman smoother after a divergent pass
    index = pd.date_range("2022-01-01", periods=5, freq="1s", tz="UTC")
    source_df = _trajectory([50.0] * 5, [10.0] * 5, [1000.0] * 5, index=index)
    target_df = _trajectory([np.nan] * 5, [np.nan] * 5, [np.nan] * 5, index=index)

    scores, _ = calculate_errors(source_df, {"original": target_df}, _config())

    for metric in SCORE_NAMES:
        assert np.isnan(scores["original"][metric])


def test_calculate_errors_handles_an_empty_dataframe():
    source_df = _trajectory([], [], [])
    target_df = _trajectory([1.0], [1.0], [1.0])

    scores, errors = calculate_errors(source_df, {"original": target_df}, _config())

    assert scores["original"] == dict.fromkeys(SCORE_NAMES)
    assert errors == {}


def test_time_offset_removes_reports_stamped_late():
    # every report is stamped 0.2 s late: 40 m off at 200 m/s, but on the track
    track_df = _east_line(periods=80, start_s=-20.0, speed=200.0)
    source_df = _east_line(speed=200.0)
    reports_df = _east_line(east_m=-40.0, speed=200.0)

    scores, errors = calculate_errors(source_df, {"original": reports_df}, _config(), source_track=track_df)

    assert scores["original"]["pos_p95"] == pytest.approx(40.0, abs=0.05)
    assert scores["original"]["time_offset_s"] == pytest.approx(0.2)
    assert scores["original"]["pos_rmse_synced"] == pytest.approx(0.0, abs=0.05)
    assert scores["original"]["pos_p95_synced"] == pytest.approx(0.0, abs=0.05)
    assert scores["original"]["pos_track_max"] == pytest.approx(0.0, abs=0.05)
    assert errors["original"]["position_error_synced"].max() == pytest.approx(0.0, abs=0.05)
    # the tolerances count from the report time, not from the offset: 0.1 s forgive 20 m of the 40 m
    assert scores["original"]["pos_p95_tol100ms"] == pytest.approx(20.0, abs=0.05)
    assert scores["original"]["pos_p95_tol300ms"] == pytest.approx(0.0, abs=0.05)


def _baro_config():
    fields = SimpleNamespace(LAT="lat", LON="lon", GEOALT="alt", BAROALT="baro")
    return {"source": fields, "target": fields}


def _climb(seconds):
    """An altitude that climbs at a changing rate (a constant rate cannot tell a time offset from a bias)."""
    return 1000.0 + 300.0 * np.sin(np.asarray(seconds, dtype=float) / 10.0)


def test_altitude_errors_are_taken_at_the_report_time():
    # reports stamped 0.5 s late, the barometric altitude 30 m too high
    track_df = _east_line(periods=800, step_s=0.1, start_s=-10.0, speed=100.0)
    track_df["alt"] = track_df["baro"] = _climb(np.arange(800) * 0.1 - 10.0)
    source_df = _east_line(periods=60, speed=100.0)
    reports_df = _east_line(periods=60, east_m=-50.0, speed=100.0)
    reports_df["alt"] = _climb(np.arange(60) - 0.5)
    reports_df["baro"] = reports_df["alt"] + 30.0

    scores, errors = calculate_errors(source_df, {"original": reports_df}, _baro_config(), source_track=track_df)

    # the offset of the position is not removed: the 0.5 s show in the climb, up to 15 m at 30 m/s
    assert scores["original"]["time_offset_s"] == pytest.approx(0.5)
    assert scores["original"]["alt_p95"] == pytest.approx(15.0, abs=1.5)
    assert errors["original"]["altitude_error"].abs().max() == pytest.approx(15.0, abs=1.5)
    # the 30 m bias between the pressure references does not spread
    assert scores["original"]["baro_nmad"] == pytest.approx(scores["original"]["alt_nmad"], abs=0.01)


def test_baro_nmad_does_not_follow_the_time_offset_of_a_filter():
    # a filter leading 0.5 s shifts the positions but not the barometric altitude
    track_df = _east_line(periods=800, step_s=0.1, start_s=-10.0, speed=100.0)
    track_df["alt"] = track_df["baro"] = _climb(np.arange(800) * 0.1 - 10.0)
    source_df = _east_line(periods=60, speed=100.0)
    reports_df = _east_line(periods=60, speed=100.0).assign(alt=_climb(np.arange(60)))
    reports_df["baro"] = reports_df["alt"]
    leading_df = _east_line(periods=60, east_m=50.0, speed=100.0).assign(alt=reports_df["alt"], baro=reports_df["baro"])

    scores, _ = calculate_errors(source_df, {"original": reports_df, "leading": leading_df}, _baro_config(),
                                    source_track=track_df)

    assert scores["leading"]["time_offset_s"] == pytest.approx(-0.5)
    assert scores["leading"]["baro_nmad"] == pytest.approx(0.0, abs=0.01)


def test_baro_nmad_is_scored_for_an_opensky_target():
    # an opensky target carries its barometric altitude as `baroaltitude`
    from qualifly_lib.data_fetching.opensky import Config as opensky_target

    source_fields = SimpleNamespace(LAT="lat", LON="lon", GEOALT="alt", BAROALT="baro_alt")
    track_df = _east_line(periods=800, step_s=0.1, start_s=-10.0, speed=100.0)
    track_df["alt"] = track_df["baro_alt"] = _climb(np.arange(800) * 0.1 - 10.0)
    source_df = _east_line(speed=100.0)
    reports_df = _east_line(periods=60, speed=100.0).rename(columns={"alt": "geoaltitude"})
    reports_df["geoaltitude"] = _climb(np.arange(60))
    reports_df["baroaltitude"] = reports_df["geoaltitude"] + 30.0

    config = {"source": source_fields, "target": opensky_target}
    scores, _ = calculate_errors(source_df, {"original": reports_df}, config, source_track=track_df)

    assert scores["original"]["alt_nmad"] == pytest.approx(0.0, abs=0.01)
    # the 30 m bias between the pressure references does not spread, but shows in baro_bias
    assert scores["original"]["baro_nmad"] == pytest.approx(0.0, abs=0.01)
    assert scores["original"]["baro_bias"] == pytest.approx(-30.0, abs=0.01)


def test_alt_p95_counts_the_offset_and_the_bad_reports_that_alt_nmad_ignores():
    # altitude 20 m too high, every 10th report 500 m off
    track_df = _east_line(periods=800, step_s=0.1, start_s=-10.0, speed=100.0)
    track_df["alt"] = _climb(np.arange(800) * 0.1 - 10.0)
    source_df = _east_line(periods=60, speed=100.0)
    reports_df = _east_line(periods=60, speed=100.0).assign(alt=_climb(np.arange(60)) + 20.0)
    reports_df.iloc[::10, reports_df.columns.get_loc("alt")] += 500.0

    scores, _ = calculate_errors(source_df, {"original": reports_df}, _config(), source_track=track_df)

    assert scores["original"]["alt_nmad"] == pytest.approx(0.0, abs=0.01)
    assert scores["original"]["alt_p95"] == pytest.approx(520.0, abs=0.01)
    # reference minus report: the reports lie 20 m too high
    assert scores["original"]["alt_bias"] == pytest.approx(-20.0, abs=0.01)


def test_altitude_errors_need_no_time_offset():
    # no report's +-1 s offset range lies inside the source
    track_df = _east_line(periods=2)
    scores, errors = calculate_errors(track_df, {"original": track_df.copy()}, _baro_config(), source_track=track_df)

    assert np.isnan(scores["original"]["time_offset_s"])
    assert scores["original"]["alt_rmse"] == pytest.approx(0.0)
    assert errors["original"]["altitude_error"].notna().all()


def test_altitude_errors_do_not_interpolate_across_a_source_gap():
    # the geometric altitude of the source is missing from 10 s to 14 s (> max_gap_s)
    track_df = _east_line(periods=80, start_s=-20.0)
    track_df.loc[track_df.index[30:35], "alt"] = np.nan
    source_df = _east_line()

    _, errors = calculate_errors(source_df, {"original": source_df.copy()}, _config(), source_track=track_df)

    assert errors["original"]["altitude_error"].isna().sum() == 5
    assert errors["original"]["position_error"].notna().all()


def _gs_config():
    fields = SimpleNamespace(LAT="lat", LON="lon", GEOALT="alt", GROUNDSPEED="gs")
    return {"source": fields, "target": fields}


def test_ground_speed_errors_are_taken_at_the_report_time():
    # the reference flies 200 m/s, the reports broadcast 190 m/s: a constant 10 m/s offset
    track_df = _east_line(periods=80, start_s=-20.0)
    track_df["gs"] = 200.0
    source_df = _east_line()
    reports_df = _east_line().assign(gs=190.0)

    scores, errors = calculate_errors(source_df, {"original": reports_df}, _gs_config(), source_track=track_df)

    assert scores["original"]["gs_bias"] == pytest.approx(10.0, abs=0.01)
    assert scores["original"]["gs_rmse"] == pytest.approx(10.0, abs=0.01)
    assert scores["original"]["gs_nmad"] == pytest.approx(0.0, abs=0.01)
    assert scores["original"]["gs_p95"] == pytest.approx(10.0, abs=0.01)
    assert errors["original"]["groundspeed_error"].notna().all()


def test_ground_speed_scores_are_nan_without_the_columns():
    source_df = _east_line()

    scores, _ = calculate_errors(source_df, {"original": source_df.copy()}, _gs_config())

    for metric in ("gs_bias", "gs_rmse", "gs_nmad", "gs_p95"):
        assert np.isnan(scores["original"][metric])
