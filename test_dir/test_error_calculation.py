from types import SimpleNamespace

import pandas as pd
import pytest

from qualifly_lib.error_calculation import calculate_errors, haversine_distance


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


def _config():
    fields = SimpleNamespace(LAT="lat", LON="lon", GEOALT="alt")
    return {"source": fields, "target": fields}


def _trajectory(lat, lon, alt):
    return pd.DataFrame({"lat": lat, "lon": lon, "alt": alt})


def test_calculate_errors_scores_an_identical_trajectory_as_zero():
    source_df = _trajectory([50.0, 50.001, 50.002], [10.0, 10.001, 10.002], [1000.0, 1010.0, 1020.0])
    target_df = source_df.copy()

    scores, position_errors, best_variant = calculate_errors(source_df, {"original": target_df}, _config())

    assert best_variant == "original"
    assert scores["original"]["pos_dfd"] == pytest.approx(0.0, abs=1e-6)
    assert scores["original"]["pos_dtw"] == pytest.approx(0.0, abs=1e-6)
    assert scores["original"]["alt_rmse"] == pytest.approx(0.0, abs=1e-6)
    assert scores["original"]["alt_nmad"] == pytest.approx(0.0, abs=1e-6)
    assert position_errors["original"].tolist() == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)


def test_calculate_errors_picks_the_lower_scoring_variant():
    source_df = _trajectory([50.0, 50.001, 50.002], [10.0, 10.001, 10.002], [1000.0, 1010.0, 1020.0])
    close_df = source_df.copy()
    close_df["alt"] = close_df["alt"] + 1.0
    far_df = source_df.copy()
    far_df["alt"] = far_df["alt"] + 500.0

    scores, _, best_variant = calculate_errors(
        source_df, {"close": close_df, "far": far_df}, _config(), metric_key="alt_rmse"
    )

    assert best_variant == "close"
    assert scores["close"]["alt_rmse"] < scores["far"]["alt_rmse"]


def test_calculate_errors_handles_an_empty_dataframe():
    source_df = _trajectory([], [], [])
    target_df = _trajectory([1.0], [1.0], [1.0])

    scores, _, best_variant = calculate_errors(source_df, {"original": target_df}, _config())

    assert scores["original"] == {"pos_dfd": None, "pos_dtw": None, "alt_rmse": None, "alt_nmad": None}
    assert best_variant is None
