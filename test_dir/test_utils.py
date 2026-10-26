# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

import pandas as pd
import pytest

from qualifly_lib import utils
from qualifly_lib.utils import (
    add_sum_columns,
    log_failed_ids,
    map_to_config_columns,
    prefer_position_time,
    prepare_baro_altitude,
)


def test_log_failed_ids_creates_file_with_new_id(tmp_path):
    filepath = tmp_path / "failed.txt"

    log_failed_ids(str(filepath), "abc123")

    assert filepath.read_text().splitlines() == ["abc123"]


def test_log_failed_ids_does_not_duplicate_an_existing_id(tmp_path):
    filepath = tmp_path / "failed.txt"
    filepath.write_text("abc123\n")

    log_failed_ids(str(filepath), "abc123")

    assert filepath.read_text().splitlines() == ["abc123"]


def test_log_failed_ids_appends_a_new_id_to_existing_file(tmp_path):
    filepath = tmp_path / "failed.txt"
    filepath.write_text("abc123\n")

    log_failed_ids(str(filepath), "def456")

    assert filepath.read_text().splitlines() == ["abc123", "def456"]


def test_prepare_baro_altitude_keeps_prepared_target():
    df = pd.DataFrame({"baro_alt": [9.0], "Raw_Baro_1": [3280.84]})

    result = prepare_baro_altitude(df, baro_params=["Raw_Baro_1"])

    assert result["baro_alt"].tolist() == [9.0]
    assert result["Raw_Baro_1"].tolist() == [3280.84]


def test_add_sum_columns_sums_the_fine_part_and_holds_the_slow_word():
    df = pd.DataFrame({
        "lat": [51.0, None, None],    # ~1 Hz word, held across the fine samples
        "LatFine": [0.0, 0.1, 0.2],   # ~20 Hz fine part
    })

    result = add_sum_columns(df, "lat", ["LatFine"])

    assert result["lat"].tolist() == pytest.approx([51.0, 51.1, 51.2])
    # the summed sources are dropped, so a second pass cannot add them twice
    assert "LatFine" not in result.columns
    # the input frame is untouched
    assert "LatFine" in df.columns
    assert df["lat"].iloc[0] == 51.0 and df["lat"].iloc[1:].isna().all()


def test_add_sum_columns_tolerates_unknown_targets_and_sources():
    df = pd.DataFrame({"lat": [51.0, 51.1], "LatFine": [0.0, 0.1]})

    assert add_sum_columns(df, "lon", ["LatFine"]).equals(df)
    assert add_sum_columns(df, "lat", ["missing"]).equals(df)
    # a single source is accepted as a plain name instead of a list
    assert add_sum_columns(df, "lat", "LatFine")["lat"].tolist() == pytest.approx(
        [51.0, 51.2]
    )


def test_add_sum_columns_falls_back_to_the_next_alternative_name():
    df_both = pd.DataFrame({
        "lat": [51.0, None],
        "LatFineOld": [0.0, 0.1],   # the first name present is used
        "LatFine": [0.0, 0.2],
    })
    df_fallback = pd.DataFrame({"lat": [51.0, None], "LatFine": [0.0, 0.2]})

    result = add_sum_columns(df_both, "lat", ["LatFineOld", "LatFine"])
    assert result["lat"].tolist() == pytest.approx([51.0, 51.1])
    # every alternative is dropped, so a second pass cannot sum one again
    assert "LatFineOld" not in result.columns and "LatFine" not in result.columns

    # without the first name the fallback is summed
    result = add_sum_columns(df_fallback, "lat", ["LatFineOld", "LatFine"])
    assert result["lat"].tolist() == pytest.approx([51.0, 51.2])
    assert "LatFine" not in result.columns


def test_add_sum_columns_adds_one_parameter_per_name_group():
    df = pd.DataFrame({
        "lat": [51.0, None],
        "LatFineA": [0.1, 0.2],
        "LatFineB": [0.01, 0.01],
    })

    # one name group per added parameter: ["LatFineOld", "LatFineA"] and ["LatFineB"]
    result = add_sum_columns(df, "lat", [["LatFineOld", "LatFineA"], "LatFineB"])

    assert result["lat"].tolist() == pytest.approx([51.11, 51.21])


def test_map_to_config_columns_sums_the_first_existing_configured_names():
    from qualifly_lib.data_fetching.local_file_in import Config

    df = pd.DataFrame({
        "Time": [0.0, 1.0],
        "my_lat": [51.0, None],
        "my_latFine": [0.0, 0.2],
    })
    options = {
        "columns": {"time": "Time", "lat": ["absent_lat", "my_lat"]},
        "sum_columns": {"lat": ["absent_fine", "my_latFine"]},
    }

    result = map_to_config_columns(df, Config, options)

    assert result["lat"].tolist() == pytest.approx([51.0, 51.2])
    assert "my_latFine" not in result.columns


def test_map_to_config_columns_adds_the_configured_sum_columns():
    from qualifly_lib.data_fetching.local_file_in import Config

    df = pd.DataFrame({
        "Time": [0.0, 0.5, 1.0],
        "my_lat": [51.0, None, None],
        "my_latFine": [0.0, 0.1, 0.2],
        "my_lon": [6.0, None, None],
        "my_lonFine": [0.0, 0.01, 0.02],
    })
    options = {
        "columns": {"time": "Time", "lat": "my_lat", "lon": "my_lon"},
        "sum_columns": {"lat": ["my_latFine"], "lon": ["my_lonFine"]},
    }

    result = map_to_config_columns(df, Config, options)

    assert result["lat"].tolist() == pytest.approx([51.0, 51.1, 51.2])
    assert result["lon"].tolist() == pytest.approx([6.0, 6.01, 6.02])
    assert result[str(Config.TIME)].tolist() == [0.0, 0.5, 1.0]

    # a second pass over the result (as read back from a cached CSV) is a no-op
    again = map_to_config_columns(result, Config, options)
    assert again.columns.tolist() == result.columns.tolist()
    assert again["lat"].tolist() == pytest.approx(result["lat"].tolist())


def test_map_to_config_columns_converts_an_ft_geo_alt_once():
    from qualifly_lib.data_fetching.local_file_in import Config

    df = pd.DataFrame({
        "Time": [0.0, 1.0],
        "my_alt_ft": [3280.84, 3379.264],  # 1000 m and 1030 m in ft
    })
    options = {
        "columns": {"time": "Time", "geo_alt": "my_alt_ft"},
        "geo_alt_is_ft": True,
    }

    result = map_to_config_columns(df, Config, options)

    assert result[str(Config.GEOALT)].tolist() == pytest.approx([1000.0, 1030.0], rel=1e-4)

    # a second pass over the converted result (as read back from a cached CSV)
    # finds no ft source column anymore and never converts it again
    again = map_to_config_columns(result, Config, options)
    assert again[str(Config.GEOALT)].tolist() == pytest.approx(
        result[str(Config.GEOALT)].tolist()
    )

    # without the flag the feet are taken as meters
    assert map_to_config_columns(df, Config, {
        "columns": {"time": "Time", "geo_alt": "my_alt_ft"}
    })[str(Config.GEOALT)].tolist() == pytest.approx([3280.84, 3379.264])


def test_prefer_position_time_falls_back_to_snapshot_before_first_fix():
    df = pd.DataFrame({
        "time": pd.to_datetime([10, 11, 12], unit="s", utc=True),
        "lastposupdate": [None, 10.5, 11.5],
    })

    result = prefer_position_time(df)

    assert result["time"].tolist() == [10.0, 10.5, 11.5]
    assert prefer_position_time(result.copy())["time"].tolist() == [10.0, 10.5, 11.5]


def test_prefer_position_time_reads_pyarrow_columns_as_pyopensky_returns_them():
    df = pd.DataFrame({
        "time": pd.to_datetime([10, 11, 12], unit="s", utc=True),
        "lastposupdate": [None, 10.5, 11.5],
    }).convert_dtypes(dtype_backend="pyarrow")

    assert prefer_position_time(df)["time"].tolist() == [10.0, 10.5, 11.5]


def test_opensky_baroaltitude_is_the_barometric_altitude():
    from qualifly_lib.data_fetching.opensky import Config

    df = pd.DataFrame({
        "time": [10.0, 11.0],
        "lat": [51.0, 51.1],
        "lon": [6.0, 6.1],
        "geoaltitude": [1000.0, 1010.0],
        "baroaltitude": [1020.0, 1030.0],
    })

    assert map_to_config_columns(df, Config)[Config.BAROALT].tolist() == [1020.0, 1030.0]


def test_msl_to_ellipsoid_height_adds_the_geoid_undulation(monkeypatch):
    monkeypatch.setattr(utils, "geoid_undulation", lambda lat, lon: [45.0] * len(lat))
    df = pd.DataFrame({
        "lat": [52.0, None, 52.1], "lon": [10.0, None, 10.1], "geo_alt": [100.0, 110.0, 120.0],
    })

    out = utils.msl_to_ellipsoid_height(df, "lat", "lon", "geo_alt")

    # the altitude sample without a position uses the neighbouring position
    assert out["geo_alt"].tolist() == [145.0, 155.0, 165.0]
    assert df["geo_alt"].tolist() == [100.0, 110.0, 120.0]


def test_map_to_config_columns_converts_a_kt_ground_speed_once():
    from qualifly_lib.data_fetching.local_file_in import Config

    df = pd.DataFrame({
        "Time": [0.0, 1.0],
        "my_gs_kt": [388.76, 388.76],  # 200 m/s in kt
    })
    options = {
        "columns": {"time": "Time", "ground_speed": "my_gs_kt"},
        "ground_speed_is_kt": True,
    }

    result = map_to_config_columns(df, Config, options)

    assert result[str(Config.GROUNDSPEED)].tolist() == pytest.approx([200.0, 200.0], rel=1e-4)

    # a second pass over the converted result (as read back from a cached CSV)
    # finds no kt source column anymore and never converts it again
    again = map_to_config_columns(result, Config, options)
    assert again[str(Config.GROUNDSPEED)].tolist() == pytest.approx(
        result[str(Config.GROUNDSPEED)].tolist()
    )


def test_map_to_config_columns_takes_an_opensky_velocity_as_the_ground_speed():
    from qualifly_lib.data_fetching.opensky import Config

    df = pd.DataFrame({
        "time": [10.0, 11.0],
        "lat": [51.0, 51.1],
        "lon": [6.0, 6.1],
        "velocity": [200.0, 210.0],
    })

    assert map_to_config_columns(df, Config)[Config.GROUNDSPEED].tolist() == [200.0, 210.0]
