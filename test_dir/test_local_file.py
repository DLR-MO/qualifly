# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

import pandas as pd
import pytest

import qualifly_lib.data_fetching.local_file_in as local_file_in
import qualifly_lib.data_fetching.local_public_ref as local_public_ref
from qualifly_lib.data_fetching._local_file import fetch_local_file, resolve_flight_file
from qualifly_lib.utils import to_epoch_seconds


def _write_canonical_csv(path, collection_id, **columns):
    data = {
        "time": [1696153787.5, 1696153788.5, 1696153789.5],
        "lat": [51.40123, 51.40141, 51.40158],
        "lon": [6.76543, 6.76598, 6.76652],
        "geo_alt": [10972.8, 10975.4, 10978.0],
    }
    data.update(columns)
    df = pd.DataFrame(data)
    file = path / f"{collection_id}.csv"
    df.to_csv(file, index=False)
    return df


def test_fetch_local_file_in_maps_canonical_columns(tmp_path):
    _write_canonical_csv(tmp_path, "f1")

    df = local_file_in.fetch_data("f1", "unused", {"path": str(tmp_path)})

    assert df is not None
    for column in ("time", "lat", "lon", "geo_alt"):
        assert column in df.columns
    assert df["time"].tolist() == [1696153787.5, 1696153788.5, 1696153789.5]


def test_fetch_local_public_ref_parquet_generic_columns(tmp_path):
    df = pd.DataFrame({
        "time": pd.to_datetime([1696153787, 1696153788], unit="s", utc=True),
        "lat": [51.4, 51.5],
        "lon": [6.7, 6.8],
        "geoaltitude": [1000.0, 1100.0],  # generic spelling -> geo_alt
        "positionnac": [3, 2],
    })
    df.to_parquet(tmp_path / "f2.parquet", index=False)

    result = local_public_ref.fetch_data("f2", "unused", {"path": str(tmp_path)})

    assert result is not None
    assert {"time", "lat", "lon", "geo_alt", "position_nac"} <= set(result.columns)
    # datetime input is normalized to epoch seconds
    assert result["time"].tolist() == [1696153787.0, 1696153788.0]
    assert result["position_nac"].tolist() == [3, 2]


def test_raw_baro_params_are_merged_metric(tmp_path):
    _write_canonical_csv(
        tmp_path, "f3",
        **{
            "Raw_Baro_1": [3280.84, 3380.84, 3480.84],  # ft
            "Raw_Baro_2": [3280.84, 3380.84, 3480.84],  # ft
        },
    )

    df = local_file_in.fetch_data(
        "f3", "unused", {"path": str(tmp_path), "baro_params": ["Raw_Baro_1", "Raw_Baro_2"]})

    assert df is not None
    # mean of the raw barometric altitudes, converted ft -> m
    assert df["baro_alt"].tolist() == pytest.approx([1000.0, 1030.48, 1060.96])
    # raw columns are converted to meters as well
    assert df["Raw_Baro_1"].tolist() == pytest.approx([1000.0, 1030.48, 1060.96])


def test_missing_file_returns_none_without_marker(tmp_path):
    result = local_file_in.fetch_data("nope", "unused", {"path": str(tmp_path)})

    assert result is None
    assert list(tmp_path.iterdir()) == []


def test_missing_path_raises():
    with pytest.raises(ValueError, match="local_file_in"):
        fetch_local_file("f1", {}, local_file_in.Config)


def test_resolve_flight_file_prefers_parquet(tmp_path):
    _write_canonical_csv(tmp_path, "f1")
    pd.DataFrame({"time": [0.0]}).to_parquet(tmp_path / "f1.parquet", index=False)

    resolved = resolve_flight_file(tmp_path, "f1")

    assert resolved.suffix == ".parquet"
    assert resolve_flight_file(tmp_path, "missing") is None


def test_filter_option_filters_rows(tmp_path):
    _write_canonical_csv(tmp_path, "f4", operator=["a", "b", "a"])

    df = local_file_in.fetch_data("f4", "unused", {"path": str(tmp_path), "filter": {"operator": "a"}})

    assert df is not None
    assert len(df) == 2


def test_filter_option_removing_all_rows_returns_none(tmp_path):
    _write_canonical_csv(tmp_path, "f5")

    result = local_file_in.fetch_data("f5", "unused", {"path": str(tmp_path), "filter": {"lat": 0.0}})

    assert result is None


def test_to_epoch_seconds_parses_mixed_iso_fractional_seconds():
    series = pd.Series([
        "2023-10-01T08:29:47+00:00",
        "2023-10-01T08:29:47.049999952+00:00",
        "2023-10-01T08:29:47.250000+00:00",
    ])

    assert to_epoch_seconds(series).tolist() == pytest.approx(
        [1696148987.0, 1696148987.05, 1696148987.25], abs=1e-6)


def test_to_epoch_seconds_falls_back_for_non_iso_values():
    series = pd.Series(["06.07.2021 08:14:00", "2023-10-01T08:29:47+00:00"])

    assert to_epoch_seconds(series).notna().sum() == 2


def test_configured_source_names_are_mapped(tmp_path):
    pd.DataFrame({
        "TimeStamp": [0.0, 1.0],
        "IMU_Lat": [51.0, 51.1],
        "IMU_Lon": [6.0, 6.1],
        "IMU_Alt": [1000.0, 1001.0],
        "Baro_A": [3280.84, 3280.84],  # ft
    }).to_csv(tmp_path / "f6.csv", index=False)
    options = {
        "path": str(tmp_path),
        "columns": {"time": "TimeStamp", "lat": "IMU_Lat", "lon": "IMU_Lon", "geo_alt": "IMU_Alt"},
        "baro_params": ["Baro_A"],
    }

    df = local_file_in.fetch_data("f6", "unused", options)

    assert df is not None
    assert df["time"].tolist() == [0.0, 1.0]
    assert df["lat"].tolist() == [51.0, 51.1]
    assert df["lon"].tolist() == [6.0, 6.1]
    assert df["geo_alt"].tolist() == [1000.0, 1001.0]
    assert df["baro_alt"].tolist() == pytest.approx([1000.0, 1000.0])


def test_local_public_ref_uses_position_fix_time(tmp_path):
    pd.DataFrame({
        "time": pd.to_datetime([1696153787, 1696153788, 1696153789], unit="s", utc=True),
        "lat": [51.4, 51.5, 51.6],
        "lon": [6.7, 6.8, 6.9],
        "geoaltitude": [1000.0, 1100.0, 1200.0],
        "lastposupdate": [1696153786.5, 1696153786.5, 1696153788.5],
    }).to_csv(tmp_path / "f7.csv", index=False)

    result = local_public_ref.fetch_data("f7", "unused", {"path": str(tmp_path)})

    assert result is not None
    assert result["time"].tolist() == [1696153786.5, 1696153786.5, 1696153788.5]
