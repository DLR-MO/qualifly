# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

import pandas as pd
import pytest

from qualifly_lib.flight_processor import (
    _at_reports, _target_grid, find_stored_data, preprocess_data, process_flight,
)


class _Source:
    TIME = "time"
    LAT = "lat"
    LON = "lon"
    GEOALT = "geo_alt"
    IDENTIFIER = "sourceid"


class _Target:
    TIME = "time"
    LAT = "lat"
    LON = "lon"
    GEOALT = "geo_alt"
    IDENTIFIER = "targetid"


def _config():
    return {"source": _Source(), "target": _Target()}


def _geo_frame(times, scale=1.0):
    return pd.DataFrame({
        "time": times,
        "lat": [scale * t for t in times],
        "lon": [2.0 * scale * t for t in times],
        "geo_alt": [100.0 * t for t in times],
    })


def _reports(times, scale=10.0):
    """Target reports at `times` (seconds), with datetime timestamps."""
    frame = _geo_frame(times, scale=scale)
    frame["time"] = pd.to_datetime(times, unit="s", utc=True)
    return frame


def _seconds(times):
    return pd.DatetimeIndex(pd.to_datetime(times, unit="s", utc=True), name="timestamp")


def test_preprocess_data_aligns_overlapping_timestamps(tmp_path):
    (tmp_path / "flight1").mkdir()

    _, _, aligned_source, aligned_target = preprocess_data(
        _geo_frame([0, 1, 2, 3]), _reports([1, 2, 3, 4]), _config(), str(tmp_path), "flight1"
    )

    assert aligned_source is not None and aligned_target is not None
    assert len(aligned_source) == len(aligned_target) == 3     # the report at 4 s is outside the source
    assert aligned_source.index.equals(aligned_target.index)
    assert (tmp_path / "flight1" / "sourceid.csv").exists()
    assert (tmp_path / "flight1" / "targetid.csv").exists()


def test_preprocess_data_returns_none_alignment_when_timestamps_do_not_overlap(tmp_path):
    (tmp_path / "flight2").mkdir()

    _, _, aligned_source, aligned_target = preprocess_data(
        _geo_frame([0, 1, 2]), _reports([1000, 1001, 1002]), _config(), str(tmp_path), "flight2"
    )

    assert aligned_source is None
    assert aligned_target is None


def test_preprocess_data_accepts_iso_and_epoch_times(tmp_path):
    (tmp_path / "flight3").mkdir()
    source_df = _geo_frame([0, 1, 2, 3])
    source_df["time"] = pd.to_datetime([0, 1, 2, 3], unit="s", utc=True).astype(str)
    target_df = _geo_frame([1.0, 2.0, 3.0], scale=10.0)                # epoch seconds

    _, _, aligned_source, aligned_target = preprocess_data(
        source_df, target_df, _config(), str(tmp_path), "flight3"
    )

    assert aligned_source is not None and aligned_target is not None
    assert len(aligned_source) == len(aligned_target) == 3


def test_preprocess_data_optional_prepare_hook(tmp_path):
    (tmp_path / "flight4").mkdir()

    class _PreparingSource(_Source):
        @staticmethod
        def prepare_data(df):
            df = df.copy()
            df["prepared"] = 1.0
            return df

    config = {"source": _PreparingSource(), "target": _Target()}

    source_out, _, aligned_source, _ = preprocess_data(
        _geo_frame([0, 1, 2]), _reports([1, 2]), config, str(tmp_path), "flight4"
    )

    assert aligned_source is not None
    assert (source_out["prepared"] == 1.0).all()


def test_preprocess_data_interpolates_the_source_at_the_report_times(tmp_path):
    (tmp_path / "flight5").mkdir()

    source_track, target_reports, aligned_source, aligned_target = preprocess_data(
        _geo_frame([0.0, 0.8, 1.6, 2.4, 3.2]), _reports([0.5, 2.5]), _config(), str(tmp_path), "flight5"
    )

    # the time base is the reports themselves
    assert aligned_source.index.equals(_seconds([0.5, 2.5]))
    assert aligned_target.index.equals(_seconds([0.5, 2.5]))
    assert aligned_source["lat"].tolist() == pytest.approx([0.5, 2.5])
    assert aligned_source["lon"].tolist() == pytest.approx([1.0, 5.0])
    assert aligned_target["lat"].tolist() == pytest.approx([5.0, 25.0])
    # source and reports themselves are returned as delivered
    assert source_track["lat"].tolist() == pytest.approx([0.0, 0.8, 1.6, 2.4, 3.2])
    assert target_reports["lat"].tolist() == pytest.approx([5.0, 25.0])


def test_preprocess_data_uses_the_real_samples_of_a_held_source_position(tmp_path):
    # a 1 Hz GNSS fix at x.1 s repeated in 20 Hz rows, next to a 20 Hz barometric altitude
    (tmp_path / "flight5h").mkdir()
    times = [round(0.05 * k, 2) for k in range(2, 120)]
    fixes = [int(round(t - 0.1, 2)) + 0.1 for t in times]
    source_df = pd.DataFrame({
        "time": times,
        "lat": fixes, "lon": [2.0 * f for f in fixes], "geo_alt": [100.0 * f for f in fixes],
        "baro_alt": [100.0 * t for t in times],
    })

    track, _, aligned_source, _ = preprocess_data(
        source_df, _reports([1.5, 2.7, 4.0], scale=1.0), _config(), str(tmp_path), "flight5h"
    )

    assert aligned_source["lat"].tolist() == pytest.approx([1.5, 2.7, 4.0])
    assert aligned_source["geo_alt"].tolist() == pytest.approx([150.0, 270.0, 400.0])
    assert track["lat"].notna().sum() == 6                   # the fixes at 0.1 ... 5.1 s
    assert track["baro_alt"].notna().all()                  # its own samples are all kept


def test_preprocess_data_keeps_a_constant_altitude_of_a_moving_position(tmp_path):
    (tmp_path / "flight5l").mkdir()
    source_df = _geo_frame([0, 1, 2, 3])
    source_df["geo_alt"] = 1000.0                           # level flight

    track, _, aligned_source, _ = preprocess_data(
        source_df, _reports([1, 2]), _config(), str(tmp_path), "flight5l"
    )

    assert track["geo_alt"].tolist() == [1000.0] * 4
    assert aligned_source["geo_alt"].tolist() == [1000.0, 1000.0]


def test_target_grid_interpolates_the_filtered_columns_and_holds_the_rest(tmp_path):
    reports = _reports([0.5, 1.5, 11.5, 12.5]).set_index("time")
    reports["velocity"] = [None, 110.0, None, 120.0]
    reports["heading"] = [None, 90.0, None, 100.0]

    grid = _target_grid(reports, _Target())

    assert grid.index.equals(_seconds(list(range(1, 13))))
    # the position is interpolated, also across the 10 s report gap, and so is the ground speed
    assert grid["lat"].tolist() == pytest.approx([10.0 * t for t in range(1, 13)])
    speeds = [110.0 + (t - 1.5) * 10 / 11 for t in range(2, 13)]
    assert grid["velocity"].tolist() == pytest.approx(speeds[:1] + speeds)      # before 1.5 s the first grid value
    # every other column takes its last reported value
    assert grid["heading"].tolist() == [90.0] * 12


def test_at_reports_applies_the_filter_correction_at_the_report_times():
    reports = _reports([0.5, 1.5, 2.5, 3.5]).set_index("time")
    grid = _target_grid(reports, _Target())

    unchanged = _at_reports(grid.copy(), grid, reports, _Target())
    shifted = grid.copy()
    shifted["lat"] += 1.0

    assert unchanged[["lat", "lon", "geo_alt"]].equals(reports[["lat", "lon", "geo_alt"]])
    assert _at_reports(shifted, grid, reports, _Target())["lat"].tolist() == pytest.approx(
        (reports["lat"] + 1.0).tolist())


def test_at_reports_keeps_the_neighbours_of_a_removed_outlier():
    # a filter that removes the outlier completely must not move the reports next to it
    times = [t + 0.5 for t in range(11)]
    clean = _reports(times).set_index("time")
    spiked = clean.copy()
    spiked.iloc[5, spiked.columns.get_loc("geo_alt")] += 5000.0

    variant = _at_reports(_target_grid(clean, _Target()), _target_grid(spiked, _Target()), spiked, _Target())

    assert variant["geo_alt"].tolist() == pytest.approx(clean["geo_alt"].tolist())


def test_preprocess_data_drops_reports_in_large_source_gaps(tmp_path):
    (tmp_path / "flight6").mkdir()

    _, _, aligned_source, aligned_target = preprocess_data(
        _geo_frame([0.0, 10.0]), _reports([0, 5, 10]), _config(), str(tmp_path), "flight6"
    )

    # Seconds inside the 10s source gap have no reference; exact sample hits stay valid
    assert aligned_source is not None and aligned_target is not None
    assert aligned_source.index.equals(_seconds([0, 10]))
    assert aligned_source["lat"].tolist() == pytest.approx([0.0, 10.0])
    assert aligned_target["lat"].tolist() == pytest.approx([0.0, 100.0])


def test_preprocess_data_respects_max_gap_s_from_config(tmp_path):
    (tmp_path / "flight7a").mkdir()
    (tmp_path / "flight7b").mkdir()
    source_df = _geo_frame([0.0, 10.0])

    _, _, strict_source, _ = preprocess_data(
        source_df.copy(), _reports([4, 6]), {**_config(), "max_gap_s": 2.0}, str(tmp_path), "flight7a")
    _, _, loose_source, _ = preprocess_data(
        source_df.copy(), _reports([4, 6]), {**_config(), "max_gap_s": 12.0}, str(tmp_path), "flight7b")

    assert strict_source is None
    assert loose_source is not None
    assert loose_source["lat"].tolist() == pytest.approx([4.0, 6.0])


def _position_frame(geo_alt):
    return pd.DataFrame({
        "time": [0.0, 1.0, 2.0, 3.0],
        "lat": [52.0, 52.001, 52.002, 52.003],
        "lon": [10.0, 10.001, 10.002, 10.003],
        "geo_alt": [geo_alt] * 4,
    })


def test_preprocess_data_converts_a_flagged_msl_altitude_but_caches_the_delivered_values(tmp_path, monkeypatch):
    monkeypatch.setattr("qualifly_lib.utils.geoid_undulation", lambda lat, lon: [45.0] * len(lat))
    (tmp_path / "flight10").mkdir()

    _, _, aligned_source, aligned_target = preprocess_data(
        _position_frame(100.0), _position_frame(145.0),
        {**_config(), "source_geo_alt_is_msl": True}, str(tmp_path), "flight10",
    )

    assert aligned_source["geo_alt"].tolist() == pytest.approx([145.0] * len(aligned_source))
    assert aligned_target["geo_alt"].tolist() == pytest.approx([145.0] * len(aligned_target))
    # the cache keeps what the backend delivered, so the flag can change without a refetch
    cached = pd.read_csv(tmp_path / "flight10" / "sourceid.csv")
    assert cached["geo_alt"].tolist() == [100.0] * 4


def test_process_flight_rejects_a_non_boolean_geo_alt_is_msl(tmp_path):
    with pytest.raises(ValueError, match="geo_alt_is_msl"):
        process_flight(
            "flight11", "local_file_in", "local_public_ref", base_path=str(tmp_path),
            fetch_options={"local_file_in": {"geo_alt_is_msl": "yes"}},
        )


@pytest.mark.parametrize("max_gap_s", [-1.0, "2"])
def test_process_flight_rejects_an_invalid_max_gap(tmp_path, max_gap_s):
    with pytest.raises(ValueError, match="max_gap_s"):
        process_flight("flight13", "local_file_in", "local_public_ref", base_path=str(tmp_path),
                       max_gap_s=max_gap_s)


def test_process_flight_rejects_a_non_positive_track_window(tmp_path):
    with pytest.raises(ValueError, match="track_window_s"):
        process_flight("flight12", "local_file_in", "local_public_ref", base_path=str(tmp_path),
                       track_window_s=0)


def test_find_stored_data_lists_only_what_the_pipeline_wrote(tmp_path):
    def file(*parts):
        path = tmp_path.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")

    file("flightA", "local_file_in.csv")             # cached download
    file("flightB", "opensky_not_found.txt")         # marker only
    file("flightC", "scores.csv")                    # results
    (tmp_path / "flightD").mkdir()                   # created, nothing fetched yet
    file("store", "in", "flightA.parquet")           # local input files, nested
    file("inputs", "flightA.csv")                    # local input files, flat
    file("summary.pq")
    file("failed_flights.txt")
    file("notes.txt")

    flight_dirs, files = find_stored_data(str(tmp_path))

    assert [folder.name for folder in flight_dirs] == ["flightA", "flightB", "flightC", "flightD"]
    assert [path.name for path in files] == ["summary.pq", "failed_flights.txt"]
