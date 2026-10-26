# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

import pandas as pd
import pytest

from qualifly_lib.summary import (
    SUMMARY_FILE,
    SCORES_FILE,
    analyze_summary,
    export_summary,
    upsert_flight,
)


SOURCE_COLUMNS = ["timestamp", "lat", "lon"]
OPENSKY_COLUMNS = ["timestamp", "lat", "lon"]


def _write_scores(flight_dir, kalman_pos_rmse=4.5):
    pd.DataFrame([
        {"variant": "original", "pos_rmse": 7.0, "pos_p95_synced": 9.0, "pos_track_max": 300.0,
         "alt_rmse": 40.0, "alt_nmad": 6.0},
        {"variant": "kalman_smoother", "pos_rmse": kalman_pos_rmse, "pos_p95_synced": 5.0, "pos_track_max": 50.0,
         "alt_rmse": 42.0, "alt_nmad": 2.5, "time_offset_s": -0.5},
    ]).to_csv(flight_dir / "scores.csv", index=False)


def _write_epu(flight_dir):
    pd.DataFrame([
        {"claimed_nacp": 10.0, "original_position_error": 5.0,
         "original_position_error_nacp": 11.0, "kalman_smoother_position_error": 4.0,
         "kalman_smoother_position_error_nacp": 11.0,
         "original_track_distance": 1.0, "original_track_distance_nacp": 11.0},
        {"claimed_nacp": 10.0, "original_position_error": 20.0,
         "original_position_error_nacp": 9.0, "kalman_smoother_position_error": 6.0,
         "kalman_smoother_position_error_nacp": 10.0,
         "original_track_distance": 2.0, "original_track_distance_nacp": 11.0},
    ]).to_csv(flight_dir / "epu_validation.csv", index=False)


def _write_coverage(flight_dir, lat, lon):
    pd.DataFrame({
        "timestamp": pd.to_datetime(
            ["2024-06-14 17:49:07+00:00", "2024-06-14 17:49:20+00:00"], utc=True
        ),
        "lat": [lat, lat + 0.1],
        "lon": [lon, lon + 0.1],
    }).to_csv(flight_dir / "local_file_in.csv", index=False)

    pd.DataFrame({
        "timestamp": pd.to_datetime(
            ["2024-06-14 17:49:10+00:00", "2024-06-14 17:49:15+00:00"], utc=True
        ),
        "lat": [lat, lat + 0.1],
        "lon": [lon, lon + 0.1],
    }).to_csv(flight_dir / "local_public_ref.csv", index=False)


def _write_flight_results(base, collection_id):
    flight_dir = base / collection_id
    flight_dir.mkdir(parents=True, exist_ok=True)
    _write_scores(flight_dir)
    _write_epu(flight_dir)
    _write_coverage(flight_dir, 52.0, 13.0)


def test_upsert_creates_summary_and_scores(tmp_path):
    _write_flight_results(tmp_path, "flightA")

    assert upsert_flight(str(tmp_path), "flightA", "local_file_in", "local_public_ref") is True

    summary = pd.read_parquet(tmp_path / SUMMARY_FILE)
    assert len(summary) == 1
    assert summary["collection_id"].iloc[0] == "flightA"
    assert "pos_rmse" not in summary.columns

    scores = pd.read_parquet(tmp_path / SCORES_FILE).set_index("variant")
    assert len(scores) == 2
    assert set(scores["collection_id"]) == {"flightA"}
    assert scores.loc["original", "pos_rmse"] == 7.0
    assert scores.loc["kalman_smoother", "pos_rmse"] == 4.5


def test_summary_contains_coverage_statistics(tmp_path):
    _write_flight_results(tmp_path, "flightA")
    upsert_flight(str(tmp_path), "flightA", "local_file_in", "local_public_ref")

    summary = pd.read_parquet(tmp_path / SUMMARY_FILE)
    row = summary.iloc[0]

    assert summary["local_file_in_start"].iloc[0] == pd.Timestamp("2024-06-14 17:49:07+00:00")
    assert summary["local_file_in_end"].iloc[0] == pd.Timestamp("2024-06-14 17:49:20+00:00")
    assert summary["local_file_in_start_lat"].iloc[0] == pytest.approx(52.0)
    assert summary["local_file_in_start_lon"].iloc[0] == pytest.approx(13.0)
    assert summary["local_file_in_end_lat"].iloc[0] == pytest.approx(52.1)
    assert summary["local_public_ref_start"].iloc[0] == pd.Timestamp("2024-06-14 17:49:10+00:00")
    assert summary["local_public_ref_end_lat"].iloc[0] == pytest.approx(52.1)

    assert row["n_aligned"] == 2


def test_scores_contain_the_claim_fulfilment_of_every_variant(tmp_path):
    _write_flight_results(tmp_path, "flightA")
    upsert_flight(str(tmp_path), "flightA", "local_file_in", "local_public_ref")

    scores = pd.read_parquet(tmp_path / SCORES_FILE).set_index("variant")
    assert scores.loc["original", "nacp_fulfilled"] == pytest.approx(50.0)
    assert scores.loc["kalman_smoother", "nacp_fulfilled"] == pytest.approx(100.0)
    assert scores.loc["original", "nacp_fulfilled_track"] == pytest.approx(100.0)
    # no offset-removed error in the file: no fulfilment either
    assert pd.isna(scores.loc["original", "nacp_fulfilled_synced"])


def test_summary_contains_overlap_coverage_and_report_gaps(tmp_path):
    flight_dir = tmp_path / "flightG"
    flight_dir.mkdir()
    t0 = pd.Timestamp("2024-06-14 17:49:00+00:00")

    def positions(seconds):
        return pd.DataFrame({"timestamp": [t0 + pd.Timedelta(seconds=s) for s in seconds],
                             "lat": 52.0, "lon": 13.0})

    positions(range(0, 21)).to_csv(flight_dir / "local_file_in.csv", index=False)
    # reports at 5-7, 12, 13 and 25 s (after the source), the one at 13 s not compared
    positions([5, 6, 7, 12, 13, 25]).to_csv(flight_dir / "local_public_ref.csv", index=False)
    positions([5, 6, 7, 12])[["timestamp"]].assign(
        original_position_error=1.0).to_csv(flight_dir / "epu_validation.csv", index=False)

    upsert_flight(str(tmp_path), "flightG", "local_file_in", "local_public_ref")

    row = pd.read_parquet(tmp_path / SUMMARY_FILE).iloc[0]
    assert row["overlap_s"] == 15.0
    assert row["coverage"] == pytest.approx(4 / 15)
    assert row["max_report_gap_s"] == 5.0
    assert row["n_dropped"] == 2


def test_upsert_overwrites_existing_row(tmp_path):
    _write_flight_results(tmp_path, "flightA")
    upsert_flight(str(tmp_path), "flightA", "local_file_in", "local_public_ref")

    # Reprocess with improved results -> row must be replaced, not duplicated
    _write_scores(tmp_path / "flightA", kalman_pos_rmse=3.0)
    upsert_flight(str(tmp_path), "flightA", "local_file_in", "local_public_ref")

    summary = pd.read_parquet(tmp_path / SUMMARY_FILE)
    scores = pd.read_parquet(tmp_path / SCORES_FILE)
    assert len(summary) == 1
    assert len(scores) == 2
    assert scores.set_index("variant").loc["kalman_smoother", "pos_rmse"] == 3.0


def test_upsert_records_flight_without_scores(tmp_path):
    (tmp_path / "incomplete").mkdir(parents=True, exist_ok=True)
    (tmp_path / "incomplete" / "local_file_in.csv").write_text("timestamp,x\n0,1\n")

    assert upsert_flight(str(tmp_path), "incomplete", "local_file_in", "local_public_ref") is True
    summary = pd.read_parquet(tmp_path / SUMMARY_FILE)
    assert len(summary) == 1
    assert summary["collection_id"].iloc[0] == "incomplete"
    # no scores.csv -> no variant rows
    assert not (tmp_path / SCORES_FILE).exists()


def test_upsert_removes_the_scores_of_a_flight_that_lost_them(tmp_path):
    _write_flight_results(tmp_path, "flightA")
    _write_flight_results(tmp_path, "flightB")
    upsert_flight(str(tmp_path), "flightA", "local_file_in", "local_public_ref")
    upsert_flight(str(tmp_path), "flightB", "local_file_in", "local_public_ref")

    (tmp_path / "flightA" / "scores.csv").unlink()
    upsert_flight(str(tmp_path), "flightA", "local_file_in", "local_public_ref")

    assert set(pd.read_parquet(tmp_path / SCORES_FILE)["collection_id"]) == {"flightB"}


def test_export_summary_to_csv(tmp_path):
    _write_flight_results(tmp_path, "flightA")
    upsert_flight(str(tmp_path), "flightA", "local_file_in", "local_public_ref")

    out, n_flights, n_scored = export_summary(str(tmp_path))

    assert (n_flights, n_scored) == (1, 1)
    df = pd.read_csv(out)
    # one row per flight and variant
    assert set(df["collection_id"]) == {"flightA"}
    assert df.set_index("variant")["pos_rmse"].to_dict() == {"original": 7.0, "kalman_smoother": 4.5}


def test_export_summary_counts_flights_with_and_without_scores(tmp_path):
    _write_flight_results(tmp_path, "flightA")
    upsert_flight(str(tmp_path), "flightA", "local_file_in", "local_public_ref")
    (tmp_path / "flightB").mkdir()
    upsert_flight(str(tmp_path), "flightB", "local_file_in", "local_public_ref")

    out, n_flights, n_scored = export_summary(str(tmp_path))

    assert (n_flights, n_scored) == (2, 1)
    df = pd.read_csv(out)
    assert set(df["collection_id"]) == {"flightA", "flightB"}
    assert df.loc[df["collection_id"] == "flightB", "variant"].isna().all()


def test_analyze_prints_summary(tmp_path, capsys):
    _write_flight_results(tmp_path, "flightA")
    upsert_flight(str(tmp_path), "flightA", "local_file_in", "local_public_ref")

    analyze_summary(str(tmp_path), "local_file_in", "local_public_ref")
    out = capsys.readouterr().out

    assert "Flights: 1" in out
    assert "kalman_smoother" in out
    assert "4.5 (1/1)" in out  # pos_rmse of kalman_smoother, better than original in 1 of 1 flights
    assert "|time_offset_s|" in out and "0.50" in out  # the amount of the offset, not compared
    assert "time-aligned" in out
    assert "Claim coverage: NACp 1/1" in out
    assert "50.0 (0/1)" in out  # original reaches its claim in 0 of 1 flight
    assert "Start gap" in out


def test_analyze_copes_with_a_summary_without_results(tmp_path, capsys):
    # no scores, no NACp validation, no coverage files: nothing to analyze but no crash
    (tmp_path / "flightN").mkdir()
    upsert_flight(str(tmp_path), "flightN", "local_file_in", "local_public_ref")

    analyze_summary(str(tmp_path), "local_file_in", "local_public_ref")

    assert "Flights: 1" in capsys.readouterr().out


def test_scores_contain_the_nacv_fulfilment(tmp_path):
    flight_dir = tmp_path / "flightV"
    flight_dir.mkdir()
    _write_scores(flight_dir)
    pd.DataFrame([
                                {"claimed_nacp": 10.0, "claimed_nacv": 2.0, "original_groundspeed_error_nacv": 2.0},
        {"claimed_nacp": 10.0, "claimed_nacv": 2.0, "original_groundspeed_error_nacv": 1.0},
    ]).to_csv(flight_dir / "epu_validation.csv", index=False)

    upsert_flight(str(tmp_path), "flightV", "local_file_in", "local_public_ref")

    scores = pd.read_parquet(tmp_path / SCORES_FILE).set_index("variant")
    assert scores.loc["original", "nacv_fulfilled"] == pytest.approx(50.0)
