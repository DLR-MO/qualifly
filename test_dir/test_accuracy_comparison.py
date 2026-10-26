# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from qualifly_lib.accuracy_comparison import (
    NACP_TABLE, epu_to_nacp, error_to_gva, error_to_nacv, nacp_fulfilment, validate_nacp,
    vepu_to_nacp, velocity_fulfilment, vertical_fulfilment,
)


@pytest.mark.parametrize("error, expected_nacp", [
    (0.5, 11),
    (2.9, 11),
    (3.0, 10),    # exactly at the NACp-11 threshold falls into the next bucket (thresholds are exclusive)
    (5, 10),
    (20, 9),
    (50, 8),
    (1000, 4),
    (20000, 0),   # worse than the loosest bucket (NACp 1, < 18520 m)
])
def test_epu_to_nacp_thresholds(error, expected_nacp):
    assert epu_to_nacp(error) == expected_nacp


def test_vepu_to_nacp_thresholds():
    assert vepu_to_nacp(3.9) == 11
    assert vepu_to_nacp(4.5) == 10
    assert vepu_to_nacp(20) == 9


def test_vepu_to_nacp_is_none_when_worse_than_any_defined_bucket():
    # VEPU is only defined for NACp 9-11 (thresholds 4, 15, 45 m)
    assert vepu_to_nacp(1000) is None


def test_nacp_table_is_ordered_best_to_worst():
    nacp_values = [entry[0] for entry in NACP_TABLE]
    assert nacp_values == sorted(nacp_values, reverse=True)


def test_validate_nacp_returns_none_without_original_data(tmp_path):
    config = {"target": SimpleNamespace(IDENTIFIER="opensky")}

    result = validate_nacp(str(tmp_path), "col1", {"median": pd.DataFrame()}, config)

    assert result is None


def test_validate_nacp_writes_every_variant(tmp_path):
    index = pd.date_range("2024-01-01", periods=3, freq="s", tz="UTC")
    original = pd.DataFrame({
        "position_error": [1.0, 4.0, 50.0], "position_error_synced": [1.0, 2.0, np.nan],
        "track_distance": [0.5, 1.0, 2.0], "positionnac": [11, 10, 9],
    }, index=index)
    median = original.assign(position_error=[0.5, 2.0, 20.0])
    config = {"target": SimpleNamespace(IDENTIFIER="opensky", POSITIONNAC="positionnac")}
    (tmp_path / "col1").mkdir()

    result = validate_nacp(str(tmp_path), "col1", {"original": original, "median": median}, config)

    assert result["claimed_nacp"].tolist() == [11, 10, 9]
    assert result["original_position_error_nacp"].tolist() == [epu_to_nacp(e) for e in [1.0, 4.0, 50.0]]
    assert result["median_position_error_nacp"].tolist() == [epu_to_nacp(e) for e in [0.5, 2.0, 20.0]]
    # a missing error is not a failed claim
    assert np.isnan(result["original_position_error_synced_nacp"].iloc[2])
    assert len(pd.read_csv(tmp_path / "col1" / "epu_validation.csv")) == 3

    shares = nacp_fulfilment(result, "original")
    assert shares["position_error"] == pytest.approx(200 / 3)
    assert shares["position_error_synced"] == pytest.approx(100.0)
    assert shares["track_distance"] == pytest.approx(100.0)


@pytest.mark.parametrize("error, expected_gva", [(45.0, 2), (45.1, 1), (150.0, 1), (151.0, 0)])
def test_error_to_gva_thresholds(error, expected_gva):
    assert error_to_gva(error) == expected_gva


def test_validate_nacp_checks_the_altitude_of_every_variant_against_vepu_and_gva(tmp_path):
    index = pd.date_range("2024-01-01", periods=4, freq="s", tz="UTC")
    original = pd.DataFrame({
        "position_error": [1.0, 1.0, 1.0, 1.0], "altitude_error": [-3.0, 10.0, 50.0, np.nan],
        "positionnac": [11, 10, 8, 9], "gva": [2, 2, 1, 0],
    }, index=index)
    vertical = original.assign(altitude_error=[1.0, 1.0, 1.0, 1.0])
    config = {"target": SimpleNamespace(IDENTIFIER="opensky", POSITIONNAC="positionnac", GVA="gva")}
    (tmp_path / "col1").mkdir()

    result = validate_nacp(str(tmp_path), "col1", {"original": original, "kalman_smoother": vertical}, config)

    assert result["original_altitude_error_nacp"].tolist()[:3] == [11, 10, 0]
    assert result["original_altitude_error_gva"].tolist()[:3] == [2, 2, 1]
    assert result["kalman_smoother_altitude_error"].tolist() == [1.0] * 4
    # VEPU: only the claims of NACp 9-11 count (11 met, 10 met, the missing error not)
    # GVA: only the claims 1 and 2 count (2 met, 2 met, 1 met)
    assert vertical_fulfilment(result, "original") == {"vepu": pytest.approx(100.0), "gva": pytest.approx(100.0)}


def test_validate_nacp_reads_a_gva_given_as_its_bound_in_meters(tmp_path):
    # OSN stores the GVA as the bound: 45 m is GVA 2, 150 m GVA 1, -1 unknown
    index = pd.date_range("2024-01-01", periods=4, freq="s", tz="UTC")
    original = pd.DataFrame({"altitude_error": [10.0, 50.0, 100.0, 1.0], "gva": [45.0, 45.0, 150.0, -1.0]},
                            index=index)
    config = {"target": SimpleNamespace(IDENTIFIER="opensky", GVA="gva")}
    (tmp_path / "col1").mkdir()

    result = validate_nacp(str(tmp_path), "col1", {"original": original}, config)

    assert result["claimed_gva"].tolist() == [2, 2, 1, -1]
    # 10 m meets GVA 2, 50 m misses it, 100 m meets GVA 1, the unknown claim does not count
    assert vertical_fulfilment(result, "original")["gva"] == pytest.approx(200 / 3)


def test_vertical_fulfilment_counts_a_missed_claim_and_is_none_without_one():
    df = pd.DataFrame({"claimed_nacp": [11, 11], "original_altitude_error_nacp": [11, 10]})

    assert vertical_fulfilment(df, "original") == {"vepu": pytest.approx(50.0), "gva": None}


@pytest.mark.parametrize("error, expected_nacv", [
    (0.2, 4),
    (0.3, 3),    # exactly at the NACv-4 threshold falls into the next bucket (thresholds are exclusive)
    (1.0, 2),
    (3.0, 1),
    (10.0, 0),   # worse than the loosest bucket (NACv 1, < 10 m/s)
])
def test_error_to_nacv_thresholds(error, expected_nacv):
    assert error_to_nacv(error) == expected_nacv


def test_validate_nacp_checks_the_ground_speed_of_every_variant(tmp_path):
    index = pd.date_range("2024-01-01", periods=4, freq="s", tz="UTC")
    original = pd.DataFrame({
        "groundspeed_error": [0.2, 3.5, 0.9, np.nan], "nacv": [4, 2, 1, 1],
    }, index=index)
    config = {"target": SimpleNamespace(IDENTIFIER="opensky", NACV="nacv")}
    (tmp_path / "col1").mkdir()

    result = validate_nacp(str(tmp_path), "col1", {"original": original}, config)

    assert result["claimed_nacv"].tolist() == [4, 2, 1, 1]
    assert result["original_groundspeed_error_nacv"].tolist()[:3] == [4, 1, 3]
    # NACv 1 missed, 2 and 1 met, the missing error is not a failed claim
    assert velocity_fulfilment(result, "original")["nacv"] == pytest.approx(200 / 3)


def test_velocity_fulfilment_is_none_without_a_claim():
    df = pd.DataFrame({"original_groundspeed_error_nacv": [4, 4]})

    assert velocity_fulfilment(df, "original") == {"nacv": None}
