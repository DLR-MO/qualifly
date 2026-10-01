from types import SimpleNamespace

import pandas as pd
import pytest

from qualifly_lib.accuracy_comparison import NACP_TABLE, epu_to_nacp, validate_nacp, vepu_to_nacp


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


def test_validate_nacp_returns_none_without_original_data():
    config = {"target": SimpleNamespace(IDENTIFIER="opensky")}

    result = validate_nacp("base", "col1", {"median": pd.DataFrame()}, "median", config)

    assert result is None


def test_validate_nacp_returns_none_without_a_best_variant():
    config = {"target": SimpleNamespace(IDENTIFIER="opensky")}

    result = validate_nacp("base", "col1", {"original": pd.DataFrame()}, None, config)

    assert result is None


def test_validate_nacp_compares_original_and_best_variant(tmp_path):
    index = pd.date_range("2024-01-01", periods=3, freq="s", tz="UTC")
    original = pd.DataFrame(
        {"position_error": [1.0, 4.0, 50.0], "positionnac": [11, 10, 8]}, index=index
    )
    best = pd.DataFrame(
        {"position_error": [0.5, 2.0, 20.0], "positionnac": [11, 10, 8]}, index=index
    )
    config = {"target": SimpleNamespace(IDENTIFIER="opensky", POSITIONNAC="positionnac")}
    (tmp_path / "col1").mkdir()

    result = validate_nacp(str(tmp_path), "col1", {"original": original, "median": best}, "median", config)

    assert set(result.keys()) == {"original", "median"}
    assert result["original"]["achieved_epu"].tolist() == [epu_to_nacp(e) for e in [1.0, 4.0, 50.0]]

    csv_path = tmp_path / "col1" / "epu_validation.csv"
    assert csv_path.exists()
    assert len(pd.read_csv(csv_path)) == 3
