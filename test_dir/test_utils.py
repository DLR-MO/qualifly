import pandas as pd
import pytest

from qualifly_lib.utils import AttributeConversion, M_TO_FT, apply, log_failed_ids


def test_apply_forward_conversion_adds_target_column():
    df = pd.DataFrame({"altitude_m": [0.0, 100.0]})
    conversions = [AttributeConversion("altitude_m", "altitude_ft", M_TO_FT)]

    result = apply(df, conversions)

    assert result["altitude_ft"].tolist() == pytest.approx([0.0, 100.0 * M_TO_FT])
    assert "altitude_ft" not in df.columns  # apply() works on a copy


def test_apply_inverse_conversion_round_trips():
    # apply() only touches a pair whose *source* column is already present, in both
    # directions -- so the inverse pass expects the source column to already exist
    # (e.g. because the forward pass added the target column earlier in the
    # pipeline) and refreshes it from the target column.
    df = pd.DataFrame({"altitude_m": [0.0, 0.0], "altitude_ft": [0.0, 328.084]})
    conversions = [AttributeConversion("altitude_m", "altitude_ft", M_TO_FT)]

    result = apply(df, conversions, inverse=True)

    assert result["altitude_m"].tolist() == pytest.approx([0.0, 100.0], abs=1e-3)


def test_apply_skips_conversions_whose_source_column_is_missing():
    df = pd.DataFrame({"other": [1, 2]})
    conversions = [AttributeConversion("missing", "target", 2.0)]

    result = apply(df, conversions)

    assert "target" not in result.columns
    assert result["other"].tolist() == [1, 2]


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
