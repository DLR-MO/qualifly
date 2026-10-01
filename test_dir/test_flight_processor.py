import pandas as pd

from qualifly_lib.flight_processor import preprocess_data


class _Source:
    TIME = "time"
    IDENTIFIER = "sourceid"


class _Target:
    TIME = "time"
    IDENTIFIER = "targetid"


def _config():
    return {"source": _Source(), "target": _Target()}


def test_preprocess_data_aligns_overlapping_timestamps(tmp_path):
    (tmp_path / "flight1").mkdir()
    source_df = pd.DataFrame({"time": [0, 1, 2, 3], "value": [1.0, 2.0, 3.0, 4.0]})  # unix seconds
    target_df = pd.DataFrame({
        "time": pd.to_datetime([1, 2, 3, 4], unit="s", utc=True),
        "value": [10.0, 20.0, 30.0, 40.0],
    })

    _, _, aligned_source, aligned_target = preprocess_data(
        source_df, target_df, _config(), str(tmp_path), "flight1"
    )

    assert aligned_source is not None and aligned_target is not None
    assert len(aligned_source) == len(aligned_target) > 0
    assert aligned_source.index.equals(aligned_target.index)
    assert (tmp_path / "flight1" / "sourceid.csv").exists()
    assert (tmp_path / "flight1" / "targetid.csv").exists()


def test_preprocess_data_returns_none_alignment_when_timestamps_do_not_overlap(tmp_path):
    (tmp_path / "flight2").mkdir()
    source_df = pd.DataFrame({"time": [0, 1, 2], "value": [1.0, 2.0, 3.0]})
    target_df = pd.DataFrame({
        "time": pd.to_datetime([1000, 1001, 1002], unit="s", utc=True),
        "value": [10.0, 20.0, 30.0],
    })

    _, _, aligned_source, aligned_target = preprocess_data(
        source_df, target_df, _config(), str(tmp_path), "flight2"
    )

    assert aligned_source is None
    assert aligned_target is None
