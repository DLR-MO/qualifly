# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

import pandas as pd

from qualifly_lib.data_fetching import data_fetcher


def _write_source(tmp_path, collection_id="f1"):
    pd.DataFrame({
        "time": [0.0, 1.0, 2.0],
        "lat": [51.0, 51.1, 51.2],
        "lon": [6.0, 6.1, 6.2],
        "geo_alt": [1000.0, 1001.0, 1002.0],
    }).to_csv(tmp_path / f"{collection_id}.csv", index=False)


def test_use_local_or_fetch_skips_cache_for_local_files(tmp_path):
    _write_source(tmp_path)
    # a stale per-flight cache copy must be ignored (the local file is the store)
    flight_dir = tmp_path / "f1"
    flight_dir.mkdir()
    (flight_dir / "local_file_in.csv").write_text("Time\n999\n", encoding="utf-8")

    df = data_fetcher.use_local_or_fetch_data(str(tmp_path), "f1", "local_file_in", {"path": str(tmp_path)})

    assert df is not None
    assert df["lat"].tolist() == [51.0, 51.1, 51.2]
