# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

import pandas as pd

from qualifly_lib.data_fetching.opensky import _query_with_retry, best_matching_flight


def test_retry_recovers_from_transient_attribute_error(capsys):
    calls = []

    def flaky_query():
        calls.append(1)
        if len(calls) < 3:
            raise AttributeError("'NoneType' object has no attribute 'stats'")
        return "dataframe"

    result = _query_with_retry(flaky_query, "flightlist", attempts=3)

    assert result == "dataframe"
    assert len(calls) == 3
    assert "attempt 1/3 failed" in capsys.readouterr().out


def test_retry_returns_none_when_all_attempts_fail(capsys):
    calls = []

    def always_fails():
        calls.append(1)
        raise AttributeError("'NoneType' object has no attribute 'stats'")

    result = _query_with_retry(always_fails, "history", attempts=3)

    assert result is None
    assert len(calls) == 3
    assert "unavailable after 3 attempts" in capsys.readouterr().out


def test_retry_does_not_swallow_other_exceptions():
    def raises_value_error():
        raise ValueError("unexpected bug")

    try:
        _query_with_retry(raises_value_error, "rawdata", attempts=3)
    except ValueError:
        pass
    else:
        raise AssertionError("ValueError should not be retried or swallowed")


def test_merge_nacp_matches_by_position_fix_time():
    import pandas as pd
    from qualifly_lib.data_fetching.opensky import merge_nacp

    states = pd.DataFrame({"time": [100.4, 101.4, 102.6, 130.0]})
    raw = pd.DataFrame({
        "mintime": [101.0, 102.5],
        "positionnac": [8, 9],
        "geometricverticalaccuracy": [2, 2],
    })

    merged = merge_nacp(states, raw)

    # 100.4 precedes the first message -> backfilled; 130.0 is > 15 s after the last
    assert merged["positionnac"].tolist()[:3] == [8, 8, 9]
    assert pd.isna(merged["positionnac"].iloc[3])


def test_merge_nacp_does_not_backfill_a_late_claim():
    import pandas as pd
    from qualifly_lib.data_fetching.opensky import merge_nacp

    states = pd.DataFrame({"time": [100.4, 103.0, 120.0, 140.0]})
    raw = pd.DataFrame({
        "mintime": [101.0, 115.0],
        "positionnac": [8, 9],
        "nacv": [None, 4],  # only the second message claims a NACv
    })

    merged = merge_nacp(states, raw)

    # The leading row takes the first message, which has no NACv claim; the
    # late NACv only applies to the report matched to its message.
    assert merged["positionnac"].tolist()[:3] == [8, 8, 9]
    assert pd.isna(merged["positionnac"].iloc[3])
    assert merged["nacv"].isna().tolist() == [True, True, False, True]


def test_merge_nacp_ignores_surface_operational_status():
    import pandas as pd
    from qualifly_lib.data_fetching.opensky import merge_nacp

    states = pd.DataFrame({"time": [100.4, 101.4, 102.6]})
    raw = pd.DataFrame({
        "mintime": [101.0, 102.5],
        "positionnac": [8, 3],
        "geometricverticalaccuracy": [2, -1],
        "nacv": [None, 4],
        "subtypecode": [0, 1],
    })

    merged = merge_nacp(states, raw)

    # The surface message (subtype 1) must not contribute any claim
    assert merged["positionnac"].tolist() == [8, 8, 8]
    assert merged["geometricverticalaccuracy"].tolist() == [2, 2, 2]
    assert merged["nacv"].isna().all()


def test_merge_nacp_reads_the_pyarrow_timestamps_pyopensky_returns():
    import pandas as pd
    from qualifly_lib.data_fetching.opensky import merge_nacp

    states = pd.DataFrame({"time": [100.4, 101.4, 102.6]})
    raw = pd.DataFrame({
        "mintime": pd.to_datetime([101.0, 102.5], unit="s", utc=True),
        "positionnac": [8, 9],
        "geometricverticalaccuracy": [2, 2],
    }).convert_dtypes(dtype_backend="pyarrow")

    assert merge_nacp(states, raw)["positionnac"].tolist() == [8, 8, 9]


def test_best_matching_flight_skips_the_previous_leg():
    # previous leg lands after the source start and comes first in the list
    flight_list = pd.DataFrame({
        "firstseen": pd.to_datetime([1000, 5000, 20000], unit="s", utc=True),
        "lastseen": pd.to_datetime([4200, 9000, 25000], unit="s", utc=True),
        "callsign": ["LEG1", "LEG2", "LEG3"],
    })

    flight = best_matching_flight(flight_list, start=4000, stop=9500)

    assert flight["callsign"] == "LEG2"


def test_merge_nacp_survives_a_claim_missing_from_the_whole_flight():
    import subprocess
    import sys

    # the pyarrow null column used to abort the process, so run it in a child
    code = """
import pandas as pd, pyarrow as pa
from qualifly_lib.data_fetching.opensky import merge_nacp
states = pd.DataFrame({"time": [0.0, 10.0, 20.0]})
raw = pd.DataFrame({
    "mintime": pd.to_datetime([15, 25], unit="s", utc=True),
    "positionnac": pd.array([8, 9], dtype=pd.ArrowDtype(pa.int64())),
    "nacv": pd.array([None, None], dtype=pd.ArrowDtype(pa.null())),
})
merged = merge_nacp(states, raw)
assert merged["positionnac"].tolist() == [8, 8, 8], merged
assert merged["nacv"].isna().all(), merged
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_best_matching_flight_rejects_legs_that_do_not_overlap():
    # only the previous leg is listed: it lands before the source start
    flight_list = pd.DataFrame({
        "firstseen": pd.to_datetime([1000], unit="s", utc=True),
        "lastseen": pd.to_datetime([3900], unit="s", utc=True),
    })

    assert best_matching_flight(flight_list, start=4000, stop=9500) is None
