from qualifly_lib.data_fetching.opensky import _query_with_retry


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