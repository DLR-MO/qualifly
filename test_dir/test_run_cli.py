# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

import pytest

import run
from run import load_ids_file, parse_args


def test_load_ids_file_ignores_blank_lines_and_comments(tmp_path):
    id_file = tmp_path / "ids.txt"
    id_file.write_text("flightA\n\n# asset list\nflightB\n", encoding="utf-8")

    assert load_ids_file(str(id_file)) == ["flightA", "flightB"]


def test_parse_args_requires_ids_or_file():
    with pytest.raises(SystemExit):
        parse_args([])


def test_parse_args_accepts_file(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run.py", "--file", "ids.txt"])
    args = parse_args()
    assert args.file == "ids.txt" and not args.ids


def test_parse_args_rejects_file_with_ids(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run.py", "--file", "ids.txt", "id1"])
    with pytest.raises(SystemExit):
        parse_args()


def test_parse_args_rejects_export_with_ids(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run.py", "--export", "id1"])
    with pytest.raises(SystemExit):
        parse_args()


def test_main_takes_max_gap_s_from_the_config(monkeypatch):
    calls = []
    monkeypatch.setattr("sys.argv", ["run.py", "id1"])
    monkeypatch.setattr(run, "load_config", lambda: {"source": "a", "target": "b", "max_gap_s": 4.0})
    monkeypatch.setattr(run, "process_flight", lambda *args, **kwargs: calls.append(kwargs))

    run.main()

    assert calls[0]["max_gap_s"] == 4.0


def _stored_data(tmp_path):
    data = tmp_path / "data"
    (data / "flightA").mkdir(parents=True)
    (data / "flightA" / "opensky.csv").write_text("time\n", encoding="utf-8")
    (data / "summary.pq").write_text("x", encoding="utf-8")
    (data / "store" / "in").mkdir(parents=True)
    (data / "store" / "in" / "flightA.parquet").write_text("x", encoding="utf-8")
    return data


def test_clean_removes_the_stored_data_after_confirmation(tmp_path, monkeypatch):
    data = _stored_data(tmp_path)
    monkeypatch.setattr("builtins.input", lambda prompt: "Yes")

    assert run.clean_data(str(data)) is True

    assert not (data / "flightA").exists()
    assert not (data / "summary.pq").exists()
    # local input files are not stored by the pipeline and stay
    assert (data / "store" / "in" / "flightA.parquet").exists()


@pytest.mark.parametrize("answer", ["n", ""])
def test_clean_keeps_everything_without_confirmation(tmp_path, monkeypatch, answer):
    data = _stored_data(tmp_path)
    monkeypatch.setattr("builtins.input", lambda prompt: answer)

    assert run.clean_data(str(data)) is False

    assert (data / "flightA" / "opensky.csv").exists()
    assert (data / "summary.pq").exists()


def test_clean_treats_a_closed_input_as_no(tmp_path, monkeypatch):
    data = _stored_data(tmp_path)

    def closed(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", closed)

    assert run.clean_data(str(data)) is False
    assert (data / "flightA").exists()
