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