# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

"""Parallel processing: dedupe, atomic writes, locked shared files, the runner."""

import threading

import pandas as pd
import pytest

import run
from run import parse_args, unique_ids
from qualifly_lib.summary import SCORES_FILE, SUMMARY_FILE, upsert_flight
from qualifly_lib.utils import atomic_path, log_failed_ids


def test_unique_ids_keeps_the_order_and_drops_duplicates():
    assert unique_ids(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


def test_parse_args_sets_jobs():
    assert parse_args(["id1"]).jobs == 4
    assert parse_args(["-j", "2", "id1"]).jobs == 2
    with pytest.raises(SystemExit):
        parse_args(["-j", "0", "id1"])


def test_atomic_path_replaces_the_target_only_after_a_successful_write(tmp_path):
    target = tmp_path / "cache.csv"
    target.write_text("old", encoding="utf-8")

    with atomic_path(target) as tmp:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("new")

    assert target.read_text(encoding="utf-8") == "new"
    assert not (tmp_path / "cache.csv.tmp").exists()


def test_atomic_path_keeps_the_target_when_the_writer_raises(tmp_path):
    target = tmp_path / "cache.csv"
    target.write_text("old", encoding="utf-8")

    with pytest.raises(ValueError):
        with atomic_path(target) as tmp:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("partial")
            raise ValueError("interrupted")

    assert target.read_text(encoding="utf-8") == "old"
    assert not (tmp_path / "cache.csv.tmp").exists()


def _in_parallel(targets):
    threads = [threading.Thread(target=target) for target in targets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def test_log_failed_ids_keeps_one_line_per_id_under_parallel_calls(tmp_path):
    filepath = tmp_path / "failed_flights.txt"
    ids = [f"flight{i}" for i in range(8)]

    def log_all():
        for collection_id in ids:
            log_failed_ids(str(filepath), collection_id)

    _in_parallel([log_all] * 8)

    lines = filepath.read_text(encoding="utf-8").splitlines()
    assert sorted(lines) == sorted(ids)


def test_upsert_flight_keeps_every_flight_under_parallel_calls(tmp_path):
    flights = [f"flight{i}" for i in range(12)]
    for flight in flights:
        flight_dir = tmp_path / flight
        flight_dir.mkdir()
        pd.DataFrame([{"pos_rmse": 1.0}], index=pd.Index(["original"], name="variant")).to_csv(
            flight_dir / "scores.csv")

    def upsert_batch(offset):
        for flight in flights[offset::4]:
            upsert_flight(tmp_path, flight, "local_file_in", "local_public_ref")

    _in_parallel([lambda offset=offset: upsert_batch(offset) for offset in range(4)])

    summary = pd.read_parquet(tmp_path / SUMMARY_FILE)
    scores = pd.read_parquet(tmp_path / SCORES_FILE)
    assert sorted(summary["collection_id"]) == sorted(flights)
    assert sorted(scores["collection_id"]) == sorted(flights)


def test_main_processes_every_flight_once_even_with_duplicate_ids(monkeypatch):
    calls = []
    # one job: the flights of a parallel run go to worker processes, which the patch does not reach
    monkeypatch.setattr("sys.argv", ["run.py", "-j", "1", "flightA", "flightB", "flightA"])
    monkeypatch.setattr(run, "load_config", lambda: {"source": "a", "target": "b"})
    monkeypatch.setattr(run, "process_flight", lambda flight, *args, **kwargs: calls.append(flight))

    run.main()

    assert sorted(calls) == ["flightA", "flightB"]


def test_main_reports_a_failed_flight_without_stopping_the_others(monkeypatch):
    calls = []

    def fake(flight, *args, **kwargs):
        calls.append(flight)
        if flight == "flightA":
            raise ValueError("boom")

    monkeypatch.setattr("sys.argv", ["run.py", "-j", "1", "flightA", "flightB"])
    monkeypatch.setattr(run, "load_config", lambda: {"source": "a", "target": "b"})
    monkeypatch.setattr(run, "process_flight", fake)

    with pytest.raises(SystemExit):
        run.main()

    assert sorted(calls) == ["flightA", "flightB"]

def test_worker_processes_share_the_lock_of_the_log_files(tmp_path):
    # run.py --jobs runs the flights in processes; the module locks only lock within one
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    filepath = str(tmp_path / "failed_flights.txt")
    ids = [f"flight{i}" for i in range(8)] * 8

    with ProcessPoolExecutor(4, initializer=run._init_worker,
                             initargs=(multiprocessing.Lock(), multiprocessing.Lock())) as pool:
        list(pool.map(log_failed_ids, [filepath] * len(ids), ids))

    lines = (tmp_path / "failed_flights.txt").read_text(encoding="utf-8").splitlines()
    assert sorted(lines) == sorted(set(ids))


def _crash_on_b(flight):
    import os
    if flight == "flightB":
        os._exit(1)  # a worker dying abruptly, like an out-of-memory kill
    return True


def test_run_parallel_fails_only_the_flight_whose_worker_dies():
    outcomes = run.run_parallel(["flightA", "flightB", "flightC"], 2, _crash_on_b)

    assert outcomes == {"flightA": True, "flightB": False, "flightC": True}
