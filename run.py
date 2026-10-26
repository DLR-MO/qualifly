# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

import argparse
import multiprocessing
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from pathlib import Path

import yaml

from qualifly_lib import process_flight

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.yaml")


class _PrefixedLog:
    """
    sys.stdout stand-in for the worker processes of a parallel run: every
    completed line is prefixed with the id of the flight being processed, so
    the output of concurrent flights stays readable and attributable.
    Output without a tag prints unchanged.
    """

    def __init__(self, wrapped):
        self.wrapped = wrapped
        self._local = threading.local()

    def set_tag(self, tag):
        """Marks the output of the calling thread as flight `tag`."""
        self._local.tag = tag
        self._local.pending = ""

    def end_flight(self):
        """Prints the unfinished line of the calling thread, if any."""
        pending = getattr(self._local, "pending", "")
        tag = getattr(self._local, "tag", "")
        if pending:
            self.wrapped.write(f"[{tag}] {pending}\n")
        self._local.pending = ""
        self._local.tag = ""

    def write(self, text):
        tag = getattr(self._local, "tag", "")
        if not tag:
            self.wrapped.write(text)
            return len(text)
        lines = (getattr(self._local, "pending", "") + text).split("\n")
        self._local.pending = lines.pop()
        for line in lines:
            self.wrapped.write(f"[{tag}] {line}\n" if line else "\n")
        return len(text)

    def __getattr__(self, name):
        # flush, fileno, isatty, ... - straight to the wrapped stream
        return getattr(self.wrapped, name)


def load_config():
    """
    Load config.yaml with its attributes source, target and optional settings.
    Relative data_fetching `path` entries are resolved against the directory of config.yaml.
    """
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    for options in (config.get("data_fetching") or {}).values():
        if isinstance(options, dict) and isinstance(options.get("path"), str):
            path = Path(options["path"]).expanduser()
            options["path"] = str(path if path.is_absolute() else Path(CONFIG_PATH).parent / path)
    return config


def load_ids_file(path):
    """
    Read a line-separated list of flight IDs, ignoring empty lines and # comments.
    """
    with open(path, "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]


def unique_ids(ids):
    """
    The ids in their input order, without duplicates: one flight per id, so
    parallel workers never write the same data/<id>/ folder twice.
    """
    return list(dict.fromkeys(ids))


# The prefixed stdout of a worker process of a parallel run (see _init_worker)
_WORKER_LOG = None


def _init_worker(upsert_lock, io_lock):
    """
    Sets up a worker process of a parallel run: the locks of summary.pq and of
    the shared log files become ones of all workers (the module ones only lock
    within a process), and its lines are prefixed with the flight id.
    """
    global _WORKER_LOG
    import faulthandler

    from qualifly_lib import summary, utils

    # a crash in native code kills the worker without a Python traceback;
    # this dumps the Python stack of the crash to stderr
    faulthandler.enable()

    summary._UPSERT_LOCK, utils._IO_LOCK = upsert_lock, io_lock
    # whole lines, so the lines of the workers do not interleave in a redirected output
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    _WORKER_LOG = sys.stdout = _PrefixedLog(sys.stdout)


def run_one_flight(collection_id, config, args, base_path, log=None):
    """
    Processes one flight and returns True on success. A failing flight is
    reported and does not stop the other flights of a run.
    """
    log = log if log is not None else _WORKER_LOG
    if log is not None:
        log.set_tag(collection_id)
    flight_start = time.perf_counter()
    ok = True
    try:
        process_flight(collection_id, config["source"], config["target"], base_path=base_path,
                       fetch_options=config.get("data_fetching"),
                       max_gap_s=config.get("max_gap_s", 2.0),
                       track_window_s=config.get("track_window_s", 2.0),
                       rescore=args.rescore)
    except Exception as e:
        ok = False
        traceback.print_exc(file=sys.stdout)
        print(f"Flight {collection_id} failed: {e}")
    finally:
        print(f"Elapsed time for flight {collection_id}: {time.perf_counter() - flight_start:.2f} seconds")
        if log is not None:
            log.end_flight()
    return ok


def _new_pool(jobs):
    return ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker,
                               initargs=(multiprocessing.Lock(), multiprocessing.Lock()))


def run_parallel(flights, jobs, task):
    """
    Runs task(flight) -> bool for the flights in worker processes and returns
    {flight: outcome}. A worker that dies (out of memory, a crash in native
    code) breaks the whole pool: the unfinished flights are then rerun one
    at a time in their own process, so only the crashing flight fails.
    """
    # processes, not threads: the filters (traffic's Kalman smoother) are
    # pure Python loops, which threads would run one at a time (GIL)
    outcomes = {}
    pool = _new_pool(jobs)
    try:
        futures = {pool.submit(task, flight): flight for flight in flights}
        for future in as_completed(futures):
            outcomes[futures[future]] = future.result()
    except BrokenProcessPool:
        pool.shutdown(wait=False, cancel_futures=True)
        for future, flight in futures.items():
            if future.done() and not future.cancelled() and future.exception() is None:
                outcomes[flight] = future.result()
        remaining = [flight for flight in flights if flight not in outcomes]
        print(f"A worker process died; rerunning the {len(remaining)} unfinished flight(s) "
              f"one at a time to find the one that crashes it.")
        for flight in remaining:
            with _new_pool(1) as single:
                try:
                    outcomes[flight] = single.submit(task, flight).result()
                except BrokenProcessPool:
                    print(f"Flight {flight} failed: its worker process died "
                          f"(out of memory or a crash in native code).")
                    outcomes[flight] = False
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown()
    return outcomes


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Flight Pipeline Starter")
    parser.add_argument(
        "ids",
        nargs="*",
        metavar="id",
        help="One or more flight collection IDs",
    )
    parser.add_argument(
        "-f", "--file",
        metavar="PATH",
        help="Read ids from a line-separated file of flight IDs",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=4,
        metavar="N",
        help="Process N flights in parallel (default: 4)",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Open the result plots in the browser",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Recalculate the results even if they are cached; the cached downloads are reused.",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export the summary parquet to a CSV file (data/summary.csv) and exit.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Print a concise analysis of the summary parquet (with claim coverage and pass counts) and exit.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove all stored data (cached downloads, results, summaries) after a confirmation and exit.",
    )
    args = parser.parse_args(argv)

    if args.jobs < 1:
        parser.error("--jobs must be a positive number of parallel flights.")
    if args.export or args.analyze or args.clean:
        if args.file or args.ids:
            parser.error("--export/--analyze/--clean cannot be combined with ids or --file.")
        return args
    if args.file and args.ids:
        parser.error("--file cannot be combined with explicit ids.")
    if not args.file and not args.ids:
        parser.error("Provide at least one id or use --file.")

    return args


def clean_data(data_path):
    """
    Removes the data the pipeline stored in `data_path` after confirmation.
    Returns True if the data was removed.
    """
    import shutil

    from qualifly_lib.flight_processor import find_stored_data

    flight_dirs, files = find_stored_data(data_path)
    if not flight_dirs and not files:
        print(f"No stored data in {data_path}.")
        return False

    print(f"This removes from {data_path}:")
    print(f"  {len(flight_dirs)} flight folder(s) with cached downloads and results")
    for file in files:
        print(f"  {file.name}")
    try:
        answer = input("Remove them? [y]es | no: ")
    except EOFError:
        answer = ""
    if answer.strip().lower() not in ("y", "yes"):
        print("Nothing removed.")
        return False

    for folder in flight_dirs:
        shutil.rmtree(folder)
    for file in files:
        file.unlink()
    print("Stored data removed.")
    return True


def main():
    start = time.perf_counter()
    args = parse_args()
    config = load_config()

    if args.clean:
        clean_data(os.path.join(SCRIPT_DIR, "data"))
        return

    if args.export:
        from qualifly_lib.summary import export_summary

        data_path = os.path.join(SCRIPT_DIR, "data")
        out_path = os.path.join(data_path, "summary.csv")
        out, n_flights, n_scored = export_summary(data_path, out_path)
        unscored = n_flights - n_scored
        detail = f" ({unscored} without scores)" if unscored else ""
        print(f"Exported {n_scored} scored flight(s){detail} to {out}")
        return

    if args.analyze:
        from qualifly_lib.summary import analyze_summary

        analyze_summary(os.path.join(SCRIPT_DIR, "data"), config["source"], config["target"])
        return

    if args.file:
        flights_to_process = load_ids_file(args.file)
        source_label = f"ids from {args.file}"
    else:
        flights_to_process = args.ids
        source_label = "given ids"

    # Duplicates (given twice or listed twice in a file)
    # would process the same data/<id>/ folder twice at the same time
    flights_to_process = unique_ids(flights_to_process)

    count = len(flights_to_process)
    print(f"--- Start Flight Processing Pipeline: {count} flight{'' if count == 1 else 's'} ({source_label}) ---")

    base_path = os.path.join(SCRIPT_DIR, "data")
    parallel = args.jobs > 1 and count > 1

    outcomes = {}
    if parallel:
        print(f"Processing with up to {args.jobs} flights in parallel "
              f"(each line is prefixed with its flight id).")
        outcomes = run_parallel(flights_to_process, args.jobs,
                                partial(run_one_flight, config=config, args=args, base_path=base_path))
    else:
        for flight in flights_to_process:
            outcomes[flight] = run_one_flight(flight, config, args, base_path)

    if args.interactive:
        # All plots are opened after the run (process_flight keeps its
        # `interactive` flag for direct library use)
        from qualifly_lib.flight_processor import _open_cached_results

        for flight in flights_to_process:
            if outcomes.get(flight):
                _open_cached_results(base_path, flight)

    print("\n--- Finished ---")
    end = time.perf_counter()
    if count != 1:
        print(f"Elapsed time for all flights: {end - start:.2f} seconds")

    failed = [flight for flight in flights_to_process if not outcomes.get(flight, False)]
    if failed:
        print(f"{len(failed)} of {count} flight(s) failed: {', '.join(failed)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
