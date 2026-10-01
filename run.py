import argparse
import os
import time

import yaml

from qualifly_lib import process_flight

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.yaml")


def load_config():
    """
    Load config.yaml with its attributes source and target and optional default list of flights.
    """
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Flight Pipeline Starter")
    parser.add_argument(
        "ids",
        nargs="*",
        metavar="id",
        help="One or more flight collection IDs separated with a space",
    )
    parser.add_argument(
        "--default",
        action="store_true",
        help="Use the FLIGHT_IDS list from config.yaml instead of explicit ids",
    )
    args = parser.parse_args()

    if args.default and args.ids:
        parser.error("--default cannot be combined with explicit ids.")
    if not args.default and not args.ids:
        parser.error("Provide at least one id or use --default.")

    return args


def main():
    start = time.perf_counter()
    args = parse_args()
    config = load_config()

    if args.default:
        # Use default list of flights from config.yaml
        if "FLIGHT_IDS" not in config:
            raise KeyError(f"'FLIGHT_IDS' not found in {CONFIG_PATH}")
        flights_to_process = config["FLIGHT_IDS"]
        source_label = "default list from config.yaml"
    else:
        flights_to_process = args.ids
        source_label = "given ids"

    count = len(flights_to_process)
    print(f"--- Start Flight Pipeline: {count} flight{'' if count == 1 else 's'} ({source_label}) ---")

    for flight in flights_to_process:
        flight_start = time.perf_counter()
        process_flight(flight, config["source"], config["target"], base_path=os.path.join(SCRIPT_DIR, "data"))
        flight_end = time.perf_counter()
        print(f"Elapsed time for flight: {flight_end - flight_start:.2f} seconds")

    print("\n--- Finished ---")
    end = time.perf_counter()
    if count != 1:
        print(f"Elapsed time for all flights: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
