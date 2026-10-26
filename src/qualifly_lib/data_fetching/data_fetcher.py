# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

import importlib
import os
from enum import StrEnum
from pathlib import Path

from qualifly_lib.utils import log_failed_ids, map_to_config_columns

DIR = Path(__file__).parent


# Placeholder
class Config(StrEnum):
    IDENTIFIER = "identifier"


def use_local_or_fetch_data(base_path, collection_id, identifier, options=None, flight_info=None):
    """
    Retrieves data from local storage or sources depending on the identifier which is the keyword for the data source.
    `options` (data_fetching.{identifier} of config.yaml) and `flight_info` are passed to the backend's fetch_data().
    Failed flights are saved in {base_path}/failed_flights.txt
    """
    import pandas as pd

    options = options or {}
    module = _load_fetcher_module(identifier)
    cache_data = getattr(module, "CACHE_DATA", True)

    if cache_data:
        path = f"{base_path}/{collection_id}/{identifier}.csv"
        if os.path.exists(path):
            print(f"Load {identifier} data from local path...")
            try:
                return map_to_config_columns(pd.read_csv(path), module.Config, options)
            except pd.errors.EmptyDataError:
                print(f"Cached {identifier} data for {collection_id} is empty - removing it and refetching...")
                os.remove(path)

        marker = f"{base_path}/{collection_id}/{identifier}_not_found.txt"
        if os.path.exists(marker):
            print(f"{identifier} data was previously not found for {collection_id} - skipping fetch. See {marker}")
            return None

    print(f"{'Downloading' if cache_data else 'Loading'} {identifier} data...")
    df = module.fetch_data(collection_id, base_path, options, flight_info=flight_info)

    # Save failed flight in txt file
    if df is None:
        print(f"No flight found for id {collection_id} in {identifier}. See {base_path}/failed_flights.txt for all "
              f"failed flight ids.")
        filepath = f"{base_path}/failed_flights.txt"
        log_failed_ids(filepath, collection_id)

    return df


def get_flight_info(identifier, collection_id, options=None):
    """Returns (icao_hex, start, stop) from the backend's optional get_flight_info(), else None."""
    module = _load_fetcher_module(identifier)
    provider = getattr(module, "get_flight_info", None)
    if not callable(provider):
        return None
    return provider(collection_id, options)


def backend_caches_data(identifier):
    """True when the backend uses the per-flight CSV cache (CACHE_DATA, default True)."""
    module = _load_fetcher_module(identifier)
    return getattr(module, "CACHE_DATA", True)


def load_config(source_identifier, target_identifier) -> dict[str, Config]:
    """
    Returns dict of Config information from source and target
    """
    return {
        "source": _load_config_class(source_identifier),
        "target": _load_config_class(target_identifier),
    }


def available_identifiers():
    """All selectable backends (module files without a leading underscore)."""
    return sorted(
        p.stem for p in DIR.glob("*.py") if p.stem != "__init__" and not p.stem.startswith("_")
    )


def _load_fetcher_module(identifier):
    """
    Loads data_fetcher/{identifier}.py which must provide fetch_data(collection_id, base_path, options).
    """
    if identifier.startswith("_") or not (DIR / f"{identifier}.py").is_file():
        raise ValueError(
            f"No data fetcher module for '{identifier}' in data_fetcher directory."
            f"Available: {available_identifiers()}"
        )

    module = importlib.import_module(f"qualifly_lib.data_fetching.{identifier}")
    if not hasattr(module, "fetch_data"):
        raise AttributeError(f"'data_fetcher.{identifier}' has no 'fetch_data' function")
    return module


def _load_config_class(identifier):
    """
    Loads data_fetcher/{identifier}.py.
    Returns the Config information of an identifier from their respective file.
    """
    module = _load_fetcher_module(identifier)

    # Get Config class from {identifier} module
    config_cls = getattr(module, "Config", None)
    if config_cls is None:
        raise AttributeError(f"'data_fetcher.{identifier}' has no 'Config' class")

    return config_cls
