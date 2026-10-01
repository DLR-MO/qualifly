import importlib
import os
from enum import StrEnum
from pathlib import Path

import pandas as pd

from qualifly_lib.data_fetching.opensky import fetch_opensky_data
from qualifly_lib.utils import log_failed_ids

DIR = Path(__file__).parent


# Placeholder
class Config(StrEnum):
    IDENTIFIER = "identifier"


def use_local_or_fetch_data(base_path, collection_id, identifier):
    """
    Retrieves data from local storage or sources depending on the identifier which is the keyword for the data source.
    Failed flights are saved in {base_path}/failed_flights.txt
    """
    df = pd.DataFrame()
    path = f"{base_path}/{collection_id}/{identifier}.csv"
    if os.path.exists(path):
        print(f"Load {identifier} data from local path...")
        df = pd.read_csv(path)
        return df

    print(f"Downloading {identifier} data...")

    # Save failed flight in txt file
    if df is None:
        print(f"No flight found for id {collection_id} in {identifier}. See {base_path}/failed_flights.txt for all "
              f"failed flight ids.")
        filepath = f"{base_path}/failed_flights.txt"
        log_failed_ids(filepath, collection_id)

    return df


def load_config(source_identifier, target_identifier) -> dict[str, Config]:
    """
    Returns dict of Config information from source and target as:
        {
            "source": {...},
            "target": {...},
        }
    """
    return {
        "source": _load_config_class(source_identifier),
        "target": _load_config_class(target_identifier),
    }


def _load_config_class(identifier):
    """
    Loads data_fetcher/{identifier}.py.
    Returns the Config information of an identifier from their respective file.
    """
    path = DIR / f"{identifier}.py"

    if not path.is_file():
        available = sorted(
            p.stem for p in DIR.glob("*.py") if p.stem != "__init__"
        )
        raise ValueError(
            f"No data fetcher module for '{identifier}' in data_fetcher directory."
            f"Available: {available}"
        )

    module = importlib.import_module(f"qualifly_lib.data_fetching.{identifier}")

    # Get Config class from {identifier} module
    config_cls = getattr(module, "Config", None)
    if config_cls is None:
        raise AttributeError(f"'data_fetcher.{identifier}' has no 'Config' class")

    return config_cls
