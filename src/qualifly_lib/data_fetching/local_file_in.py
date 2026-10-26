# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

"""
Local file input data fetcher: reads local flight data (e.g. anonymized onboard
records) from {path}/{collection_id}.parquet|.pq|.csv.
See README section "Local data format".
"""

from enum import StrEnum

from qualifly_lib.data_fetching._local_file import fetch_local_file, get_flight_info

# Local files are the data store itself - no per-flight CSV copies are cached
CACHE_DATA = False


# Canonical internal column names; the source names of the stored files are
# configured in config.yaml
# (data_fetching.local_file_in.columns / .sum_columns / .baro_params)
class Config(StrEnum):
    IDENTIFIER = "local_file_in"
    TIME = "time"
    LAT = "lat"
    LON = "lon"
    GEOALT = "geo_alt"
    BAROALT = "baro_alt"
    GROUNDSPEED = "ground_speed"


def fetch_data(collection_id, base_path, options=None, flight_info=None):
    """Backend entry point (see data_fetcher.use_local_or_fetch_data); `flight_info` is ignored."""
    return fetch_local_file(collection_id, options or {}, Config)
