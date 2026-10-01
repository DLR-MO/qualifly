from __future__ import annotations

import pandas as pd
from typing import List
from enum import StrEnum
from pyopensky.trino import Trino
from pyopensky.schema import OperationalStatusData4

from qualifly_lib.utils import M_TO_FT, MPS_TO_KT, MPS_TO_FTMIN, AttributeConversion, apply


# Predefined parameters for comparison
class Config(StrEnum):
    IDENTIFIER = "opensky"
    TIME = "time"
    LAT = "lat"
    LON = "lon"
    GEOALT = "geoaltitude"
    POSITIONNAC = "positionnac"


OPENSKY_TO_TRAFFIC: List[AttributeConversion] = [
    AttributeConversion(Config.LAT, "latitude", 1.0),
    AttributeConversion(Config.LON, "longitude", 1.0),
    AttributeConversion(Config.GEOALT, "altitude", M_TO_FT),          # m -> ft
    AttributeConversion("velocity", "groundspeed", MPS_TO_KT),         # m/s -> kt
    AttributeConversion("heading", "track", 1.0),
    AttributeConversion("vertrate", "vertical_rate", MPS_TO_FTMIN),    # m/s -> ft/min
]


def to_traffic_units(df):
    """OpenSky (SI) -> traffic (ft, kt, ft/min)."""
    return apply(df, OPENSKY_TO_TRAFFIC)


def to_opensky_units(df):
    """traffic -> OpenSky (SI)."""
    return apply(df, OPENSKY_TO_TRAFFIC, inverse=True)


def fetch_opensky_data(icao_hex, start, stop, collection_id, base_path):
    """
    Retrieves opensky parameters from state_vector table and raw data for NACp values.
    Saves the data in a CSV file within {base_path}/{collection_id}/opensky.csv
    """
    # Initialize Clients
    client = Trino()

    # Searches for flight with start < lastseen < stop
    flight_list = client.flightlist(
        start=start,
        stop=stop,
        icao24=icao_hex)

    # If opensky data has more data throughout the end of the flight than the flight list
    if flight_list is None:
        flight_list = client.flightlist(
            start=start,
            icao24=icao_hex)
        if flight_list is None:
            return None
        flight_list = flight_list.sort_values("firstseen")

    # Get data
    opensky_df = client.history(
        start=flight_list['firstseen'].iloc[0],
        stop=flight_list['lastseen'].iloc[0],
        icao24=icao_hex,
    )

    opensky_df = opensky_df.sort_values(Config.TIME).reset_index(drop=True)

    # Get raw data with nac score
    raw_df = client.rawdata(
        start=flight_list.iloc[0].firstseen,
        stop=flight_list.iloc[0].lastseen,
        icao24=icao_hex,
        Table=OperationalStatusData4,
        extra_columns=(OperationalStatusData4.positionnac,
                       OperationalStatusData4.geometricverticalaccuracy),
    )

    # Merge nac with opensky_df
    if raw_df is not None and not raw_df.empty:
        # Convert timestamps into suitable format
        opensky_df[Config.TIME] = pd.to_datetime(opensky_df[Config.TIME]).dt.tz_convert('UTC')
        raw_df[Config.TIME] = (
            pd.to_datetime(raw_df['mintime'], unit='s', utc=True)
            .dt.floor('s')
            .astype(opensky_df[Config.TIME].dtype)
        )
        raw_df = raw_df.sort_values(Config.TIME)

        opensky_df = pd.merge_asof(
            opensky_df,
            raw_df[[Config.TIME, Config.POSITIONNAC, 'geometricverticalaccuracy']],
            on=Config.TIME,
            direction='backward',
            tolerance=pd.Timedelta('15s'),
        )
        # Fill values in the beginning
        opensky_df[['positionnac']] = opensky_df[['positionnac']].bfill()

    opensky_df.to_csv(f"{base_path}/{collection_id}/opensky.csv", index=False)

    return opensky_df
