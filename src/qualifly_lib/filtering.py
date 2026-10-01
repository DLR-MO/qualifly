import numpy as np
import pandas as pd
from cartes.crs import EuroPP
from traffic.core import Flight
from traffic.algorithms.filters import FilterMedian, FilterMean
from traffic.algorithms.filters.aggressive import FilterDerivative
from traffic.algorithms.filters.kalman import KalmanFilter6D, KalmanSmoother6D
from qualifly_lib.data_fetching.opensky import to_opensky_units, to_traffic_units


def prepare_for_filtering(df):
    """
    Renaming + Unit conversion for traffic's Filters.
    """
    df = df.reset_index()
    df = to_traffic_units(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return Flight(df)


def restore_from_filtering(flights):
    """
    Conversion to old naming convention and units of Flight objects for visualization later.
    """
    dfs = {}
    for variant, flight in flights.items():
        df = to_opensky_units(flight.data)
        df = df.set_index('timestamp')
        dfs[variant] = df
    return dfs


def use_filtering(df):
    """
    Applies several filtering techniques to one dataframe.
    Returns a dict with filtered dataframes.
    """
    if len(df) == 0:
        raise ValueError("Got an empty DataFrame — cannot filter anything.")

    results = {}
    flight = prepare_for_filtering(df)

    # AboveSigmaMedian
    results["default"] = flight.filter()

    # Rolling Median
    results["median"] = flight.filter(FilterMedian(latitude=5, longitude=5))

    # Rolling Mean
    results["mean"] = flight.filter(FilterMean(latitude=5, longitude=5))

    # Derivative-based
    scale = 1 / np.cos(np.radians(flight.data.latitude.mean()))
    results["derivative"] = flight.filter(FilterDerivative(
        latitude=dict(first=0.0055, second=0.0060, kernel=5),
        longitude=dict(first=0.0055 * scale, second=0.0060 * scale, kernel=5)
    ))

    # Kalman Filter (6D: lat, lon, alt, track, groundspeed, vertical rate)
    # Needs first a projection in x, y
    projection = EuroPP()
    processed_flight = flight.compute_xy(projection)
    results["kalman"] = processed_flight.filter(KalmanFilter6D()).compute_latlon_from_xy(projection)

    # Kalman Smoother
    results["kalman_smoother"] = processed_flight.filter(KalmanSmoother6D()).compute_latlon_from_xy(projection)

    return restore_from_filtering(results)
