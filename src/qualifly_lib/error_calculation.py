import numpy as np
import similaritymeasures

_EARTH_RADIUS = 6371000  # Earth radius in meters
_NMAD_SCALE_FACTOR = 1.4826  # scales MAD to be comparable to a std-dev under a normal-error assumption


def calculate_errors(source_df, all_target_dfs, columns_config, metric_key="pos_dtw"):
    """
    Calculates discrete Fréchet distance (DFD) and dynamic time warping (DTW) distance
    for position, and root-mean-square error (RMSE) and normalized median absolute
    deviation (NMAD) for altitude, between a source trajectory and one or more target
    trajectory variants.

    metric_key: the score entry used to pick the best-performing filtered variant.

    Returns (scores, all_position_errors, best_variant):
        * scores is a dictionary with dict {pos_dfd, pos_dtw, alt_rmse, alt_nmad} per variant
        * all_position_errors are dicts keyed by variant name with errors per timestamp
        * best_variant is the dictionary of the filtered variant with the lowest metric_key value
    """
    scores = {}
    all_position_errors = {}
    for variant, df in all_target_dfs.items():

        if source_df.empty or df.empty:
            print("Something went wrong. No data found.")
            for variant in all_target_dfs:
                scores[variant] = {
                    "pos_dfd": None, "pos_dtw": None,
                    "alt_rmse": None, "alt_nmad": None,
                }
            break

        variant_score, variant_position_errors = _calculate_errors_per_df(source_df, df, columns_config, label=variant)
        if variant_score is None:
            print("No overlapping timestamps between datasets thus cannot compute correlation.")
            print(f"source range:  {source_df.index.min()} to {source_df.index.max()}")
            print(f"target range:  {df.index.min()} to {df.index.max()}")
            break
        scores[variant] = variant_score
        all_position_errors[variant] = variant_position_errors

    best_variant = _best_variant(scores, metric_key)

    return scores, all_position_errors, best_variant


def _best_variant(scores, metric_key):
    """
    Finds the variant with the lowest metric_key value in scores, ignoring variants where that value is None.
    Returns the variant name, or None if no variant qualifies.
    """
    candidates = {
        variant: values[metric_key]
        for variant, values in scores.items()
        if values.get(metric_key) is not None
    }
    if not candidates:
        return None

    return min(candidates, key=candidates.get)


def _calculate_errors_per_df(source_df, target_df, config, label=None):
    """
    Calculates DFD and DTW distance for position, and RMSE and NMAD for altitude.
    Returns a dictionary with four entries: {pos_dfd, pos_dtw, alt_rmse, alt_nmad}
    """
    source = config["source"]
    target = config["target"]

    n_points = len(source_df)
    m_points = len(target_df)

    # Position error via haversine in meters
    position_errors = haversine_distance(
        source_df[source.LAT], source_df[source.LON],
        target_df[target.LAT], target_df[target.LON]
    )

    # Position curves projected to local meters for DFD and DTW
    source_xy = _latlon_to_local_xy(source_df[source.LAT], source_df[source.LON])
    target_xy = _latlon_to_local_xy(target_df[target.LAT], target_df[target.LON])

    # Discrete Fréchet distance on position
    pos_dfd = similaritymeasures.frechet_dist(source_xy, target_xy)

    # Dynamic time warping distance on position, normalized to √(dtw/max(m,n))
    pos_dtw_raw, _ = similaritymeasures.dtw(source_xy, target_xy)
    pos_dtw = np.sqrt(pos_dtw_raw / max(n_points, m_points))

    # Altitude error, pointwise (source/target assumed time-aligned; any non-overlapping
    # timestamps become NaN in the subtraction and are dropped before scoring)
    altitude_errors = (source_df[source.GEOALT] - target_df[target.GEOALT]).dropna()
    alt_rmse = np.sqrt((altitude_errors ** 2).mean())
    alt_nmad = _normalized_mad(altitude_errors)

    if label is not None:
        print(f"Deviation calculated for {label} data")

    return {
        "pos_dfd": pos_dfd, "pos_dtw": pos_dtw,
        "alt_rmse": alt_rmse, "alt_nmad": alt_nmad,
    }, position_errors


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Returns distance in meters between two lat/lon points.
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return 2 * _EARTH_RADIUS * np.arcsin(np.sqrt(a))


def _latlon_to_local_xy(lat, lon):
    """
    Projects lat/lon (degrees) onto a local equirectangular plane in meters,
    referenced to the mean latitude of the input.
    Avoids mixing degrees of lat/lon (which have different real-world
    distances) directly into a Euclidean Fréchet/DTW distance.
    Returns an (N, 2) array suitable for similaritymeasures.frechet_dist / dtw.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat0 = np.radians(lat.mean())
    x = _EARTH_RADIUS * np.radians(lon) * np.cos(lat0)
    y = _EARTH_RADIUS * np.radians(lat)
    return np.column_stack([x, y])


def _normalized_mad(errors):
    """
    Normalized Median Absolute Deviation (NMAD) of an error array:
        NMAD = 1.4826 * median(|e_i - median(e)|)
    """
    errors = np.asarray(errors, dtype=float)
    median_error = np.median(errors)
    return _NMAD_SCALE_FACTOR * np.median(np.abs(errors - median_error))
