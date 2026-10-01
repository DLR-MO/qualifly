import os
import sys
from pathlib import Path
import time
import pandas as pd
import matplotlib.pyplot as plt
import urllib3

from qualifly_lib.data_fetching.data_fetcher import use_local_or_fetch_data, load_config
from qualifly_lib.error_calculation import calculate_errors
from qualifly_lib.filtering import use_filtering
from qualifly_lib.accuracy_comparison import validate_nacp
from qualifly_lib.visualization import (plot_trajectory_map, plot_time_series_overview,
                                        plot_time_series_overview_interactive)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_CONFIG_PATH = Path(__file__).parent / "column_config.yaml"


def process_flight(collection_id, source_identifier, target_identifier, base_path="data"):
    """
    Processes a single flight.
    """
    print(f"\n--- Processing Flight: {collection_id} ---")

    # Setup paths and read source and target
    os.makedirs(f"{base_path}/{collection_id}", exist_ok=True)
    config = load_config(source_identifier, target_identifier)

    # Fetch Data
    source_df = use_local_or_fetch_data(base_path, collection_id, source_identifier)
    target_df = use_local_or_fetch_data(base_path, collection_id, target_identifier)
    if source_df is None or target_df is None:
        return

    # Preprocessing
    start = time.perf_counter()
    source_df, target_df, aligned_source_df, aligned_target_df = preprocess_data(
        source_df, target_df, config, base_path, collection_id
    )
    if aligned_source_df is None:
        return

    all_target_dfs = {"original": aligned_target_df}
    # Filtering
    filtered_results = use_filtering(aligned_target_df)
    for variant, df in filtered_results.items():
        all_target_dfs[variant] = df
    end = time.perf_counter()
    print(f"Elapsed time for preprocessing: {end - start:.2f} seconds")

    # Error Calculation
    start = time.perf_counter()
    score, position_errors, best_variant = calculate_errors(aligned_source_df, all_target_dfs, config)
    end = time.perf_counter()
    print(f"Elapsed time for error calculation: {end - start:.2f} seconds")
    all_target_dfs["original"]["position_error"] = position_errors["original"]
    all_target_dfs[best_variant]["position_error"] = position_errors[best_variant]
    scores_df = pd.DataFrame.from_dict(score, orient="index")
    scores_df.index.name = "variant"
    scores_df.to_csv(f"{base_path}/{collection_id}/scores.csv")
    all_target_dfs[best_variant].to_csv(f"{base_path}/{collection_id}/{best_variant}.csv")

    # NACp Validation
    start = time.perf_counter()
    validate_nacp(base_path, collection_id, all_target_dfs, best_variant, config)
    end = time.perf_counter()
    print(f"Elapsed time for NACp validation: {end - start:.2f} seconds")

    # Plotting
    start = time.perf_counter()
    path = f"{base_path}/{collection_id}/time_series.svg"
    fig_map = plot_trajectory_map(
        source_df=source_df,
        target_df=target_df,
        config=config,
        best_variant_name=best_variant,
        best_variant_df=all_target_dfs[best_variant]
    )
    fig_ts = plot_time_series_overview(
        source_df=source_df,
        target_df=target_df,
        config=config,
        filtered_results=filtered_results,
        score=score
    )
    end = time.perf_counter()
    print(f"Elapsed time for visualization: {end - start:.2f} seconds")
    fig_map.show(renderer="browser")
    fig_map.write_html(f"{base_path}/{collection_id}/trajectory_map.html")
    fig_ts.savefig(path, bbox_inches="tight")
    plt.close(fig_ts)

    fig_ts_interactive = plot_time_series_overview_interactive(
        source_df, target_df, config, filtered_results, collection_id, score
    )
    fig_ts_interactive.show(renderer="browser")
    fig_ts_interactive.write_html(f"{base_path}/{collection_id}/time_series.html")


def preprocess_data(source_df, target_df, config, base_path, collection_id):
    """
    Uses methods like Resampling, Unit conversion, Data sorting to preprocess the data.
    """
    source = config["source"]
    target = config["target"]

    # Convert UNIX timestamp to datetime format
    source_df['timestamp'] = pd.to_datetime(source_df[source.TIME], utc=True, unit='s', errors='coerce')
    target_df['timestamp'] = pd.to_datetime(target_df[target.TIME], utc=True)
    # Set index
    source_df = source_df.set_index('timestamp')
    target_df = target_df.set_index('timestamp')
    # Sort by time column
    source_df = source_df.sort_index()
    target_df = target_df.sort_index()

    # Save with new timstamp
    source_df.to_csv(f"{base_path}/{collection_id}/{source.IDENTIFIER}.csv")
    target_df.to_csv(f"{base_path}/{collection_id}/{target.IDENTIFIER}.csv")

    # Resample to 1s intervals
    try:
        source_df = source_df.resample('1s').median(numeric_only=True).ffill()
        resampled = target_df.resample('1s').median(numeric_only=True)
        # Backward fill to fill initial null values
        target_df = resampled.set_axis(
            pd.DatetimeIndex(resampled.index, name='timestamp'), axis=0
        ).ffill().bfill()
    except KeyError:
        sys.exit("An error occurred while processing different parameters. This could be due to differently named "
                 "parameters. Please check your data and the respective flight.")

    # Align both datasets
    source_aligned, target_aligned = source_df.align(target_df, join='inner', axis=0)
    if source_aligned.empty or target_aligned.empty:
        print("No overlapping timestamps between datasets thus cannot compute correlation.")
        print(f"source range:   {source_df.index.min()} to {source_df.index.max()}")
        print(f"opensky range:  {target_df.index.min()} to {target_df.index.max()}")
        return source_df, target_df, None, None

    return source_df, target_df, source_aligned, target_aligned
