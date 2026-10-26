# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

__version__ = "0.1.0"

import importlib

_LAZY_EXPORTS = {
    "vepu_to_nacp": "qualifly_lib.accuracy_comparison",
    "epu_to_nacp": "qualifly_lib.accuracy_comparison",
    "NACP_TABLE": "qualifly_lib.accuracy_comparison",
    "process_flight": "qualifly_lib.flight_processor",
    "preprocess_data": "qualifly_lib.flight_processor",
    "use_local_or_fetch_data": "qualifly_lib.data_fetching.data_fetcher",
    "get_flight_info": "qualifly_lib.data_fetching.data_fetcher",
    "load_config": "qualifly_lib.data_fetching.data_fetcher",
    "upsert_flight": "qualifly_lib.summary",
    "export_summary": "qualifly_lib.summary",
    "analyze_summary": "qualifly_lib.summary",
    "fetch_opensky_data": "qualifly_lib.data_fetching.opensky",
    "calculate_errors": "qualifly_lib.error_calculation",
    "haversine_distance": "qualifly_lib.error_calculation",
    "use_filtering": "qualifly_lib.filtering",
    "plot_trajectory_map": "qualifly_lib.visualization",
    "plot_time_series_overview": "qualifly_lib.visualization",
    "plot_time_series_overview_interactive": "qualifly_lib.visualization",
    "_source_to_target": "qualifly_lib.visualization",
    "M_TO_FT": "qualifly_lib.utils",
    "log_failed_ids": "qualifly_lib.utils",
    "open_in_browser": "qualifly_lib.utils",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        value = getattr(importlib.import_module(_LAZY_EXPORTS[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
