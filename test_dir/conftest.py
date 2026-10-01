from __future__ import annotations

import importlib
import sys
import types
from typing import Any


class _Placeholder:
    """Stands in for a class from an optional/unavailable dependency."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


def _ensure_module(dotted_name: str, **attributes: Any) -> types.ModuleType:
    """
    Return the real module named `dotted_name` if it can be imported as-is;
    otherwise register (and return) a bare stub module with the given
    attributes, creating any missing parent packages the same way.
    """
    try:
        module = importlib.import_module(dotted_name)
    except ImportError:
        parent_name, _, child_name = dotted_name.rpartition(".")
        module = types.ModuleType(dotted_name)
        sys.modules[dotted_name] = module
        if parent_name:
            parent = _ensure_module(parent_name)
            setattr(parent, child_name, module)

    for attr_name, value in attributes.items():
        if not hasattr(module, attr_name):
            setattr(module, attr_name, value)
    return module


# --- cartes (geospatial projections, used for Kalman filtering) -------------
_ensure_module("cartes.crs", EuroPP=_Placeholder)

# --- traffic (trajectory filtering library) ---------------------------------
_ensure_module("traffic.core", Flight=_Placeholder)
_ensure_module("traffic.algorithms.filters", FilterMedian=_Placeholder, FilterMean=_Placeholder)
_ensure_module("traffic.algorithms.filters.aggressive", FilterDerivative=_Placeholder)
_ensure_module("traffic.algorithms.filters.kalman", KalmanFilter6D=_Placeholder, KalmanSmoother6D=_Placeholder)

# --- pyopensky (OpenSky Network Trino client) -------------------------------
_ensure_module("pyopensky.trino", Trino=_Placeholder)
_ensure_module("pyopensky.schema", OperationalStatusData4=_Placeholder)
