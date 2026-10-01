from __future__ import annotations

import os
from collections import namedtuple
from typing import Iterable

AttributeConversion = namedtuple(
    "AttributeConversion", ["source", "target", "conversion_factor"]
)

M_TO_FT = 3.28084
MPS_TO_KT = 1.94384
MPS_TO_FTMIN = 196.850394


def apply(df, conversions: Iterable[AttributeConversion], inverse=False):
    """
    Generic worker function for applying each conversion in-place on a copy of `df`.
    If `inverse=True`, the conversion is reversed (target -> source, dividing by the factor instead of multiplying).
    """
    df = df.copy()
    for conv in conversions:
        if conv.source not in df.columns:
            continue
        if inverse:
            df[conv.source] = df[conv.target] / conv.conversion_factor
        else:
            df[conv.target] = df[conv.source] * conv.conversion_factor
    return df

def log_failed_ids(filepath, collection_id):
    """
    Log collection_id in a file saved as filepath if it is not already there.
    """
    existing_ids = set()
    if os.path.exists(filepath):
        with open(filepath, "r") as f_read:
            existing_ids = {line.strip() for line in f_read}
    if str(collection_id) not in existing_ids:
        with open(filepath, "a") as f_write:
            f_write.write(f"{collection_id}\n")
    return None
