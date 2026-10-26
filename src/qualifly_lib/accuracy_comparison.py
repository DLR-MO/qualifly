# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

import pandas as pd

from qualifly_lib.utils import TIME_TOLERANCES, atomic_path, log_failed_ids

# Share of reports (percent) within the claimed bound needed to fulfil a NACp claim
NACP_REQUIRED_SHARE = 95.0

# Per-report errors held against the claim, with their label, strictest first
ERROR_KINDS = {
    "position_error": "time-aligned",
    "position_error_synced": "offset removed",
    **{f"position_error_{name}": f"{seconds} s tolerance" for name, seconds in TIME_TOLERANCES.items()},
    "track_distance": "path distance",
}


def validate_nacp(base_path, collection_id, all_target_dfs, config):
    """
    Writes the per-report claims and achieved levels of every variant to
    {base_path}/{collection_id}/epu_validation.csv: NACp for the ERROR_KINDS
    errors, VEPU/GVA for the altitude error and NACv for the ground speed error.
    Returns the written frame, or None without "original".
    """
    print("Comparing NACp values...")
    target = config["target"]

    if "original" not in all_target_dfs:
        print(f"No 'original' data found in {target.IDENTIFIER}.")
        return None

    original = all_target_dfs["original"]
    output = pd.DataFrame(index=original.index)
    nac = getattr(target, "POSITIONNAC", None)
    if nac in original.columns:
        output["claimed_nacp"] = original[nac]
    else:
        # Log flight id without NACp value
        print(f"No usable NACp data - skipping comparison with the claim. See {base_path}/failed_nacp.txt for "
              f"all flights with failed nacp.")
        filepath = f"{base_path}/failed_nacp.txt"
        log_failed_ids(filepath, collection_id)
    gva = getattr(target, "GVA", None)
    if gva in original.columns:
        # OSN stores the bound in meters (45, 150), other sources the GVA itself (0-3)
        claimed = pd.to_numeric(original[gva], errors="coerce")
        output["claimed_gva"] = claimed.where(claimed <= 3, claimed.map({bound: level for level, bound in GVA_TABLE}))
    nacv = getattr(target, "NACV", None)
    if nacv in original.columns:
        output["claimed_nacv"] = pd.to_numeric(original[nacv], errors="coerce")

    for variant, df in all_target_dfs.items():
        for kind in ERROR_KINDS:
            if kind not in df.columns:
                continue
            output[f"{variant}_{kind}"] = df[kind]
            output[f"{variant}_{kind}_nacp"] = df[kind].map(epu_to_nacp, na_action="ignore")
        if "altitude_error" in df.columns:
            errors = df["altitude_error"].abs()
            output[f"{variant}_altitude_error"] = df["altitude_error"]
            output[f"{variant}_altitude_error_nacp"] = errors.map(lambda error: vepu_to_nacp(error) or 0,
                                                                  na_action="ignore")
            output[f"{variant}_altitude_error_gva"] = errors.map(error_to_gva, na_action="ignore")
        if "baro_error" in df.columns:
            # no accuracy category bounds the barometric altitude
            output[f"{variant}_baro_error"] = df["baro_error"]
        if "groundspeed_error" in df.columns:
            speeds = df["groundspeed_error"].abs()
            output[f"{variant}_groundspeed_error"] = df["groundspeed_error"]
            output[f"{variant}_groundspeed_error_nacv"] = speeds.map(lambda error: error_to_nacv(error),
                                                                     na_action="ignore")

    output = output.rename_axis("timestamp").reset_index()

    filepath = f"{base_path}/{collection_id}/epu_validation.csv"
    with atomic_path(filepath) as tmp:
        output.to_csv(tmp, index=False)
    return output


def nacp_fulfilment(df, prefix):
    """
    Share of reports (percent) per error kind whose achieved NACp reaches the
    claimed one, from an epu_validation.csv frame; None without claim or error.
    """
    return {kind: _reached_share(df, "claimed_nacp", f"{prefix}_{kind}_nacp") for kind in ERROR_KINDS}


def vertical_fulfilment(df, prefix):
    """
    Share of reports (percent) whose altitude error reaches the claimed VEPU
    ("vepu", NACp 9-11 only) or GVA ("gva", 1-2 only), from an
    epu_validation.csv frame; None without claim or error.
    """
    return {
        "vepu": _reached_share(df, "claimed_nacp", f"{prefix}_altitude_error_nacp", claims=(9, 11)),
        "gva": _reached_share(df, "claimed_gva", f"{prefix}_altitude_error_gva", claims=(1, 2)),
    }


def velocity_fulfilment(df, prefix):
    """
    Share of reports (percent) whose ground speed error reaches the claimed
    NACv ("nacv", 1-4 only; NACv 0 claims nothing), from an
    epu_validation.csv frame; None without claim or error.
    """
    return {
        "nacv": _reached_share(df, "claimed_nacv", f"{prefix}_groundspeed_error_nacv", claims=(1, 4)),
    }


def _reached_share(df, claim_column, achieved_column, claims=None):
    """
    Share of reports (percent) whose achieved level reaches the claimed one,
    counting only the claims within `claims` (lowest, highest) if given.
    """
    if claim_column not in df.columns or achieved_column not in df.columns:
        return None
    claimed = pd.to_numeric(df[claim_column], errors="coerce")
    achieved = pd.to_numeric(df[achieved_column], errors="coerce")
    valid = achieved.notna() & (claimed.notna() if claims is None else claimed.between(*claims))
    return float((achieved[valid] >= claimed[valid]).mean() * 100) if valid.any() else None


# NACp lookup table: (nacp, epu_threshold_m, vepu_threshold_m)
# EPU = horizontal accuracy bound, VEPU = vertical accuracy bound.
# VEPU is only defined for NACp 9-11.
NACP_TABLE = [
    (11, 3, 4),
    (10, 10, 15),
    (9, 30, 45),
    (8, 93, None),
    (7, 185, None),
    (6, 556, None),
    (5, 926, None),
    (4, 1852, None),
    (3, 3704, None),
    (2, 7408, None),
    (1, 18520, None),
]


def epu_to_nacp(error):
    """
    Maps horizontal position error (meters) to a NACp level based on the EPU (horizontal accuracy) thresholds.
    """
    for nacp, epu_threshold, _ in NACP_TABLE:
        if error < epu_threshold:
            return nacp
    return 0


def vepu_to_nacp(error):
    """
    Maps a vertical (altitude) error (meters) to a NACp level based on the VEPU (vertical accuracy) thresholds.
    """
    for nacp, _, vepu_threshold in NACP_TABLE:
        if vepu_threshold is not None and error < vepu_threshold:
            return nacp
    return None


# NACv lookup table (DO-260B): (nacv, bound of the 95 % velocity error in m/s);
# NACv 0 means unknown or >= 10 m/s
NACV_TABLE = [
    (4, 0.3),
    (3, 1.0),
    (2, 3.0),
    (1, 10.0),
]


def error_to_nacv(error):
    """
    Maps a ground speed error (m/s, absolute) to the NACv it achieves.
    """
    for nacv, bound in NACV_TABLE:
        if error < bound:
            return nacv
    return 0


# GVA lookup table (DO-260B): (gva, bound of the 95 % vertical error in meters);
# GVA 0 means unknown or > 150 m, 3 is reserved
GVA_TABLE = [
    (2, 45),
    (1, 150),
]


def error_to_gva(error):
    """
    Maps an altitude error (meters, absolute) to the GVA it achieves.
    """
    for gva, bound in GVA_TABLE:
        if error <= bound:
            return gva
    return 0
