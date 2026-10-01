import pandas as pd

from qualifly_lib.utils import log_failed_ids


def validate_nacp(base_path, collection_id, all_target_dfs, best_variant, config):
    """
    Validates the NACp of "original" data
    against best-performing filtered variant calculated with the actual position errors.

    PARAMS:
        * all_target_dfs: dicts keyed by variant name
        * best_variant: the variant name which scored the best
        * config: source and target config

    Returns a dict with up to two entries, "original" and best_variant, each
    holding claimed_epu, position_errors and achieved_epu.

    Returns None if
        * best_variant is None (e.g. no filtered variant scored successfully) or
        * "original" isn't in all_target_dfs or
        * this flight has no usable NACp data
    """
    print("Comparing NACp values...")
    target_fraction = 0.95
    target = config["target"]

    if "original" not in all_target_dfs:
        print(f"No 'original' data found in {target.IDENTIFIER}.")
        return None

    if best_variant is None:
        print("No best-performing filtered variant available - nothing to compare against original.")
        return None

    result = {}
    for variant in ("original", best_variant):
        position_errors = all_target_dfs[variant]["position_error"]
        # Map EPU to calculated position errors
        achieved_epu = position_errors.apply(epu_to_nacp)
        if target.POSITIONNAC in all_target_dfs["original"].columns:
            result[variant] = {
                "claimed_epu": all_target_dfs[variant][target.POSITIONNAC],
                "position_errors": position_errors,
                "achieved_epu": achieved_epu,
            }
        else:
            result[variant] = {
                "position_errors": position_errors,
                "achieved_epu": achieved_epu,
            }

    _save_results_csv(base_path, collection_id, target, all_target_dfs, best_variant, result)

    return result


def _save_results_csv(base_path, collection_id, target, all_target_dfs, best_variant, result):
    original = result["original"]
    best = result[best_variant]

    output = pd.DataFrame(index=all_target_dfs["original"].index)
    if target.POSITIONNAC in all_target_dfs["original"].columns:
        output["claimed_epu"] = original["claimed_epu"]
    else:
        # Log flight id without NACp value
        print(f"No usable NACp data - skipping comparison with claimed_epu. See {base_path}/failed_nacp.txt for "
              f"all flights with failed nacp.")
        filepath = f"{base_path}/failed_nacp.txt"
        log_failed_ids(filepath, collection_id)
    output["original_position_error"] = original["position_errors"]
    output["original_achieved_epu"] = original["achieved_epu"]
    output["best_position_error"] = best["position_errors"]
    output["best_achieved_epu"] = best["achieved_epu"]
    output = output.rename_axis("timestamp").reset_index()

    filepath = f"{base_path}/{collection_id}/epu_validation.csv"
    output.to_csv(filepath, index=False)


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
