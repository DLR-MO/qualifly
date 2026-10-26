<!--
SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)

SPDX-License-Identifier: MIT
-->

# Scores

What each score describes and which data it is computed on. All scores compare a
**target** (data under test, e.g. OpenSky reports) with a **source** (reference,
e.g. onboard records), for the unfiltered target (`original`) and every filter
variant. Distances are in meters, offsets in seconds, speed differences in m/s.

## Processing steps

1. **Load.** Fetch or read both sides, rename the columns to the canonical names
   (`time`, `lat`, `lon`, `geo_alt`, `baro_alt`, …), convert raw barometric
   parameters from ft to m, and merge rows with the same timestamp by median.
2. **Time.** Convert times to UTC. OpenSky rows are timed by their position fix
   (`lastposupdate`) when present, not by the state-vector snapshot.
3. **Altitude datum.** `geo_alt_is_msl: true` converts a backend's geometric
   altitude from MSL to the WGS-84 ellipsoid (EGM96).
4. **Real source samples.** A GNSS fix stored in rows of a higher rate repeats
   its value until the next fix (e.g. a 1 Hz position in 20 Hz rows). These
   repeats are dropped for `lat`, `lon` and `geo_alt`, so the reference is the
   line between the real fixes instead of a staircase lagging by up to one fix
   interval. Repeats are detected on `lat`/`lon` only, so a constant altitude in
   level flight is kept. Other columns (e.g. a 20 Hz `baro_alt`) keep all samples.
5. **Per-report time base.** Every target position report is one evaluated row.
   The source is interpolated to the report time; reports outside the source or
   in a source gap wider than `max_gap_s` are dropped (a position held longer
   than that, e.g. a parked aircraft, leaves such a gap too). `geo_alt`,
   `baro_alt` and `ground_speed` are interpolated column by column with their own
   gaps, so a report can have a position error but no altitude error, and the
   `alt_*`/`baro_*`/`gs_*` scores may rest on fewer reports than the `pos_*` ones.
6. **Filters.** `mean` is a centred 5 s rolling mean; `median` a centred rolling
   median over 5 reports, applied to the reports themselves (on the 1 s grid a
   spike next to a gap becomes a ramp over many seconds). It removes up to two
   outlying reports in a row and leaves a straight flight unchanged.
   `kalman_smoother` is traffic's `KalmanSmoother6D`, fusing positions with the
   reported velocity, heading and vertical rate (requires the OSN columns
   `velocity`, `heading`, `vertrate`). Known flaws: it runs about 0.5 s ahead
   (shown by `time_offset_s`, removed by `pos_p95_synced`), takes its position
   noise from a 17 s trailing mean (~1 km) and therefore mostly dead-reckons from
   the velocities, and fails on a constant velocity column (e.g. `vertrate` in
   level flight), in which case the variant is skipped.
   `mean` and `kalman_smoother` need a regular 1 s series and run on the reports
   interpolated to full seconds (also across gaps; edge values are held).
   `median` runs on the reports themselves. The filtered series is interpolated
   back to the report times; reports outside the grid take the report plus the
   interpolated correction. Every variant is therefore scored at the same
   instants as `original`, and a filter that changes nothing scores exactly like
   `original`.
   All variants smooth `filtering.filtered_columns`: `lat`, `lon`, `geo_alt` and,
   where present, `baro_alt` and the configured ground-speed column. The
   `kalman_smoother` derives its ground speed from the smoothed track and runs a
   second time for `baro_alt`.

## Scores

| Score | Describes |
|-------|-----------|
| `pos_rmse` | Root mean square of the horizontal distance between report and source at the report time. Time offsets and outliers count. |
| `pos_p95` | 95th percentile of those distances: the value a NACp bound (95 %) refers to. |
| `time_offset_s` | Constant time offset of the reports: the shift within ±1 s (5 ms steps) minimizing the median distance to the source, interpolated anew at each shifted report time. Positive = reports stamped late. Every variant gets its own offset, so a lagging filter shows here. `--analyze` shows the median of its amount (opposite signs cancel). |
| `pos_rmse_synced`, `pos_p95_synced` | `pos_rmse` and `pos_p95` after removing each variant's own `time_offset_s`, so a filter cannot win by a lag. The offset minimizes the median distance, not the RMSE or p95, so synced scores can be larger. |
| `pos_p95_tol100ms`, `pos_p95_tol300ms`, `pos_p95_tol600ms` | `pos_p95` with a time error of up to 0.1/0.3/0.6 s forgiven: p95 of the distance to the path section flown in that window (see [Time tolerance](#time-tolerance-path-sections)). No offset removed. |
| `pos_track_mean`, `pos_track_p95`, `pos_track_max` | Mean, p95 and maximum distance from each report to the path section flown within ±`track_window_s`, no offset removed. How far the report is off the path, whatever the time. |
| `alt_bias` | Median geometric altitude difference (source − report) at the report time. Positive = reports too low. Robust to outliers. |
| `alt_rmse` | Root mean square of the same difference, including offset and outliers. No time offset removed: at 10 m/s vertical speed 0.1 s is 1 m, below the 25 ft grid of OSN altitudes. |
| `alt_nmad` | Spread of that difference (1.4826 · median absolute deviation). Offset and outliers do not count; the 25 ft OSN grid alone gives about 2.8 m. |
| `alt_p95` | 95th percentile of the absolute geometric altitude difference, offset and outliers included: the value VEPU and GVA (95 %) refer to. |
| `baro_bias` | Median barometric altitude difference (source − report), when both sides carry `baro_alt`. Mostly the pressure reference difference, hardly a report quality. Smoothing makes it differ slightly by variant. |
| `baro_nmad` | Spread of that difference. No RMSE, since the bias depends on the pressure references. |
| `gs_bias` | Median ground speed difference (source − report): the constant speed offset. Positive = reports too slow. |
| `gs_rmse` | Root mean square of the same difference, including offset and outliers. No time offset removed. |
| `gs_nmad` | Spread of that difference (1.4826 · median absolute deviation). |
| `gs_p95` | 95th percentile of the absolute ground speed difference: the value a NACv bound (95 %) refers to. |

The `baro_*` and `gs_*` scores differ by variant because the filters smooth
`baro_alt` and the ground speed as well; for `kalman_smoother` the ground speed
comes from the smoothed track, not from a filtered reported speed.

## Time offset per score

`time_offset_s` is estimated per variant from its positions. Only the `*_synced`
scores remove it; all others take the source at the report time, some with a
window around it.

| Score | Source taken at | Time offset |
|-------|-----------------|-------------|
| `pos_rmse`, `pos_p95` | report time | not removed |
| `pos_rmse_synced`, `pos_p95_synced` | report time − `time_offset_s` | the variant's own, removed |
| `pos_p95_tol*` | path section at report time ± tolerance | forgiven up to ± tolerance |
| `pos_track_*` | path section at report time ± `track_window_s` | forgiven up to ±2 s |
| `alt_*`, `baro_*`, `gs_*` | report time | not removed |

A constant altitude bias (without a time offset) is separate: `alt_bias` and
`baro_bias` measure it, `alt_nmad` and `baro_nmad` ignore it, `alt_rmse` and
`alt_p95` include it. NACp validation uses the per-report position errors; VEPU
and GVA use the per-report altitude error behind `alt_rmse`, not the RMSE itself.

Filters and the offset: `original`, `mean` and `median` have no time shift of
their own. `kalman_smoother` runs ≈0.5 s ahead, so its `time_offset_s` is about
0.5 s lower; the `*_synced` scores remove that lead, `pos_p95_tol600ms` and
`pos_track_*` forgive most of it, all other scores keep it (with late reports it
cancels part of their offset and flatters the filter).

## Time tolerance: path sections

The flown path is the line through the real source fixes (step 4). Every point of
it has a time, interpolated between two fixes. For a report and a window
`[a, b]`, the **path section** is the part of the line flown between `a` and `b`
(the segments inside the window, cut at its edges). The score of the report is
its shortest distance to that section.

| Score | Window | Time error forgiven |
|-------|--------|---------------------|
| `pos_p95` | report time | none |
| `pos_p95_tol100ms` | report time ± 0.1 s | up to 0.1 s |
| `pos_p95_tol300ms` | report time ± 0.3 s | up to 0.3 s |
| `pos_p95_tol600ms` | report time ± 0.6 s | up to 0.6 s |
| `pos_track_*` | report time ± `track_window_s` | up to 2 s |
| `pos_p95_synced` | report time − `time_offset_s` | the constant offset |

The windows from `pos_p95` to `pos_track_*` are nested, so those scores can only
fall from one to the next. `pos_p95_synced` is different: it forgives one
constant shift of up to 1 s, not a scatter.

A tolerance forgives time error *along* the path (up to tolerance × speed: 10 m
at 100 m/s for 0.1 s), nothing *across* it, and it cannot tell a wrong timestamp
from an along-track position error. The steps: 0.1 s covers a small scatter,
e.g. one row of a 10 Hz recording; 0.6 s is the uncompensated ADS-B position
latency allowed by 14 CFR 91.227. This is no compliance check: the latency is
one-sided, the window is symmetric, and OpenSky stamps the reception.

Reading the steps: a drop already at `tol100ms` means small timestamp scatter;
a drop only at `tol600ms` or `pos_track_p95` means large along-track errors; no
drop means the error lies across the path.

## NACp validation

Each report is held against its claimed NACp with six errors: time-aligned,
offset removed, with 0.1/0.3/0.6 s tolerance, and distance to the path. The
tolerance/path errors are nested; the offset-removed error is a separate
diagnostic (the offset minimizes the median, not the p95) and can be larger than
the time-aligned one. The share of reports meeting the claim is stored for every
variant. A flight fulfils a claim when at least 95 % of its reports do; the
analysis output reports how many flights carry a claim and how many reach that
share. Reports without a claim or an error are not counted.

`data/<flight>/epu_validation.csv` holds per report the claim (`claimed_nacp`)
and, prefixed with the variant name (`original_`, `mean_`, …), each error and
the NACp it achieves (`…_nacp`).

## Altitude validation

The absolute per-report altitude error is held against two vertical claims:

* **VEPU** of the claimed NACp (DO-260A): < 4 m for NACp 11, < 15 m for 10,
  < 45 m for 9. Claims below NACp 9 bound no altitude and are not counted.
* **GVA** from the operational status message (DO-260B, ADS-B version 2):
  ≤ 45 m for GVA 2, ≤ 150 m for GVA 1. GVA 0 (unknown or worse) claims nothing.
  Only checked when the target has a `gva` column (the `opensky` backend fetches
  it). OSN stores the bound in meters (45, 150, −1 unknown); it is read as the
  GVA it stands for.

A claim is fulfilled when at least 95 % of its reports reach it. Not every flight
carries a usable VEPU or GVA claim.

The OSN geometric altitude lies on a 25 ft (7.62 m) grid, as do the barometric
altitude and their difference. This
quantization alone spreads the error over ±7.6 m with a 95 % bound of about
5.9 m, so VEPU < 4 m (NACp 11) cannot be confirmed with OSN data, whatever the
aircraft's accuracy. GVA and VEPU from NACp 10 on are not affected.

## Velocity validation (NACv)

The absolute per-report ground speed error is held against the claimed NACv
(DO-260B): < 0.3 m/s for NACv 4, < 1 m/s for 3, < 3 m/s for 2 and < 10 m/s for
1. NACv 0 claims nothing. Only checked when the target has a `nacv` column (the
`opensky` backend fetches it from the operational status message). A claim is
fulfilled when at least 95 % of its reports reach it; NACv claims are rare in
OSN, so the result can rest on very few flights. The checked error is the ground
speed difference alone, one component of the velocity error the category bounds.

## Reading them together

* `pos_p95` large, `pos_p95_synced` small: a constant time offset
  (`time_offset_s`), not a position error.
* `pos_p95_synced` still large, `pos_track_*` small: reports lie on the path but
  ahead of or behind the aircraft, varying per report. Timestamp jitter and
  along-track position error cannot be told apart; the smallest tolerance close
  to `pos_track_p95` bounds the time error.
* `alt_rmse` large, `alt_nmad` small: a constant altitude offset or a few corrupt
  altitudes. `alt_bias` tells them apart (near `alt_rmse` for an offset, typically
  the vertical datum from step 3; near zero for corrupt altitudes).
* Altitude errors growing in climbs and descents: a delay of the altitude (no
  time offset is removed from it).
* Isolated outliers raise `pos_rmse` and `pos_track_max`, but hardly the p95s.
* Gaps in the reports are not scored; `summary.pq` records them instead:
  `overlap_s`, `coverage`, `max_report_gap_s`/`p95_report_gap_s` and `n_dropped`.

Every delivered report stays in the evaluation: there is no plausibility filter,
and outliers stay in the denominator of the 95 % claim shares. `pos_rmse`,
`alt_rmse` and `pos_track_max` are deliberately sensitive to them; `pos_p95`,
`alt_p95`, `alt_nmad`, the tolerance scores and the variants are the robust view.

## Settings and storage

`max_gap_s` and `track_window_s` (default 2 s each) and `geo_alt_is_msl` (per
backend) in `config.yaml` change the scores. The time tolerances
(`TIME_TOLERANCES` in `utils.py`) and the offset search range/step (±1 s, 5 ms,
in `error_calculation.py`) are fixed. Results are cached and the cache does not
track these settings: after a change, delete `data/<flight>/scores.csv` for the
flights to recalculate, or remove everything stored with `--clean`.

The scores of every variant are in `data/<flight>/scores.csv`. `data/scores.pq`
holds them with the claim fulfilment per flight and variant, `data/summary.pq`
the coverage per flight. No variant is picked as best: each score answers a
different question, and `--analyze` compares the variants over all flights.
