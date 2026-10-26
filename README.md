<!--
SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)

SPDX-License-Identifier: MIT
-->

# QUALIFLY

This data-engineering and analysis tool bridges crowdsourced flight data from the
[OpenSky Network](https://opensky-network.org/) and high-fidelity onboard
aircraft records of the DLR (German Aerospace Center) research fleet.

It contains the code for the thesis "**Qualifying crowdsourced flight data based
on inflight records to improve MRO-targeted historical data sets**" (TODO: link
to DLR-thesis-release).

### Key features

* Runs one flight or several flights at once, in parallel
* Searches and downloads the matching ADS-B/Mode-S data from OpenSky
* Applies filters and compares how they change data quality
* Calculates errors and assigns NACp scores to original and filtered data
* Compares filtering techniques by their impact on mismatch and errors
* Shows diagrams of comparable attributes and a trajectory comparison
* Rolls up per-flight metrics into `summary.pq`/`scores.pq`, with CSV export
  and an analysis view

## Requirements

Python 3.12 to 3.14 (`.python-version` pins what uv picks).

Data backends that need credentials must be configured before a run:

* **OpenSky**: set up your account as described in the
  [OpenSky API documentation](https://github.com/openskynetwork/opensky-api#authentication).
* **Local data**: `local_file_in` and `local_public_ref` need only a directory
  of flight files (see [Local data format](#local-data-format)).

## Installation

`uv.lock` pins the exact versions; uv creates the virtual environment:

`uv sync`

Select `.venv` as the Python interpreter in your IDE.

## Configuration

`config.yaml` selects the data backends and their options. The values below run
the pipeline fully on local flight data:

```yaml
source: local_file_in       # ground truth data
target: local_public_ref    # data compared against the source

max_gap_s: 2.0              # optional, largest source gap still interpolated across (s)
track_window_s: 2.0         # optional, time window of pos_track_* (s, > 0)

data_fetching:              # optional, per-backend fetch options
  local_file_in:
    path: local_flight_data/in   # one local file per flight
  local_public_ref:
    path: local_flight_data/reference
```

Available backends:

| Identifier         | Data                                               | Requires |
|--------------------|----------------------------------------------------|----------|
| `opensky`          | OpenSky Network data (target)                      | `pyopensky` + credentials, flight identity from the source backend |
| `local_file_in`    | local input data (e.g. anonymized onboard records), source side | `path` only |
| `local_public_ref` | local stored OSN data (OpenSky format), target side | `path` only |

`path` is the directory with one data file per flight; relative paths are
resolved against the directory of `config.yaml`. The local fetchers read
straight from it (no per-flight CSV copies), so new or anonymized files are
picked up on the next run. With both local backends no credentials or remote
connection are needed.

Per-backend options:

* `filter` is an optional JSON object of exact key/value matches: for `opensky`
  it filters flightlist columns, for the local fetchers the rows of the data
  file. A filtered-out flight returns no data and leaves no not-found marker, so
  a changed filter takes effect on the next run.
* `columns` names the source parameter per canonical attribute (`time`, `lat`,
  `lon`, `geo_alt`, `baro_alt`, `ground_speed`, `position_nac`, `gva`, `nacv`).
  Omitted names fall back to the canonical name or the generic spellings in
  [Local data format](#local-data-format). A list of names gives alternatives
  (e.g. a parameter renamed in a later data period); the first present is used.
* `sum_columns` maps a canonical column to a source parameter that is **added**
  onto it on load (no unit conversion), e.g. the fine part of a slowly refreshed
  sensor word. Every component contributes its last known value, so a slowly
  refreshed latitude summed with its fine part gets the update rate of the fine
  part.
  Alternatives work as in `columns`; several parameters need one list of names
  per parameter. The summed sources and unused alternatives are dropped, so a
  second mapping pass over a cached file cannot add them twice.
* `baro_params` lists the raw onboard barometric altitude parameters (ft); they
  are converted to meters and merged into `baro_alt` as their row-wise mean.
* `ground_speed_is_kt` (`true`/`false`, default `false`): the configured
  `ground_speed` source column is in knots and is converted to m/s on load. A
  column already named `ground_speed` is taken as the canonical m/s schema.
* `geo_alt_is_ft` (`true`/`false`, default `false`): the configured `geo_alt`
  source column is in feet and is converted to meters on load. A column already
  named `geo_alt` is taken as the canonical meters schema.
* `geo_alt_is_msl` (`true`/`false`, default `false`): the backend's `geo_alt` is
  a height above mean sea level instead of the WGS-84 ellipsoid. Geometric
  altitudes are compared as ellipsoidal heights (what ADS-B transmits, RTCA
  DO-260B); an MSL source shows a constant offset (high `alt_rmse`, low
  `alt_nmad`) equal to the geoid undulation. With `true`, h = H<sub>MSL</sub> + N
  is applied, with N from EGM96 (`egm96` package, grid bundled, nothing to
  download). Cached per-flight CSVs keep the delivered values, so the flag can
  change without refetching.

## Local data format

Local flight data is one file per flight:

```
{path}/{collection_id}.parquet    # primary format (also .pq)
{path}/{collection_id}.csv        # fallback (e.g. hand-edited files)
```

Parquet is preferred when both exist; the first hit of `.parquet`, `.pq`, `.csv`
wins.

### Schema

All data is loaded into the canonical schema below. Alternative column names are
either configured in `config.yaml` (`columns`, `sum_columns`, `baro_params`) or
matched against the generic spellings here. Data-set-specific names (e.g. the
onboard parameter names of a fleet) occur **only** in `config.yaml`, never in
the code.

| Column         | Type                  | Unit | Required | Generic spellings (examples) |
|----------------|-----------------------|------|----------|------------------------------|
| `time`         | float epoch seconds UTC, or ISO 8601 datetime | s | yes | `Time`, `timestamp` |
| `lat`          | float                 | degree | yes | `latitude` |
| `lon`          | float                 | degree | yes | `longitude` |
| `geo_alt`      | float, WGS-84 ellipsoid (or MSL with `geo_alt_is_msl`) | m | yes | `geoaltitude`, `altitude`; ft with `geo_alt_is_ft` |
| `baro_alt`     | float                 | m    | no  | `baroaltitude`; raw parameters via `baro_params` |
| `ground_speed` | float                 | m/s  | no  | `velocity`, `groundspeed`; kt with `ground_speed_is_kt` |
| `nacv`         | int (0-4)             | –    | no  | `nacv` |
| `position_nac` | int                   | –    | no  | `positionnac` |
| `gva`          | int (0-2)             | –    | no  | `geometricverticalaccuracy` |

All altitudes are in **meters** and `time` is normalized to **unix epoch seconds
(UTC)** on load. For OSN data (`opensky`, `local_public_ref`) the row time is the
position-fix time (`lastposupdate`); the snapshot time is only used for rows
without a fix. Extra columns are kept and shown in the plots. The OSN kinematics
columns (`velocity`, `heading`, `vertrate`) are optional: without them the
`kalman_smoother` variant is skipped, since it fuses the positions with these
velocities, while `mean` runs on the positions alone.

### Flight metadata sidecar (optional)

A `{path}/{collection_id}.meta` sidecar (YAML) can hold the flight identity so
the opensky target can resolve the OpenSky data of a fully local source:

```yaml
icao_hex: 3c6444             # ICAO 24-bit address of the aircraft
start: 1696153787            # flight start (unix seconds or a timestamp)
stop: 1696156387             # flight stop
```

`start`/`stop` accept unix epoch seconds or timestamps; timestamp-like YAML
values are read as UTC when they carry no timezone. Missing or unparseable
values make the sidecar incomplete, and the flight proceeds without a resolved
identity.

Example for anonymized onboard data with its original parameter names (see the
`local_file_in` template in `config.yaml`):

```yaml
data_fetching:
  local_file_in:
    path: data/store/in
    columns:
      time: Time
      lat: latitude
      lon: longitude
      geo_alt: altitude
      baro_alt: baroaltitude
    baro_params:
      - baro_altitude_1
      # ...
```

### Example (CSV)

```csv
time,lat,lon,geo_alt,baro_alt,positionnac
1696153787.5,51.40123,6.76543,10972.8,10972.5,3
1696153788.5,51.40141,6.76598,10975.4,10974.9,3
```

The same data as parquet (recommended to store anonymized onboard data):

```python
import pandas as pd

df = pd.DataFrame({
    "time": [1696153787.5, 1696153788.5],
    "lat": [51.40123, 51.40141],
    "lon": [6.76543, 6.76598],
    "geo_alt": [10972.8, 10975.4],       # meters
    "baro_alt": [10972.5, 10974.9],      # meters
    "position_nac": [3, 3],
})
df.to_parquet("data/store/in/<collection_id>.parquet", index=False)
```

## How source and target are compared

The source is the reference (e.g. onboard data), the target the data under test
(e.g. about one ADS-B position report per second). Every **position report of
the target** is one evaluated row, compared with the source interpolated to the
report time:

* **Source:** only its real samples. A GNSS fix stored in rows of a higher rate
  repeats its value until the next fix; those rows are dropped for `lat`, `lon`
  and `geo_alt`, so the reference does not lag by up to one fix interval. Other
  columns (e.g. a 20 Hz `baro_alt`) keep all samples. Reports outside the source
  or in source gaps wider than `max_gap_s` have no reference and are dropped.
* **Target:** the reports as delivered, with their own NACp claim.
* **Filter variants:** `mean` (centred 5 s rolling mean), `median` (centred
  rolling median over 5 reports, an outlier filter) and `kalman_smoother`
  (traffic's `KalmanSmoother6D`, fusing positions with the reported velocities;
  runs about 0.5 s ahead). `mean` and `kalman_smoother` run on a regular 1 s
  series, `median` on the reports themselves. Every variant is scored on the
  same reports as `original`, and all three smooth `baro_alt` and the
  ground-speed column as well.

See [docs/scores.md](docs/scores.md) for each score, the rationale and how to
read them together. The processing steps are summarized in
[docs/pipeline.md](docs/pipeline.md).

## Metrics

All metrics are computed for `original` and every filter variant.

* `pos_rmse`, `pos_p95` – horizontal position error at the report time
  (great-circle distance).
* `time_offset_s`, `pos_rmse_synced`, `pos_p95_synced` – the constant time
  offset (positive = stamped late, searched within ±1 s) and the position error
  after removing it.
* `pos_p95_tol100ms`, `pos_p95_tol300ms`, `pos_p95_tol600ms` – `pos_p95` with a
  time error of up to 0.1, 0.3 and 0.6 s tolerated.
* `pos_track_mean`, `pos_track_p95`, `pos_track_max` – distance to the flown
  path within `track_window_s` (default 2 s, the ADS-B position latency allowed
  by 14 CFR 91.227).
* `alt_bias`, `alt_rmse`, `alt_nmad`, `alt_p95` – geometric altitude error:
  median, RMSE, 1.4826·MAD and the 95th percentile of its absolute value.
* `baro_bias`, `baro_nmad` – median and spread of the barometric altitude error.
* `gs_bias`, `gs_rmse`, `gs_nmad`, `gs_p95` – ground speed error.

The NACp validation holds each report's claim against the six position errors
(time-aligned, offset removed, 0.1/0.3/0.6 s tolerance, path distance) and
reports the share of reports meeting it; a flight fulfils the claim at ≥ 95 %.
The altitude error is checked against the VEPU of the NACp and the claimed GVA,
the ground speed error against NACv. Claims exist only for part of the fleet
(OSN operational status is not transmitted by every aircraft, NACv in
particular is rare), so `--analyze` reports coverage and pass counts per claim.

No plausibility filter is applied: every delivered report is evaluated,
including gross outliers. `pos_rmse`, `alt_rmse` and `pos_track_max` are
deliberately sensitive to them; `pos_p95`, `alt_p95`, `alt_nmad`, the tolerance
scores and the filter variants are the robust view.

### References

* 14 CFR § 91.227, Automatic Dependent Surveillance-Broadcast (ADS-B) Out
  equipment performance requirements (total position latency: 2.0 s).

## Tests

`uv run pytest`

## Usage

```
python run.py <flight_id> [<flight_id> ...]   # one or more flight collection IDs
python run.py -f <ids.txt>                    # line-separated ID file
```

Options:

```
-i, --interactive   open the result plots in the browser
-j, --jobs N        process N flights in parallel (default: 4)
--rescore           recalculate the results even if cached (downloads are reused)
--export            export summary to data/summary.csv and exit
--analyze           analyse summary.pq with claim coverage and pass counts, and exit
--clean             remove all stored data (cached downloads, results, summaries)
                    after confirmation and exit
```

`--export`/`--analyze`/`--clean` cannot be combined with ids or `--file`.

Per-flight results are cached in `data/<flight_id>/` (re-runs skip
recalculation) and rolled up into `data/summary.pq` and `data/scores.pq`. The
cache does not track settings or code versions: after changing `max_gap_s`,
`track_window_s`, `geo_alt_is_msl`, the filters or the scoring code, rerun the
flights with `--rescore` (or delete their `scores.csv`; downloads stay cached),
or remove everything with `--clean`. `--clean` only removes what the pipeline
wrote into `data/`; other content, e.g. local input files in `data/store/`, is
kept.

Several flights run in parallel (`-j`, default 4; one `run.py` call at a time).
Duplicate ids are processed once, every flight writes only its own
`data/<flight_id>/` folder, and shared files (`summary.pq`/`scores.pq`,
`failed_*.txt`) are updated under a lock and written atomically. With more than
one parallel flight every output line is prefixed with its flight id. A failing
flight is reported without stopping the others; the run then exits with code 1.
With `-j 1` the flights run one after another.

---

## Authors and acknowledgment

* **Developer/Author:** Laura Keller
* **Primary Supervisor:** Emy Arts-Bautsch
* **Secondary Supervisor:** Martin Poppinga

We acknowledge the efforts of the crowdsourced ADS-B community OpenSky Network
for providing the public datasets that make this research impactful.

_Gitlab Aircraft icon credits_: <a href="https://www.vecteezy.com/free-vector/aircraft">Aircraft Vectors by Vecteezy</a>
