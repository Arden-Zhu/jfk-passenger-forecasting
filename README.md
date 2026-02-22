# JFK Airport Daily Passenger Throughput Forecasting

Predicting daily passenger volume at JFK International Airport using classical
machine learning and deep learning methods.

## Project Organization

```
jfk-passenger-forecasting/
|-- README.md                          <- This file
|-- requirements.txt                   <- Python dependencies
|-- .gitignore
|
|-- data/
|   |-- raw/                           <- Original, immutable data
|   |   |-- TsaThroughput_JFK.csv      <- TSA hourly checkpoint data
|   |   +-- weather.csv                <- NOAA daily weather observations
|   |-- interim/                       <- Intermediate transformed data
|   |-- processed/                     <- Final datasets for modeling
|   +-- external/                      <- Third-party reference data
|
|-- notebooks/
|   |-- 01_data_preprocessing.ipynb    <- Data cleaning & EDA
|
|-- src/                               <- Reusable source code
|
|-- models/                            <- Serialized trained models
|-- reports/figures/                    <- Generated plots
|-- references/                        <- Data dictionary, papers
+-- docs/                              <- Additional documentation
```

---

## Data Sources

### 1. TSA Security Checkpoint Throughput (Target Variable)

| Item        | Detail |
|-------------|--------|
| **Source**   | TSA FOIA Electronic Reading Room |
| **URL**      | https://www.tsa.gov/foia/readingroom |
| **Obtained** | Via [mikelor/TsaThroughput](https://github.com/mikelor/TsaThroughput) GitHub repository (`data/processed/tsa/throughput/csv/TsaThroughput.JFK.csv`) |
| **Scope**    | JFK International Airport only |
| **Period**   | 2018-12-30 to 2025-05-31 |
| **Granularity** | Hourly, per checkpoint |
| **Rows**     | ~56,000 (hourly); ~2,340 after daily aggregation |
| **License**  | Public domain (U.S. Government FOIA) |

The TSA publishes weekly PDF reports containing hourly passenger throughput
counts for every security checkpoint at every U.S. airport. The mikelor
repository parses these PDFs into machine-readable CSV files. Each row
represents one hour at one checkpoint; the daily target variable is computed
by summing all terminals and all hours for each date.

### 2. NOAA Daily Weather Observations (Features)

| Item        | Detail |
|-------------|--------|
| **Source**   | NOAA Climate Data Online (CDO) |
| **URL**      | https://www.ncdc.noaa.gov/cdo-web/search |
| **Station**  | USW00094789 -- JFK INTERNATIONAL AIRPORT, NY US |
| **Dataset**  | Global Historical Climatology Network - Daily (GHCND) |
| **Period**   | 2017-01-01 to 2026-02-01 |
| **Granularity** | Daily |
| **Rows**     | ~3,319 |
| **License**  | Public domain (U.S. Government) |

Ordered from NOAA CDO with the following settings:
- Weather Observation Type: **Daily Summaries**
- Output Format: **CSV**
- Station: **GHCND:USW00094789**

### 3. BTS On-Time Performance — Scheduled Flights (Features, optional)

| Item        | Detail |
|-------------|--------|
| **Source**   | Bureau of Transportation Statistics (BTS) |
| **URL**      | https://www.transtats.bts.gov/DL_SelectFields.aspx |
| **Dataset**  | On-Time Reporting Carrier On-Time Performance |
| **Period**   | Jan 2019 to latest available (~Oct 2025) |
| **Granularity** | Per-flight record, aggregated to daily |
| **License**  | Public domain (U.S. Government) |

Individual flight records filtered to JFK (origin or destination), then
aggregated to daily counts of scheduled departures and arrivals. Only
**scheduled** (not actual) counts are used as features, since flight
schedules are published months in advance and are known at prediction time.

**Download:** Run `data/raw/bts_flights.download_all.bat` to download files, then `python src/data/download_bts_flights.py` to process.

### 4. U.S. Federal Holidays (Features, generated)

Generated programmatically using the Python `holidays` library. No download
required.

---

## Data Dictionary

### TSA Throughput -- `TsaThroughput_JFK.csv`

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `Date` | date | YYYY-MM-DD | Observation date |
| `Hour` | time | HH:MM:SS | Hour of day (00:00 to 23:00) |
| `JFK Terminal 1` | float | passengers | T1 checkpoint -- international carriers |
| `JFK Terminal 2` | float | passengers | T2 checkpoint -- **closed ~2023**, expect NaN after closure |
| `JFK Terminal 4 Delta 1` | float | passengers | T4 Delta-dedicated checkpoint |
| `JFK Terminal 4 FIS CP` | float | passengers | T4 Federal Inspection Station (inbound re-screening, sparse) |
| `JFK Terminal 4 Main` | float | passengers | T4 main checkpoint -- **busiest terminal** |
| `JFK Terminal 5` | float | passengers | T5 checkpoint -- JetBlue hub |
| `JFK Terminal 7` | float | passengers | T7 checkpoint -- **closed ~2024**, expect NaN after closure |
| `JFK Terminal 8` | float | passengers | T8 checkpoint -- American Airlines hub |

**Processing notes:**
- NaN in terminal columns = checkpoint not operating that hour; treat as 0.
- Daily target = sum of all terminal columns across all 24 hours for each date.

### NOAA Weather -- `weather.csv`

#### Columns to keep (features)

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `DATE` | date | YYYY-MM-DD | Observation date |
| `AWND` | float | mph | Average daily wind speed |
| `PRCP` | float | inches | Total daily precipitation ("T" = trace, treat as 0) |
| `SNOW` | float | inches | Total daily snowfall ("T" = trace, treat as 0) |
| `SNWD` | float | inches | Snow depth on ground |
| `TAVG` | int | deg F | Average temperature |
| `TMAX` | int | deg F | Maximum temperature |
| `TMIN` | int | deg F | Minimum temperature |
| `WSF2` | float | mph | Fastest 2-minute sustained wind speed |
| `WSF5` | float | mph | Fastest 5-second wind gust |
| `WT01` | binary | -- | Fog (1 = occurred) |
| `WT02` | binary | -- | Heavy fog |
| `WT03` | binary | -- | Thunder |
| `WT04` | binary | -- | Ice pellets |
| `WT05` | binary | -- | Hail |
| `WT06` | binary | -- | Glaze / rime ice |
| `WT08` | binary | -- | Smoke / haze |
| `WT09` | binary | -- | Blowing snow |

#### Columns to drop

| Column | Reason |
|--------|--------|
| `STATION`, `NAME`, `LATITUDE`, `LONGITUDE`, `ELEVATION` | Constant (single station) |
| `PGTM` | Peak gust time -- mostly missing |
| `WDF2`, `WDF5` | Wind direction in degrees -- circular variable, low predictive value |
| All `*_ATTRIBUTES` columns | NOAA quality flags, not needed for modeling |

### BTS Scheduled Flights -- `jfk_daily_scheduled_flights.csv` (generated)

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `date` | date | YYYY-MM-DD | Observation date |
| `scheduled_departures` | int | flights | Flights scheduled to depart from JFK |
| `scheduled_arrivals` | int | flights | Flights scheduled to arrive at JFK |
| `total_scheduled_flights` | int | flights | Departures + arrivals |
| `num_carriers` | int | airlines | Unique airlines operating that day |

**Note:** Only `scheduled_departures`, `scheduled_arrivals`, and
`total_scheduled_flights` should be used as model features. `cancelled_flights`
and `cancellation_rate` are only available after the fact and would constitute
data leakage if used for forecasting.

---

## Processed Files (`data/processed/`)

| File | Generated by | Description |
|------|-------------|-------------|
| `jfk_daily_scheduled_flights.csv` | `src/data/download_bts_flights.py` | Daily scheduled departure/arrival counts for JFK, aggregated from raw BTS per-flight records. Intermediate output -- kept as input for Notebook 01. |
| `jfk_daily_merged.csv` | `notebooks/01_data_preprocessing.ipynb` | **Primary modeling dataset.** Three-way merge of TSA throughput (daily) + NOAA weather + BTS scheduled flights. This is the file all downstream notebooks read. |

---

## Getting Started

This project uses **conda** (for Python version management) + **uv** (for
fast, reproducible package management).

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed

### Setup

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd jfk-passenger-forecasting

# 2. Create conda environment (Python 3.11 + uv)
conda env create -f environment.yml

# 3. Activate the conda environment
conda activate jfk-forecast

# 4. Install all Python packages via uv
uv pip install -e ".[dev]"
```

### Day-to-day usage

```bash
conda activate jfk-forecast    # always activate conda first
jupyter notebook                # run notebooks 
uv pip install <package-name>  # add a new package
```

---

*Project template: Cookiecutter Data Science.*
