# Katyani — The Flare Guardian

A starter research prototype for combining Aditya-L1 SoLEXS soft X-ray and HEL1OS hard X-ray light curves to explore automated flare nowcasting and, after label validation, forecasting.

> **Important:** The dashboard's bundled sample is synthetic demonstration data. It is not Aditya-L1 data and its event detections are not scientific results. This project is not an operational space-weather warning service.

## Research workflow

1. **Inspect** one real file from each instrument before bulk processing.
2. **Parse and align** time series with verified timestamp semantics, retaining gaps rather than inventing measurements.
3. **Nowcast** exploratory candidate events using a transparent past-only rolling-median/MAD baseline for each payload.
4. **Validate** event candidates against data-quality flags and a separately curated event catalogue; merge detections within a documented temporal tolerance.
5. **Forecast** only after a vetted `onset_utc` catalogue is available. Build labels for events in a fixed future horizon, use chronological train/test splits, and report precision/recall, false alarms per unit time, and lead-time distributions.
6. **Visualize** observations and candidate alerts in the Streamlit dashboard.

## Instrument context

The official Aditya-L1 PRADAN portal describes SoLEXS as covering 2–22 keV and HEL1OS as covering 10–150 keV, with HEL1OS supplying light curves and Type-II spectra. Confirm product details and time references in the matching instrument documentation for the exact file version downloaded.

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use `\.venv\Scripts\python.exe -m pip install -r requirements.txt` and `\.venv\Scripts\python.exe -m streamlit run app.py`.

## Step 1 — inspect the downloaded files

```powershell
python scripts/inspect_files.py "data/raw/solexs/example.fits" "data/raw/hel1os/example.fits"
```

This prints HDU/table names, columns, sample values, and common time-reference keywords. **Do not assume a numeric TIME column is Unix time**: the reader will refuse some ambiguous elapsed-time cases rather than silently generating wrong UTC matches. If the table selection or timestamp reference is uncertain, share the inspection output before running the pipeline on all files.

## Step 2 — align the light curves

```powershell
python scripts/prepare_curves.py --solexs "path/to/solexs_lightcurve.fits" --hel1os "path/to/hel1os_lightcurve.fits" --out data/processed/combined_curves.csv --cadence 10s
```

If auto-detection chooses wrong columns, specify `--soft-time-col`, `--soft-value-col`, `--hard-time-col`, and `--hard-value-col`. Only use these when the chosen columns are confirmed to represent a timestamp and a light-curve measurement. The script uses bin means and leaves missing bins as missing; no interpolation is performed across gaps.

## Step 3 — exploratory nowcasting

```powershell
python scripts/run_nowcast.py --input data/processed/combined_curves.csv --out-dir outputs --cadence-seconds 10 --threshold 5
```

Outputs `outputs/nowcast_timeseries.csv` and `outputs/nowcast_catalogue.csv`. The robust-score threshold is a tunable starting point, not a scientifically calibrated universal threshold. Assess performance per instrument and per flare strength, and tune on training/validation periods only.

## Step 4 — dashboard

```powershell
streamlit run app.py
```

The dashboard accepts a prepared CSV with `timestamp`, `solexs`, and `hel1os` columns. It also includes synthetic data for an interface demonstration. Do not present synthetic demo alerts as mission detections.

## Forecast labels and fair evaluation

A forecast target should be defined unambiguously, for example: *does a reviewed flare onset occur in the next 10 minutes?* Keep event onset separate from peak time. A catalogue derived from the same threshold detector is useful for debugging but is not an independent scientific ground truth. Split data chronologically, prevent overlapping-window leakage, and report event-level recall, false alarms per hour/day, precision-recall metrics, and warning lead time relative to onset (and peak separately). GOES A/B/C/M/X class labels cannot be inferred directly from arbitrary SoLEXS/HEL1OS counts without appropriate calibration or external labels.

## Project structure

```text
app.py
scripts/
  inspect_files.py
  prepare_curves.py
  run_nowcast.py
src/katyani/
  data_io.py
  processing.py
  detection.py
  forecasting.py
data/demo/combined_demo.csv
tests/test_detection.py
```

## Current limitations

- Format and timing assumptions must be verified against your exact SoLEXS/HEL1OS products before full analysis.
- The current detector is a transparent baseline, not a tuned or validated flare classifier.
- `forecasting.py` provides future-horizon label/feature preparation; a trained forecaster and calibrated probability are not included yet.
- No real-time PRADAN ingestion or operational alerting is implemented.
