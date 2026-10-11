from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from katyani.detection import detect_flares

p = argparse.ArgumentParser(description="Run baseline, past-only robust-score nowcasting")
p.add_argument("--input", default="data/processed/combined_curves.csv")
p.add_argument("--out-dir", default="outputs")
p.add_argument("--cadence-seconds", type=int, default=10)
p.add_argument("--threshold", type=float, default=5.0)
p.add_argument("--baseline-minutes", type=float, default=15.0)
a = p.parse_args()
df = pd.read_csv(a.input, parse_dates=["timestamp"])
annotated, events = detect_flares(df, a.cadence_seconds, a.threshold, a.baseline_minutes)
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
annotated.to_csv(out / "nowcast_timeseries.csv", index=False)
events.to_csv(out / "nowcast_catalogue.csv", index=False)
print(f"Saved annotated time series and {len(events)} candidate events to {out.resolve()}")
print("These are preliminary algorithmic candidates; review against instrument quality flags and a vetted flare catalogue.")
