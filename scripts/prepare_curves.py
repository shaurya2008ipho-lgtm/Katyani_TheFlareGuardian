from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from katyani.processing import combine_curves

p = argparse.ArgumentParser(description="Prepare aligned SoLEXS + HEL1OS light curves")
p.add_argument("--solexs", required=True, help="SoLEXS light-curve CSV/FITS path")
p.add_argument("--hel1os", required=True, help="HEL1OS light-curve CSV/FITS path")
p.add_argument("--out", default="data/processed/combined_curves.csv")
p.add_argument("--cadence", default="10s", help="Pandas resampling frequency, e.g. 10s, 30s, 1min")
p.add_argument("--soft-time-col", default=None)
p.add_argument("--soft-value-col", default=None)
p.add_argument("--hard-time-col", default=None)
p.add_argument("--hard-value-col", default=None)
a = p.parse_args()
try:
    df = combine_curves(a.solexs, a.hel1os, a.cadence, a.soft_time_col, a.soft_value_col, a.hard_time_col, a.hard_value_col)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} aligned bins to {out}")
    print(f"Time range: {df.timestamp.min()} to {df.timestamp.max()}")
    print(f"Missing SoLEXS bins: {df.solexs.isna().sum():,}; missing HEL1OS bins: {df.hel1os.isna().sum():,}")
except Exception as exc:
    raise SystemExit(f"Could not prepare curves: {exc}\nRun scripts/inspect_files.py on the source files and verify the timestamp reference before retrying.")
