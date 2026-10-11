from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from katyani.data_io import inspect_file

p = argparse.ArgumentParser(description="Inspect CSV/FITS table schema before processing")
p.add_argument("files", nargs="+", help="CSV/TXT/FITS files")
args = p.parse_args()
for name in args.files:
    print(f"\n=== {name} ===")
    try:
        print(json.dumps(inspect_file(name), indent=2, default=str))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
