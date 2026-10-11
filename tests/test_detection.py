import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from katyani.detection import detect_flares


def test_detector_returns_expected_schema():
    rng = np.random.default_rng(42)
    t = pd.date_range("2025-01-01", periods=500, freq="10s", tz="UTC")
    soft = rng.poisson(10, 500).astype(float)
    hard = rng.poisson(5, 500).astype(float)
    soft[300:307] += np.array([20, 35, 55, 80, 60, 35, 20])
    hard[300:307] += np.array([12, 20, 40, 70, 50, 25, 15])
    df = pd.DataFrame({"timestamp": t, "solexs": soft, "hel1os": hard})
    annotated, events = detect_flares(df, cadence_seconds=10, threshold=4, baseline_minutes=5)
    assert "solexs_score" in annotated.columns
    assert "hel1os_trigger" in annotated.columns
    assert list(events.columns) == ["source", "start_utc", "peak_utc", "end_utc", "peak_value", "max_robust_score", "samples_over_threshold", "threshold"]
