"""Starter forecasting utilities. Requires a reviewed flare-onset catalogue."""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "soft_log_level", "soft_slope", "soft_volatility",
    "hard_log_level", "hard_slope", "hard_volatility",
]


def make_forecast_table(curves: pd.DataFrame, catalogue: pd.DataFrame, horizon_minutes: int = 10,
                        history_minutes: int = 5, cadence_seconds: int = 10) -> pd.DataFrame:
    """Create per-timestamp examples, labelling onset in the *future* horizon.

    catalogue must contain a reviewed `onset_utc` column. Do not use peak times
    as onset labels unless that is deliberately the task definition.
    """
    if "onset_utc" not in catalogue.columns:
        raise ValueError("Reviewed flare catalogue must have an onset_utc column")
    df = curves.copy().sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    onset = pd.to_datetime(catalogue["onset_utc"], utc=True, errors="coerce").dropna().sort_values().to_numpy()
    rows = max(3, int(history_minutes * 60 / cadence_seconds))
    for col, prefix in (("solexs", "soft"), ("hel1os", "hard")):
        vals = np.log1p(pd.to_numeric(df[col], errors="coerce").clip(lower=0))
        df[f"{prefix}_log_level"] = vals.rolling(rows, min_periods=max(3, rows // 2)).mean()
        df[f"{prefix}_slope"] = vals.diff(max(1, rows // 3)) / max(1, rows // 3)
        df[f"{prefix}_volatility"] = vals.rolling(rows, min_periods=max(3, rows // 2)).std()
    horizon = np.timedelta64(horizon_minutes, "m")
    t_ns = df["timestamp"].astype("int64").to_numpy()
    onset_ns = pd.DatetimeIndex(onset).asi8
    labels = np.zeros(len(df), dtype=np.int8)
    for i, t in enumerate(t_ns):
        labels[i] = int(np.any((onset_ns > t) & (onset_ns <= t + horizon.astype("timedelta64[ns]").astype(np.int64))))
    df["label_flare_in_horizon"] = labels
    return df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
