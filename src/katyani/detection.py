"""Transparent baseline nowcaster using past-only rolling median and MAD."""
from __future__ import annotations
import numpy as np
import pandas as pd


def robust_scores(series: pd.Series, window: int = 90, min_periods: int = 30) -> pd.Series:
    """Robust z-score against *previous* values only (no centered-window leakage)."""
    baseline = series.rolling(window=window, min_periods=min_periods).median().shift(1)
    abs_dev = (series - baseline).abs()
    mad = abs_dev.rolling(window=window, min_periods=min_periods).median().shift(1)
    scale = 1.4826 * mad
    # A small global floor handles very quiet/quantized series without division by zero.
    finite = series[np.isfinite(series)]
    floor = max(float(finite.abs().median()) * 1e-6 if len(finite) else 0.0, 1e-12)
    return (series - baseline) / scale.clip(lower=floor)


def detect_flares(curves: pd.DataFrame, cadence_seconds: int = 10, threshold: float = 5.0,
                  baseline_minutes: float = 15.0, merge_gap_minutes: float = 2.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return annotated time series and a simple event catalogue.

    This is a research baseline, not a validated operational alert system.
    Expected columns: timestamp, solexs, hel1os. Event start/end are threshold
    crossings; peak is the maximum robust score within an event.
    """
    required = {"timestamp", "solexs", "hel1os"}
    missing = required - set(curves.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = curves.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    window = max(3, int(baseline_minutes * 60 / cadence_seconds))
    active = []
    for source in ("solexs", "hel1os"):
        df[f"{source}_score"] = robust_scores(pd.to_numeric(df[source], errors="coerce"), window=window, min_periods=max(5, window // 3))
        df[f"{source}_trigger"] = df[f"{source}_score"] >= threshold
        flags = df[f"{source}_trigger"].fillna(False).to_numpy(dtype=bool)
        times = df["timestamp"].to_numpy()
        values = pd.to_numeric(df[source], errors="coerce").to_numpy(dtype=float)
        scores = df[f"{source}_score"].to_numpy(dtype=float)
        idx = np.flatnonzero(flags)
        if not len(idx):
            continue
        groups = []
        group = [idx[0]]
        max_gap = pd.Timedelta(minutes=merge_gap_minutes).total_seconds()
        for current in idx[1:]:
            gap_s = (pd.Timestamp(times[current]) - pd.Timestamp(times[group[-1]])).total_seconds()
            if gap_s <= max_gap + cadence_seconds:
                group.append(current)
            else:
                groups.append(group)
                group = [current]
        groups.append(group)
        for g in groups:
            p = g[int(np.nanargmax(scores[g]))]
            active.append({
                "source": source,
                "start_utc": pd.Timestamp(times[g[0]]).isoformat(),
                "peak_utc": pd.Timestamp(times[p]).isoformat(),
                "end_utc": pd.Timestamp(times[g[-1]]).isoformat(),
                "peak_value": float(values[p]),
                "max_robust_score": float(np.nanmax(scores[g])),
                "samples_over_threshold": len(g),
                "threshold": threshold,
            })
    events = pd.DataFrame(active, columns=["source", "start_utc", "peak_utc", "end_utc", "peak_value", "max_robust_score", "samples_over_threshold", "threshold"])
    if not events.empty:
        events = events.sort_values("peak_utc").reset_index(drop=True)
    return df, events
