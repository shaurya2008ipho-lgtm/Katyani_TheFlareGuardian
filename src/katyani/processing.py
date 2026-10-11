from __future__ import annotations
import pandas as pd
from .data_io import read_light_curve


def combine_curves(soft_path: str, hard_path: str, cadence: str = "10s", soft_time_col: str | None = None,
                   soft_value_col: str | None = None, hard_time_col: str | None = None,
                   hard_value_col: str | None = None) -> pd.DataFrame:
    """Load and time-align SoLEXS/HEL1OS on a regular cadence using bin means.

    Large gaps are left missing; the function does not interpolate across them.
    """
    soft = read_light_curve(soft_path, "solexs", soft_time_col, soft_value_col).set_index("timestamp")["value"].rename("solexs")
    hard = read_light_curve(hard_path, "hel1os", hard_time_col, hard_value_col).set_index("timestamp")["value"].rename("hel1os")
    soft_binned = soft.resample(cadence).mean()
    hard_binned = hard.resample(cadence).mean()
    combined = pd.concat([soft_binned, hard_binned], axis=1).sort_index()
    combined.index.name = "timestamp"
    return combined.reset_index()
