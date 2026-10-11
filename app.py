from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from katyani.detection import detect_flares
except Exception as import_exc:  # keep the dashboard able to explain a setup issue
    detect_flares = None  # type: ignore[assignment]
    DETECTION_IMPORT_ERROR = str(import_exc)
else:
    DETECTION_IMPORT_ERROR = ""

DATA_PATH = ROOT / "data" / "processed" / "combined_curves.csv"
DEMO_PATH = ROOT / "data" / "demo" / "combined_demo.csv"
SAVED_CATALOGUE_PATH = ROOT / "outputs" / "nowcast_catalogue.csv"
SAVED_TIMESERIES_PATH = ROOT / "outputs" / "nowcast_timeseries.csv"

SOFT_COLOR = "#FFB454"
HARD_COLOR = "#50C8D8"
ACCENT = "#B9A2FF"

st.set_page_config(
    page_title="Katyani — The Flare Guardian",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: radial-gradient(ellipse at 12% 0%, #15253b 0%, #0b1220 42%, #080d17 100%); }
    section[data-testid="stSidebar"] { background: #0b1321; border-right: 1px solid #26344a; }
    .hero { padding: 1.25rem 1.4rem; border: 1px solid #2b3d58; border-radius: 18px;
            background: linear-gradient(115deg, rgba(27,48,75,.92), rgba(17,28,46,.88)); margin-bottom: 1rem; }
    .hero-kicker { letter-spacing: .16em; text-transform: uppercase; font-size: .72rem; color: #9fb7d5; font-weight: 700; }
    .hero h1 { margin: .25rem 0 .25rem 0; font-size: 2.25rem; color: #f5f8ff; }
    .hero p { color: #c4d1e3; margin: 0; }
    .status-line { border: 1px solid #344862; background: rgba(20,34,54,.7); padding: .75rem 1rem;
                   border-radius: 12px; margin: .5rem 0 1rem 0; color: #d7e5f7; }
    .small-muted { color: #a5b4c8; font-size: .88rem; }
    .section-card { border: 1px solid #2b3c53; border-radius: 14px; padding: 1rem; background: rgba(14,24,39,.7); }
    div[data-testid="stMetric"] { border: 1px solid #263951; border-radius: 12px; padding: .75rem 1rem; background: rgba(21,35,55,.65); }
    .tag { display:inline-block; border:1px solid #405572; border-radius:999px; padding:.15rem .55rem; color:#cfdef0; font-size:.78rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Aditya-L1 · X-ray time-series intelligence</div>
      <h1>☀️ Katyani — The Flare Guardian</h1>
      <p>Soft + hard X-ray observation fusion · Candidate nowcasting · Forecast-readiness workspace</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def load_curve_csv(source: Any) -> pd.DataFrame:
    frame = pd.read_csv(source)
    required = {"timestamp", "solexs", "hel1os"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prepared CSV must contain {sorted(required)}. Missing: {sorted(missing)}")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["solexs"] = pd.to_numeric(frame["solexs"], errors="coerce")
    frame["hel1os"] = pd.to_numeric(frame["hel1os"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    if frame.empty:
        raise ValueError("No valid timestamps were found in the prepared CSV.")
    return frame


def infer_cadence_seconds(frame: pd.DataFrame) -> int:
    delta = frame["timestamp"].sort_values().diff().dt.total_seconds()
    delta = delta[(delta > 0) & np.isfinite(delta)]
    if delta.empty:
        return 10
    return max(1, int(round(float(delta.median()))))


def empty_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["source", "start_utc", "peak_utc", "end_utc", "peak_value", "max_robust_score", "samples_over_threshold", "threshold"]
    )


def standardize_events(events: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["source", "start_utc", "peak_utc", "end_utc", "peak_value", "max_robust_score", "samples_over_threshold", "threshold"]
    if events is None or events.empty:
        out = empty_events()
        out["event_id"] = pd.Series(dtype=str)
        out["duration_seconds"] = pd.Series(dtype=float)
        out["review_priority"] = pd.Series(dtype=str)
        return out
    events = events.copy()
    for col in base_cols:
        if col not in events.columns:
            events[col] = np.nan
    for col in ["start_utc", "peak_utc", "end_utc"]:
        events[col] = pd.to_datetime(events[col], utc=True, errors="coerce")
    for col in ["peak_value", "max_robust_score", "samples_over_threshold", "threshold"]:
        events[col] = pd.to_numeric(events[col], errors="coerce")
    events["source"] = events["source"].astype(str).str.lower()
    events = events[events["source"].isin(["solexs", "hel1os"])].copy()
    events = events.sort_values("peak_utc").reset_index(drop=True)
    events["event_id"] = [f"KAT-{i:03d}" for i in range(1, len(events) + 1)]
    events["duration_seconds"] = (events["end_utc"] - events["start_utc"]).dt.total_seconds().clip(lower=0)

    def priority(row: pd.Series) -> str:
        n = row.get("samples_over_threshold", 0)
        score = row.get("max_robust_score", np.nan)
        if pd.isna(n) or n <= 1:
            return "Single-bin spike · inspect"
        if pd.notna(score) and score >= 20:
            return "High-priority candidate"
        return "Candidate · review"

    events["review_priority"] = events.apply(priority, axis=1)
    return events


def event_label(row: pd.Series) -> str:
    peak = row.get("peak_utc")
    peak_text = peak.strftime("%Y-%m-%d %H:%M:%S UTC") if pd.notna(peak) else "unknown time"
    score = row.get("max_robust_score", np.nan)
    score_text = f"score {score:.1f}" if pd.notna(score) else "score n/a"
    return f"{row.get('event_id', 'KAT')} · {str(row.get('source', '')).upper()} · {peak_text} · {score_text}"


def pair_cross_instrument_events(events: pd.DataFrame, tolerance_seconds: int) -> pd.DataFrame:
    columns = ["soft_event_id", "soft_peak_utc", "hard_event_id", "hard_peak_utc", "peak_delta_seconds", "interpretation"]
    if events.empty:
        return pd.DataFrame(columns=columns)
    ev = events.copy().reset_index(drop=True)
    soft = ev[ev["source"] == "solexs"]
    hard = ev[ev["source"] == "hel1os"]
    possible: list[tuple[float, int, int]] = []
    for si, sr in soft.iterrows():
        for hi, hr in hard.iterrows():
            if pd.isna(sr["peak_utc"]) or pd.isna(hr["peak_utc"]):
                continue
            delta = abs((sr["peak_utc"] - hr["peak_utc"]).total_seconds())
            if delta <= tolerance_seconds:
                possible.append((delta, int(si), int(hi)))
    possible.sort(key=lambda item: item[0])
    used_soft: set[int] = set()
    used_hard: set[int] = set()
    rows = []
    for delta, si, hi in possible:
        if si in used_soft or hi in used_hard:
            continue
        sr, hr = ev.loc[si], ev.loc[hi]
        samples_s = sr.get("samples_over_threshold", np.nan)
        samples_h = hr.get("samples_over_threshold", np.nan)
        single = (pd.notna(samples_s) and samples_s <= 1) or (pd.notna(samples_h) and samples_h <= 1)
        rows.append({
            "soft_event_id": sr["event_id"],
            "soft_peak_utc": sr["peak_utc"],
            "hard_event_id": hr["event_id"],
            "hard_peak_utc": hr["peak_utc"],
            "peak_delta_seconds": delta,
            "interpretation": "Coincident candidates; at least one is single-bin" if single else "Coincident candidates; inspect both curves",
        })
        used_soft.add(si)
        used_hard.add(hi)
    return pd.DataFrame(rows, columns=columns)


def transform_values(values: pd.Series, use_log1p: bool) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if use_log1p:
        # Signed log preserves negative background-subtracted measurements if present.
        return np.sign(arr) * np.log1p(np.abs(arr))
    return arr


def make_overview_plot(frame: pd.DataFrame, events: pd.DataFrame, use_log1p: bool, show_candidates: bool) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
        subplot_titles=("SoLEXS · soft X-ray · 2–22 keV", "HEL1OS CZT1 · hard X-ray · 20–40 keV"),
    )
    for source, row, color in [("solexs", 1, SOFT_COLOR), ("hel1os", 2, HARD_COLOR)]:
        raw = frame[source]
        y = transform_values(raw, use_log1p)
        fig.add_trace(
            go.Scatter(
                x=frame["timestamp"], y=y, mode="lines", name="SoLEXS · 2–22 keV" if source == "solexs" else "HEL1OS CZT1 · 20–40 keV",
                line=dict(color=color, width=1.4),
                customdata=np.column_stack([raw.to_numpy(dtype=float)]),
                hovertemplate="UTC %{x|%Y-%m-%d %H:%M:%S}<br>Displayed: %{y:.3f}<br>Recorded value: %{customdata[0]:.4g}<extra></extra>",
                connectgaps=False,
            ), row=row, col=1
        )
        if show_candidates and not events.empty:
            for _, ev in events[events["source"] == source].iterrows():
                if pd.isna(ev["peak_utc"]):
                    continue
                x0 = ev["start_utc"] if pd.notna(ev["start_utc"]) else ev["peak_utc"]
                x1 = ev["end_utc"] if pd.notna(ev["end_utc"]) else ev["peak_utc"]
                fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.14, line_width=0, row=row, col=1)
                fig.add_vline(x=ev["peak_utc"], line=dict(color=color, width=1, dash="dot"), row=row, col=1)
    y_label = "Signed log1p(recorded value)" if use_log1p else "Recorded value (source units)"
    fig.update_yaxes(title_text=y_label, row=1, col=1, gridcolor="rgba(157,181,210,.14)")
    fig.update_yaxes(title_text=y_label, row=2, col=1, gridcolor="rgba(157,181,210,.14)")
    fig.update_xaxes(title_text="Time (UTC)", row=2, col=1, gridcolor="rgba(157,181,210,.14)")
    fig.update_layout(
        template="plotly_dark", height=680, margin=dict(l=15, r=15, t=70, b=15),
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,15,27,.55)",
    )
    return fig


def make_event_detail_plot(frame: pd.DataFrame, event: pd.Series, events: pd.DataFrame, use_log1p: bool) -> go.Figure:
    peak = event["peak_utc"]
    start = event["start_utc"] if pd.notna(event["start_utc"]) else peak
    end = event["end_utc"] if pd.notna(event["end_utc"]) else peak
    padding = pd.Timedelta(minutes=5)
    window = frame[(frame["timestamp"] >= start - padding) & (frame["timestamp"] <= end + padding)].copy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.11,
                        subplot_titles=("SoLEXS · 2–22 keV", "HEL1OS CZT1 · 20–40 keV"))
    for source, row, color in [("solexs", 1, SOFT_COLOR), ("hel1os", 2, HARD_COLOR)]:
        raw = window[source]
        fig.add_trace(go.Scatter(
            x=window["timestamp"], y=transform_values(raw, use_log1p), mode="lines+markers", marker=dict(size=3),
            name=source.upper(), line=dict(color=color, width=1.5),
            customdata=np.column_stack([raw.to_numpy(dtype=float)]),
            hovertemplate="UTC %{x|%H:%M:%S}<br>Displayed: %{y:.3f}<br>Recorded: %{customdata[0]:.4g}<extra></extra>",
            connectgaps=False,
        ), row=row, col=1)
        fig.add_vrect(x0=start, x1=end, fillcolor=color if source == event["source"] else "#7D8BA0", opacity=0.18, line_width=0, row=row, col=1)
        fig.add_vline(x=peak, line=dict(color=color if source == event["source"] else ACCENT, dash="dash", width=1.5), row=row, col=1)
        # Mark nearby candidate peaks on the appropriate instrument panel.
        related = events[(events["source"] == source) & (events["peak_utc"] >= start - padding) & (events["peak_utc"] <= end + padding)]
        for _, candidate in related.iterrows():
            if candidate.get("event_id") == event.get("event_id"):
                continue
            fig.add_vline(x=candidate["peak_utc"], line=dict(color=color, dash="dot", width=1), row=row, col=1)
    label = "Signed log1p(recorded value)" if use_log1p else "Recorded value (source units)"
    fig.update_yaxes(title_text=label, gridcolor="rgba(157,181,210,.14)")
    fig.update_xaxes(title_text="Time (UTC)", row=2, col=1, gridcolor="rgba(157,181,210,.14)")
    fig.update_layout(template="plotly_dark", height=570, margin=dict(l=10, r=10, t=60, b=10), hovermode="x unified",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,15,27,.55)")
    return fig


# Sidebar: data source and settings
st.sidebar.markdown("## Mission console")
if DATA_PATH.exists():
    data_options = ["Use project combined_curves.csv", "Upload a prepared CSV", "Use bundled demo data"]
    default_data_index = 0
else:
    data_options = ["Use bundled demo data", "Upload a prepared CSV"]
    default_data_index = 0
source_choice = st.sidebar.radio("Observation source", data_options, index=default_data_index)
uploaded = st.sidebar.file_uploader("Prepared CSV", type=["csv"]) if source_choice == "Upload a prepared CSV" else None

is_demo = False
source_name = ""
try:
    if source_choice == "Upload a prepared CSV":
        if uploaded is None:
            st.info("Upload a prepared CSV with `timestamp`, `solexs`, and `hel1os` columns in the sidebar.")
            st.stop()
        df = load_curve_csv(uploaded)
        source_name = f"Uploaded file: {uploaded.name}"
    elif source_choice == "Use project combined_curves.csv":
        df = load_curve_csv(DATA_PATH)
        source_name = str(DATA_PATH.relative_to(ROOT))
    else:
        df = load_curve_csv(DEMO_PATH)
        is_demo = True
        source_name = "data/demo/combined_demo.csv"
except Exception as exc:
    st.error(f"Could not load observations: {exc}")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("## Nowcast controls")
threshold = st.sidebar.slider("Robust-score threshold", min_value=2.0, max_value=12.0, value=5.0, step=0.5,
                              help="Exploratory threshold. Tune only on reviewed training/validation intervals.")
baseline_minutes = st.sidebar.slider("Past-only baseline window (minutes)", min_value=5, max_value=60, value=15, step=5)
merge_gap_minutes = st.sidebar.slider("Merge gap (minutes)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
pair_tolerance_minutes = st.sidebar.slider("Cross-instrument coincidence window (minutes)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
show_candidate_spans = st.sidebar.checkbox("Show candidate spans on curves", value=True)
log1p_view = st.sidebar.checkbox("Compress extreme peaks (signed log1p view)", value=True,
                                 help="Changes the display only; exported measurements remain untransformed.")

if is_demo:
    st.warning("DEMONSTRATION MODE: bundled observations are synthetic, not Aditya-L1 data. Do not present these curves or candidates as mission results.")
else:
    st.info(f"Loaded observations: **{source_name}**. Research analysis only; not an operational space-weather warning service.")

if DETECTION_IMPORT_ERROR:
    st.error(f"The detector module could not be imported: {DETECTION_IMPORT_ERROR}")
    st.stop()

cadence_seconds = infer_cadence_seconds(df)
try:
    annotated, computed_events = detect_flares(
        df, cadence_seconds=cadence_seconds, threshold=threshold,
        baseline_minutes=float(baseline_minutes), merge_gap_minutes=float(merge_gap_minutes)
    )
except Exception as exc:
    st.error(f"Nowcasting failed: {exc}")
    annotated = df.copy()
    computed_events = empty_events()

# Allow the real CLI output to be inspected alongside interactive results.
if (not is_demo) and SAVED_CATALOGUE_PATH.exists() and source_choice == "Use project combined_curves.csv":
    catalogue_mode = st.sidebar.radio("Catalogue view", ["Saved CLI run", "Recalculate from controls"], index=0,
                                      help="Saved CLI run shows outputs/nowcast_catalogue.csv. Recalculate uses the controls above.")
else:
    catalogue_mode = "Recalculate from controls"

if catalogue_mode == "Saved CLI run":
    try:
        raw_events = pd.read_csv(SAVED_CATALOGUE_PATH)
        events = standardize_events(raw_events)
        if events.empty and not raw_events.empty:
            st.warning("Saved catalogue could not be standardized; displaying recalculated candidates instead.")
            events = standardize_events(computed_events)
            catalogue_mode = "Recalculate from controls"
    except Exception as exc:
        st.warning(f"Saved catalogue could not be loaded ({exc}); displaying recalculated candidates.")
        events = standardize_events(computed_events)
        catalogue_mode = "Recalculate from controls"
else:
    events = standardize_events(computed_events)

pairs = pair_cross_instrument_events(events, int(pair_tolerance_minutes * 60))
valid_both = int(df[["solexs", "hel1os"]].notna().all(axis=1).sum())
soft_valid = int(df["solexs"].notna().sum())
hard_valid = int(df["hel1os"].notna().sum())
all_bins = len(df)
start_time, end_time = df["timestamp"].min(), df["timestamp"].max()
coverage_hours = (end_time - start_time).total_seconds() / 3600 if all_bins > 1 else 0
max_score = events["max_robust_score"].max() if not events.empty else np.nan

# Primary app navigation
tab_overview, tab_nowcast, tab_forecast, tab_method = st.tabs(
    ["Mission overview", "Nowcasting & catalogue", "Forecasting", "Methodology & data"]
)

with tab_overview:
    st.subheader("Mission snapshot")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Time bins", f"{all_bins:,}")
    m2.metric("Shared valid bins", f"{valid_both:,}", help="Rows with a non-missing value from both instruments.")
    m3.metric("SoLEXS valid", f"{soft_valid:,}")
    m4.metric("HEL1OS valid", f"{hard_valid:,}")
    m5.metric("Candidate events", f"{len(events):,}")

    st.markdown(
        f'<div class="status-line"><b>Observation window (UTC):</b> {start_time:%Y-%m-%d %H:%M:%S} → {end_time:%Y-%m-%d %H:%M:%S} &nbsp; · &nbsp; '
        f'<b>Cadence:</b> ~{cadence_seconds}s &nbsp; · &nbsp; <b>Hours covered:</b> {coverage_hours:.2f} &nbsp; · &nbsp; '
        f'<b>Catalogue:</b> {catalogue_mode}</div>', unsafe_allow_html=True)

    fig = make_overview_plot(df, events, log1p_view, show_candidate_spans)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})
    st.caption("Vertical dotted lines mark candidate peaks; shaded regions show the detector's grouped threshold-crossing intervals. Missing segments remain gaps, not interpolated measurements.")

    k1, k2, k3 = st.columns(3)
    k1.markdown("<div class='section-card'><span class='tag'>NOWCAST</span><h3>Candidate detection</h3><p>Past-only rolling median/MAD robust scores flag unusual soft and hard X-ray activity independently.</p></div>", unsafe_allow_html=True)
    k2.markdown("<div class='section-card'><span class='tag'>FUSION</span><h3>Cross-instrument review</h3><p>Nearby candidate peaks can be paired by time tolerance to help prioritize joint review—not to confirm a flare by themselves.</p></div>", unsafe_allow_html=True)
    k3.markdown("<div class='section-card'><span class='tag'>FORECAST</span><h3>Prediction readiness</h3><p>A trained and validated forecaster requires independently reviewed flare onsets, chronological evaluation, and calibrated alert metrics.</p></div>", unsafe_allow_html=True)

with tab_nowcast:
    st.subheader("Nowcast candidates")
    left, right = st.columns([1.3, 1])
    with left:
        st.write(f"**Catalogue displayed:** {catalogue_mode}. **Threshold setting:** {threshold:.1f} robust-score units.")
        st.caption("Events are algorithmic candidates. Strong scores indicate deviation from the local baseline, not confirmed solar-flare class or causation.")
    with right:
        source_filter = st.multiselect("Filter by instrument", ["solexs", "hel1os"], default=["solexs", "hel1os"], format_func=lambda x: "SoLEXS · soft X-ray" if x == "solexs" else "HEL1OS · hard X-ray")

    filtered = events[events["source"].isin(source_filter)].copy() if source_filter else events.iloc[0:0].copy()
    if filtered.empty:
        st.info("No candidate events match the current filter and detector settings.")
    else:
        display_cols = ["event_id", "source", "start_utc", "peak_utc", "end_utc", "peak_value", "max_robust_score", "samples_over_threshold", "duration_seconds", "review_priority"]
        display = filtered[display_cols].copy()
        display = display.rename(columns={
            "event_id": "Candidate ID", "source": "Instrument", "start_utc": "Start (UTC)", "peak_utc": "Peak (UTC)", "end_utc": "End (UTC)",
            "peak_value": "Value at max-score sample", "max_robust_score": "Max robust score", "samples_over_threshold": "Bins over threshold",
            "duration_seconds": "Span (s)", "review_priority": "Review hint",
        })
        st.dataframe(display, use_container_width=True, hide_index=True, column_config={
            "Max robust score": st.column_config.NumberColumn(format="%.2f"),
            "Value at max-score sample": st.column_config.NumberColumn(format="%.4g"),
            "Span (s)": st.column_config.NumberColumn(format="%.0f"),
        })
        download_events = events.copy()
        for time_col in ["start_utc", "peak_utc", "end_utc"]:
            if time_col in download_events:
                download_events[time_col] = download_events[time_col].astype(str)
        st.download_button("Download complete candidate catalogue (CSV)", download_events.to_csv(index=False).encode("utf-8"),
                           file_name="katyani_candidate_catalogue.csv", mime="text/csv")

        labels = {event_label(row): idx for idx, row in filtered.iterrows()}
        selected_label = st.selectbox("Inspect a candidate", list(labels.keys()))
        selected = filtered.loc[labels[selected_label]]
        detail_cols = st.columns(4)
        detail_cols[0].metric("Instrument", str(selected["source"]).upper())
        detail_cols[1].metric("Robust score", f"{selected['max_robust_score']:.2f}" if pd.notna(selected["max_robust_score"]) else "n/a")
        detail_cols[2].metric("Threshold bins", f"{int(selected['samples_over_threshold'])}" if pd.notna(selected["samples_over_threshold"]) else "n/a")
        detail_cols[3].metric("Candidate span", f"{selected['duration_seconds']:.0f} s" if pd.notna(selected["duration_seconds"]) else "n/a")
        st.plotly_chart(make_event_detail_plot(df, selected, events, log1p_view), use_container_width=True, config={"displaylogo": False, "scrollZoom": True})

    st.markdown("### Cross-instrument coincidence review")
    st.caption(f"One-to-one nearest-peak matching with a ±{pair_tolerance_minutes:g}-minute tolerance. Coincidence is a review clue, not independent validation.")
    if pairs.empty:
        st.info("No SoLEXS/HEL1OS candidate peaks fall within the selected coincidence window.")
    else:
        pair_view = pairs.copy().rename(columns={
            "soft_event_id": "SoLEXS candidate", "soft_peak_utc": "SoLEXS peak (UTC)", "hard_event_id": "HEL1OS candidate",
            "hard_peak_utc": "HEL1OS peak (UTC)", "peak_delta_seconds": "Peak separation (s)", "interpretation": "Review note",
        })
        st.dataframe(pair_view, use_container_width=True, hide_index=True)
        st.download_button("Download coincidence review (CSV)", pairs.to_csv(index=False).encode("utf-8"),
                           file_name="katyani_cross_instrument_review.csv", mime="text/csv")

with tab_forecast:
    st.subheader("Forecasting and lead-time workspace")
    st.markdown("The challenge's forecasting objective is to estimate whether a flare will begin within a defined future horizon using precursor patterns from the combined light curves.")
    horizon = st.selectbox("Target forecast horizon", [5, 10, 15, 30], index=1, format_func=lambda x: f"Next {x} minutes")
    st.info("**Forecast status: model not yet trained or validated.** This dashboard does not fabricate a future flare probability. The current dataset and detector catalogue are suitable for pipeline development, but a reviewed onset catalogue is still needed to create defensible training labels.")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Forecast target", f"Onset in next {horizon} min")
    f2.metric("Trained model", "Not available")
    f3.metric("Measured lead time", "Not evaluated")
    f4.metric("False alarms / hour", "Not evaluated")

    st.markdown("#### What the forecasting module must do")
    steps = [
        ("1 · Create reviewed labels", "Record independently reviewed flare onset times. Keep onset separate from peak time; detector-generated candidates alone are not ground truth."),
        ("2 · Build past-only windows", "Use soft/hard X-ray levels, slopes, variability, energy-band features and recent candidate context without using future data."),
        ("3 · Train with chronological splits", "Split by time/day, avoid overlapping-window leakage, and tune thresholds only on training/validation periods."),
        ("4 · Evaluate warning utility", "Report event-level recall, precision-recall performance, false alarms per hour/day, and lead time relative to onset and peak separately."),
    ]
    for title, body in steps:
        with st.container():
            st.markdown(f"**{title}**")
            st.write(body)
    st.markdown("#### Proposed forecasting output")
    st.code(
        "timestamp_utc, horizon_minutes, forecast_probability, alert_threshold, alert_state, model_version\n"
        f"# forecast_probability remains unavailable until a model is trained and calibrated for the {horizon}-minute target.",
        language="text",
    )

with tab_method:
    st.subheader("Methodology, data quality and problem-statement coverage")
    status_rows = [
        {"Challenge objective": "Automated nowcasting", "Katyani status": "Implemented as an exploratory baseline", "Evidence": "Rolling median/MAD detector and candidate catalogue"},
        {"Challenge objective": "Combine soft + hard X-rays", "Katyani status": "Implemented for prepared, overlapping observations", "Evidence": "UTC-aligned combined_curves.csv with gaps preserved"},
        {"Challenge objective": "Visualize light curves and alerts", "Katyani status": "Implemented", "Evidence": "Overview, candidate spans, close-up plots and downloadable catalogue"},
        {"Challenge objective": "Forecast future flares with lead time", "Katyani status": "Not yet trained / validated", "Evidence": "Requires reviewed onset labels and chronological evaluation"},
        {"Challenge objective": "Low/high class flare evaluation", "Katyani status": "Not established", "Evidence": "Requires a vetted reference catalogue and suitable calibration"},
    ]
    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    qa1, qa2, qa3 = st.columns(3)
    qa1.metric("Missing SoLEXS bins", f"{int(df['solexs'].isna().sum()):,}", f"{df['solexs'].isna().mean()*100:.2f}% of rows")
    qa2.metric("Missing HEL1OS bins", f"{int(df['hel1os'].isna().sum()):,}", f"{df['hel1os'].isna().mean()*100:.2f}% of rows")
    qa3.metric("Simultaneous valid bins", f"{valid_both:,}", f"{valid_both / max(all_bins, 1) * 100:.1f}% of rows")

    st.markdown("#### Current run configuration")
    config = pd.DataFrame([
        {"Setting": "Dataset source", "Value": source_name},
        {"Setting": "Start UTC", "Value": start_time.isoformat()},
        {"Setting": "End UTC", "Value": end_time.isoformat()},
        {"Setting": "Inferred cadence", "Value": f"{cadence_seconds} seconds"},
        {"Setting": "Detector threshold", "Value": threshold if catalogue_mode != "Saved CLI run" else "Saved CLI catalogue; see CSV for threshold"},
        {"Setting": "Baseline window", "Value": f"{baseline_minutes} minutes"},
        {"Setting": "Merge gap", "Value": f"{merge_gap_minutes} minutes"},
        {"Setting": "Catalogue mode", "Value": catalogue_mode},
    ])
    st.dataframe(config, use_container_width=True, hide_index=True)

    st.markdown("#### Interpretation guardrails")
    st.markdown(
        "- Candidate events are not confirmed solar flares. High robust scores quantify unusualness relative to a local baseline, not physical class.\n"
        "- SoLEXS and HEL1OS have different energy ranges, time coverage and detector responses; compare only verified common UTC intervals.\n"
        "- Never replace absent measurements with zero or interpolate through gaps for alert evaluation.\n"
        "- GOES A/B/C/M/X classes cannot be inferred directly from arbitrary instrument count values without suitable calibration or external reference labels.\n"
        "- This is a research prototype and is not an operational space-weather warning service."
    )

st.markdown("---")
st.markdown(
    "<div class='small-muted'>Katyani — The Flare Guardian · Research prototype · Candidate detections are not operational alerts. "
    "Forecast probabilities and lead-time metrics will only appear after independent label review, training and validation.</div>",
    unsafe_allow_html=True,
)
