from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from katyani.detection import detect_flares

st.set_page_config(page_title="Katyani — The Flare Guardian", page_icon="☀️", layout="wide")
st.title("☀️ Katyani — The Flare Guardian")
st.caption("SoLEXS + HEL1OS solar X-ray flare research dashboard")
st.warning("Research prototype, not an operational space-weather warning service. The bundled data is synthetic demonstration data, not Aditya-L1 observations.")

uploaded = st.file_uploader("Upload a prepared CSV with timestamp, solexs, hel1os columns", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded, parse_dates=["timestamp"])
else:
    demo = ROOT / "data" / "demo" / "combined_demo.csv"
    df = pd.read_csv(demo, parse_dates=["timestamp"])
    st.info("Showing synthetic demo data. Replace it with your prepared combined_curves.csv to inspect real observations.")

needed = {"timestamp", "solexs", "hel1os"}
if not needed.issubset(df.columns):
    st.error(f"CSV is missing columns: {sorted(needed - set(df.columns))}")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Time bins", f"{len(df):,}")
c2.metric("SoLEXS missing", f"{df['solexs'].isna().mean()*100:.1f}%")
c3.metric("HEL1OS missing", f"{df['hel1os'].isna().mean()*100:.1f}%")

st.subheader("Combined soft and hard X-ray light curves")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.timestamp, y=df.solexs, name="SoLEXS · soft X-ray", mode="lines"))
fig.add_trace(go.Scatter(x=df.timestamp, y=df.hel1os, name="HEL1OS · hard X-ray", mode="lines", yaxis="y2"))
fig.update_layout(height=520, xaxis_title="UTC time", yaxis=dict(title="SoLEXS counts/rate (source units)"),
                  yaxis2=dict(title="HEL1OS counts/rate (source units)", overlaying="y", side="right"), legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Baseline nowcast candidates")
with st.expander("Tune exploratory detector", expanded=True):
    threshold = st.slider("Robust-score threshold", min_value=2.0, max_value=10.0, value=5.0, step=0.5)
    cadence = st.selectbox("Input cadence", options=[10, 30, 60], index=0, format_func=lambda x: f"{x} seconds")
try:
    annotated, events = detect_flares(df, cadence_seconds=cadence, threshold=threshold)
    triggers = annotated[(annotated.solexs_trigger.fillna(False)) | (annotated.hel1os_trigger.fillna(False))]
    st.write(f"Threshold-crossing bins: **{len(triggers):,}** · Candidate events: **{len(events):,}**")
    if not events.empty:
        st.dataframe(events, use_container_width=True, hide_index=True)
        st.download_button("Download candidate catalogue CSV", events.to_csv(index=False).encode(), "katyani_nowcast_catalogue.csv", "text/csv")
    else:
        st.info("No threshold-crossing events found with the current settings.")
except Exception as exc:
    st.error(f"Detection could not run: {exc}")

st.subheader("Forecasting status")
st.write("Forecasting is intentionally not presented as trained yet. Next, use a reviewed event-onset catalogue to create future-horizon labels, then evaluate on dates held out from training.")
st.code("python scripts/inspect_files.py path/to/solexs_file.fits path/to/hel1os_file.fits\npython scripts/prepare_curves.py --solexs path/to/solexs_lc.fits --hel1os path/to/hel1os_lc.fits --out data/processed/combined_curves.csv --cadence 10s\npython scripts/run_nowcast.py --input data/processed/combined_curves.csv")
