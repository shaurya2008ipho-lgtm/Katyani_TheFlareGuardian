from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

data_path = Path("data/processed/combined_curves.csv")
catalogue_path = Path("outputs/nowcast_catalogue.csv")
out_dir = Path("outputs/candidate_review")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path)
events = pd.read_csv(catalogue_path)

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
for col in ["solexs", "hel1os"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

for col in ["start_utc", "peak_utc", "end_utc"]:
    events[col] = pd.to_datetime(events[col], utc=True, errors="coerce")

# Full observation-period overview
fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

for source, ax, label in [
    ("solexs", axes[0], "SoLEXS: 2–22 keV"),
    ("hel1os", axes[1], "HEL1OS CZT1: 20–40 keV"),
]:
    ax.plot(df["timestamp"], df[source], linewidth=0.7)
    source_events = events[events["source"] == source]

    for _, event in source_events.iterrows():
        ax.axvspan(event["start_utc"], event["end_utc"], alpha=0.18)
        ax.axvline(event["peak_utc"], linestyle="--", linewidth=0.8)

    ax.set_ylabel("Recorded value")
    ax.set_title(label)
    ax.grid(True, alpha=0.25)

axes[-1].set_xlabel("Time (UTC)")
fig.suptitle("Katyani — Candidate flare overview")
fig.tight_layout()
fig.savefig(out_dir / "nowcast_overview.png", dpi=150)
plt.close(fig)

# Individual close-up plot for every candidate
for index, event in events.iterrows():
    source = event["source"]
    start = event["start_utc"]
    peak = event["peak_utc"]
    end = event["end_utc"]

    if source not in ("solexs", "hel1os") or pd.isna(peak):
        continue

    padding = pd.Timedelta(minutes=5)
    window = df[
        (df["timestamp"] >= start - padding) &
        (df["timestamp"] <= end + padding)
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(window["timestamp"], window[source], marker=".", markersize=3)
    ax.axvline(start, linestyle=":", label="Candidate start")
    ax.axvline(peak, linestyle="--", label="Candidate peak")
    ax.axvline(end, linestyle=":", label="Candidate end")
    ax.axvspan(start, end, alpha=0.15)

    score = event.get("max_robust_score", float("nan"))
    samples = event.get("samples_over_threshold", float("nan"))

    ax.set_title(
        f"{source.upper()} candidate {index + 1}: "
        f"score={score:.2f}, samples={samples}"
    )
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Recorded value")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    stamp = peak.strftime("%Y%m%dT%H%M%SZ")
    filename = f"event_{index + 1:02d}_{source}_{stamp}.png"
    fig.savefig(out_dir / filename, dpi=150)
    plt.close(fig)

print("Plots saved in:", out_dir.resolve())
print("Candidates plotted:", len(events))
