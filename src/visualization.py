"""Visualization utilities for the lvsim dataset."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA_CSV = os.path.join(BASE_DIR, "data", "metadata.csv")
MISC_DIR = os.path.join(BASE_DIR, "data", "misc")


def plot_duration_distribution(csv_path: str = METADATA_CSV):
    """Plot histogram of audio durations, split by label. Saves to data/misc/."""
    os.makedirs(MISC_DIR, exist_ok=True)
    df = pd.read_csv(csv_path)

    labels = df["label"].unique()
    fig, ax = plt.subplots(figsize=(8, 4))

    for label in sorted(labels):
        subset = df[df["label"] == label]["duration_sec"]
        ax.hist(subset, bins=30, alpha=0.6, label=f"{label} (n={len(subset)})")

    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Count")
    ax.set_title("Audio Duration Distribution")
    ax.legend()
    ax.axvline(5.0, color="red", linestyle="--", linewidth=0.8)
    fig.tight_layout()

    out_path = os.path.join(MISC_DIR, "duration_distribution.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_clips_per_gt(csv_path: str = METADATA_CSV, threshold: int = 7):
    """Bar chart of clip count per (gt_idx, severity), highlighting outliers.

    Saves to data/misc/clips_per_gt.png and prints outlier groups to stdout.
    """
    os.makedirs(MISC_DIR, exist_ok=True)
    df = pd.read_csv(csv_path)

    counts = df.groupby(["gt_idx", "severity"]).size().reset_index(name="clip_count")

    severity_order = ["none", "mild", "moderate", "severe"]
    colors = {"none": "#4c72b0", "mild": "#55a868", "moderate": "#c44e52", "severe": "#8172b2"}

    fig, ax = plt.subplots(figsize=(16, 5))

    for sev in severity_order:
        subset = counts[counts["severity"] == sev]
        if subset.empty:
            continue
        ax.bar(
            subset["gt_idx"] + severity_order.index(sev) * 0.2 - 0.3,
            subset["clip_count"],
            width=0.18,
            label=sev,
            color=colors[sev],
            alpha=0.8,
        )

    ax.axhline(threshold, color="red", linestyle="--", linewidth=1, label=f"threshold ({threshold})")
    ax.set_xlabel("gt_idx")
    ax.set_ylabel("Clip count")
    ax.set_title("Clips per (gt_idx, severity) — outliers above red line")
    ax.legend(loc="upper right")
    ax.set_xticks(range(int(counts["gt_idx"].max()) + 1))
    fig.tight_layout()

    out_path = os.path.join(MISC_DIR, "clips_per_gt.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")

    # Print outliers
    outliers = counts[counts["clip_count"] > threshold].sort_values(
        ["gt_idx", "severity"]
    )
    if outliers.empty:
        print(f"No groups exceed {threshold} clips.")
    else:
        print(f"\nOutliers (>{threshold} clips):")
        for _, row in outliers.iterrows():
            print(f"  gt_{int(row['gt_idx']):03d}  {row['severity']:>10s}  → {int(row['clip_count'])} clips")


if __name__ == "__main__":
    plot_duration_distribution()
    plot_clips_per_gt()
