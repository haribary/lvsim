"""Concatenate clips within each gt:severity:speaker group into single audio files."""
import csv
import os
from collections import defaultdict

import torch
import torchaudio

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA_CSV = os.path.join(BASE_DIR, "data", "metadata.csv")
TARGET_SR = 16000
CROSSFADE_MS = 50
CROSSFADE_SAMPLES = int(TARGET_SR * CROSSFADE_MS / 1000)

# Read metadata and group by (gt_idx, severity, speaker_id)
groups = defaultdict(list)
with open(METADATA_CSV) as f:
    for row in csv.DictReader(f):
        key = (int(row["gt_idx"]), row["severity"], int(row["speaker_id"]))
        groups[key].append(row["file_path"])

# Sort file paths within each group so sentence order is consistent
for key in groups:
    groups[key].sort()

print(f"Total groups: {len(groups)}")

severity_to_dir = {
    "none": os.path.join(BASE_DIR, "data", "concat_control"),
    "mild": os.path.join(BASE_DIR, "data", "concat_dysfluent", "mild"),
    "moderate": os.path.join(BASE_DIR, "data", "concat_dysfluent", "moderate"),
    "severe": os.path.join(BASE_DIR, "data", "concat_dysfluent", "severe"),
}

counts = defaultdict(int)

for (gt_idx, severity, speaker_id), files in sorted(groups.items()):
    # Load and resample all clips
    clips = []
    for fp in files:
        wav, sr = torchaudio.load(os.path.join(BASE_DIR, fp))
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        clips.append(wav.squeeze(0))

    # Concatenate with crossfade
    concat = clips[0]
    for clip in clips[1:]:
        n = min(CROSSFADE_SAMPLES, len(concat), len(clip))
        fade_out = torch.linspace(1, 0, n)
        fade_in = torch.linspace(0, 1, n)
        concat[-n:] = concat[-n:] * fade_out + clip[:n] * fade_in
        concat = torch.cat([concat, clip[n:]])

    # Save
    out_dir = severity_to_dir[severity]
    if severity == "none":
        fname = f"ctrl_gt{gt_idx}_spk{speaker_id}.wav"
    else:
        fname = f"dys_{severity}_gt{gt_idx}_spk{speaker_id}.wav"

    out_path = os.path.join(out_dir, fname)
    torchaudio.save(out_path, concat.unsqueeze(0), TARGET_SR)
    dur = concat.shape[0] / TARGET_SR
    counts[severity] += 1

    if counts[severity] <= 3 or (gt_idx % 20 == 0):
        print(f"  {severity:>10s}  gt={gt_idx:3d}  spk={speaker_id:3d}  {len(clips)} clips → {dur:.1f}s → {fname}")

print(f"\nDone!")
for sev in ["none", "mild", "moderate", "severe"]:
    print(f"  {sev:>10s}: {counts[sev]} files")
print(f"  Total: {sum(counts.values())} files")
