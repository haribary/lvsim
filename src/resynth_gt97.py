"""Re-synthesize gt97 moderate and severe from cleaned phoneme files."""
import os
import torch

from src.synth_phonemes import (
    load_model, synthesize_sentences, save_wav,
    build_metadata_row, log_metadata,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHONEME_DIR = os.path.join(BASE_DIR, "data", "phonemes")
METADATA_CSV = os.path.join(BASE_DIR, "data", "metadata.csv")

GT_IDX = 97
PROMPT_IDX = 19  # gt_idx 97 = prompt 19, speech 2 (97 = 19*5 + 2)
SPEAKER_ID = 42

SEVERITY_LABELS = ["moderate", "severe"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading VITS model (device={device})...")
model, hps = load_model(device)
sr = hps.data.sampling_rate

for severity_label in SEVERITY_LABELS:
    phone_path = os.path.join(PHONEME_DIR, f"phone_{severity_label}_gt{GT_IDX}.txt")
    with open(phone_path, encoding="utf-8") as f:
        ipa = f.read().strip()

    print(f"\n[{severity_label}] Synthesizing from cleaned phonemes...")
    print(f"  IPA length: {len(ipa)} chars")

    audio_dir = os.path.join(BASE_DIR, "data", "dysfluent", severity_label)
    os.makedirs(audio_dir, exist_ok=True)

    results = synthesize_sentences(model, hps, ipa, SPEAKER_ID, device)
    print(f"  Got {len(results)} sentences")

    for r in results:
        fname = f"dys_{severity_label}_gt{GT_IDX}_spk{SPEAKER_ID}_s{r['sentence_idx']:03d}.wav"
        wav_path = os.path.join(audio_dir, fname)
        save_wav(r["audio"], wav_path, sr)
        row = build_metadata_row(
            file_path=os.path.relpath(wav_path, BASE_DIR),
            label="dysfluent",
            severity=severity_label,
            speaker_id=SPEAKER_ID,
            result=r,
            prompt_idx=PROMPT_IDX,
            gt_idx=GT_IDX,
        )
        log_metadata(METADATA_CSV, row)
        print(f"  [s{r['sentence_idx']:03d}] {r['duration_sec']:.1f}s → {fname}")

print("\nDone!")
