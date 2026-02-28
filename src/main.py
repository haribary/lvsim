"""Generate speech dataset: both dysfluent (lvPPA) and control samples.

Reads ground-truth speeches from data/gt_seed/gt_{prompt_idx}.json (5 speeches
each). Each speech gets a globally unique gt_idx = prompt_idx * 5 + speech_offset.

Usage:
    python -m src.main --severity 1 --prompt-idx 0 --speaker 42
    python -m src.main --severity 0 --prompt-idx 2          # random speaker
    python -m src.main --severity 2 --prompt-idx 0 --dry-run  # skip synthesis

Outputs:
    data/dysfluent/<severity>/<wav files>
    data/control/<wav files>
    data/gt/gt_000.txt ... gt_NNN.txt
    data/metadata.csv
    data/phonemes/<intermediate text files>
"""
import logging
logging.basicConfig(level=logging.WARNING)
import os
import sys
import argparse
import random

import torch
from google import genai
from dotenv import load_dotenv

from src.prompts import load_gt_seed, SPEECHES_PER_PROMPT
from src.dysfluency import generate_dysfluent_ipa
from src.control import generate_control_ipa

from .synth_phonemes import (
    load_model, synthesize_sentences, save_wav,
    build_metadata_row, log_metadata,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEVERITY_LABELS = ["mild", "moderate", "severe"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHONEME_DIR = os.path.join(BASE_DIR, "data", "phonemes")
GT_DIR = os.path.join(BASE_DIR, "data", "gt")
METADATA_CSV = os.path.join(BASE_DIR, "data", "metadata.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate dysfluent + control speech dataset"
    )
    parser.add_argument("--severity", type=int, required=True, choices=[0, 1, 2],
                        help="Severity level: 0=mild, 1=moderate, 2=severe")
    parser.add_argument("--prompt-idx", type=int, required=True,
                        help="Index into gt_seed files (gt_0.json, gt_1.json, ...)")
    parser.add_argument("--speaker", type=int, default=None,
                        help="VCTK speaker ID 0-108 (default: random)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run LLM generation only, skip synthesis")
    args = parser.parse_args()

    severity = args.severity
    prompt_idx = args.prompt_idx
    severity_label = SEVERITY_LABELS[severity]
    speaker_id = args.speaker if args.speaker is not None else random.randint(0, 108)

    # ---- directories ----
    os.makedirs(PHONEME_DIR, exist_ok=True)
    os.makedirs(GT_DIR, exist_ok=True)
    dys_audio_dir = os.path.join(BASE_DIR, "data", "dysfluent", severity_label)
    ctrl_audio_dir = os.path.join(BASE_DIR, "data", "control")
    os.makedirs(dys_audio_dir, exist_ok=True)
    os.makedirs(ctrl_audio_dir, exist_ok=True)

    # ---- load speeches from gt_seed ----
    speeches = load_gt_seed(prompt_idx)

    print(f"{'='*60}")
    print(f"  severity={severity_label}  prompt_idx={prompt_idx}  speaker={speaker_id}")
    print(f"  {len(speeches)} speeches from gt_{prompt_idx}.json")
    print(f"{'='*60}")

    # ---- LLM client ----
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set")
        sys.exit(1)
    client = genai.Client(api_key=api_key)

    # ---- load VITS model (once, before the loop) ----
    model, hps, sr, device = None, None, None, None
    if not args.dry_run:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nLoading VITS model (device={device})...")
        model, hps = load_model(device)
        sr = hps.data.sampling_rate

    # ---- process each speech ----
    total_dys = 0
    total_ctrl = 0

    for speech_offset, ref_text in enumerate(speeches):
        gt_idx = prompt_idx * SPEECHES_PER_PROMPT + speech_offset

        print(f"\n{'─'*60}")
        print(f"  Speech {speech_offset + 1}/{len(speeches)}  gt_idx={gt_idx}")
        print(f"{'─'*60}")

        # ---- write ground truth file ----
        gt_path = os.path.join(GT_DIR, f"gt_{gt_idx:03d}.txt")
        with open(gt_path, "w", encoding="utf-8") as f:
            f.write(ref_text.strip())
        print(f"  Ground truth → {gt_path}")

        # ---- dysfluent generation ----
        print(f"\n  [dysfluent] Running L1 + L2...")
        dys_word_text, dys_ipa = generate_dysfluent_ipa(client, ref_text, severity)

        tag = f"{severity_label}_gt{gt_idx}"
        word_path = os.path.join(PHONEME_DIR, f"word_{tag}.txt")
        with open(word_path, "w", encoding="utf-8") as f:
            f.write(dys_word_text)

        ipa_path = os.path.join(PHONEME_DIR, f"phone_{tag}.txt")
        with open(ipa_path, "w", encoding="utf-8") as f:
            f.write(dys_ipa + "\n")

        # ---- control generation ----
        print(f"  [control] Running filler insertion...")
        ctrl_word_text, ctrl_ipa = generate_control_ipa(client, ref_text)

        ctrl_tag = f"control_gt{gt_idx}"
        word_path = os.path.join(PHONEME_DIR, f"word_{ctrl_tag}.txt")
        with open(word_path, "w", encoding="utf-8") as f:
            f.write(ctrl_word_text)

        ipa_path = os.path.join(PHONEME_DIR, f"phone_{ctrl_tag}.txt")
        with open(ipa_path, "w", encoding="utf-8") as f:
            f.write(ctrl_ipa + "\n")

        if args.dry_run:
            print(f"  [dry-run] Skipping synthesis.")
            print(f"  Dysfluent IPA: {dys_ipa[:80]}...")
            print(f"  Control IPA:   {ctrl_ipa[:80]}...")
            continue

        # ---- synthesize dysfluent ----
        print(f"  [dysfluent] Synthesizing...")
        dys_results = synthesize_sentences(model, hps, dys_ipa, speaker_id, device)

        for r in dys_results:
            fname = f"dys_{severity_label}_gt{gt_idx}_spk{speaker_id}_s{r['sentence_idx']:03d}.wav"
            wav_path = os.path.join(dys_audio_dir, fname)
            save_wav(r["audio"], wav_path, sr)

            row = build_metadata_row(
                file_path=os.path.relpath(wav_path, BASE_DIR),
                label="dysfluent",
                severity=severity_label,
                speaker_id=speaker_id,
                result=r,
                prompt_idx=prompt_idx,
                gt_idx=gt_idx,
            )
            log_metadata(METADATA_CSV, row)
            print(f"    [s{r['sentence_idx']:03d}] {r['duration_sec']:.1f}s → {fname}")

        total_dys += len(dys_results)

        # ---- synthesize control ----
        print(f"  [control] Synthesizing...")
        ctrl_results = synthesize_sentences(model, hps, ctrl_ipa, speaker_id, device)

        for r in ctrl_results:
            fname = f"ctrl_gt{gt_idx}_spk{speaker_id}_s{r['sentence_idx']:03d}.wav"
            wav_path = os.path.join(ctrl_audio_dir, fname)
            save_wav(r["audio"], wav_path, sr)

            row = build_metadata_row(
                file_path=os.path.relpath(wav_path, BASE_DIR),
                label="control",
                severity="none",
                speaker_id=speaker_id,
                result=r,
                prompt_idx=prompt_idx,
                gt_idx=gt_idx,
            )
            log_metadata(METADATA_CSV, row)
            print(f"    [s{r['sentence_idx']:03d}] {r['duration_sec']:.1f}s → {fname}")

        total_ctrl += len(ctrl_results)

    total = total_dys + total_ctrl
    print(f"\n{'='*60}")
    print(f"Done. {total} WAVs ({total_dys} dysfluent + {total_ctrl} control)")
    print(f"Metadata → {METADATA_CSV}")


if __name__ == "__main__":
    main()
