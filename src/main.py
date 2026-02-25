"""Generate speech dataset: both dysfluent (lvPPA) and control samples.

Usage:
    python main.py --severity 1 --prompt-idx 0 --speaker 42
    python main.py --severity 0 --prompt-idx 2          # random speaker
    python main.py --severity 2 --prompt-idx 0 --dry-run  # skip synthesis

Outputs:
    data/dysfluent/<severity>/<wav files>
    data/control/<wav files>
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

from src.prompts import get_ref_text
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
                        help="Index into ref_texts list")
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
    dys_audio_dir = os.path.join(BASE_DIR, "data", "dysfluent", severity_label)
    ctrl_audio_dir = os.path.join(BASE_DIR, "data", "control")
    os.makedirs(dys_audio_dir, exist_ok=True)
    os.makedirs(ctrl_audio_dir, exist_ok=True)

    # ---- ref text ----
    ref_texts = get_ref_text()
    if prompt_idx >= len(ref_texts):
        print(f"ERROR: prompt_idx {prompt_idx} out of range (have {len(ref_texts)} texts)")
        sys.exit(1)
    ref_text = ref_texts[prompt_idx]

    print(f"{'='*60}")
    print(f"  severity={severity_label}  prompt_idx={prompt_idx}  speaker={speaker_id}")
    print(f"{'='*60}")

    # ---- LLM generation ----
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set")
        sys.exit(1)
    client = genai.Client(api_key=api_key)

    # Dysfluent: L1 + L2
    print("\n[dysfluent] Running L1 + L2 dysfluency generation...")
    dys_word_text, dys_ipa = generate_dysfluent_ipa(client, ref_text, severity)

    word_path = os.path.join(PHONEME_DIR, f"word_{severity_label}_p{prompt_idx}.txt")
    with open(word_path, "w", encoding="utf-8") as f:
        f.write(dys_word_text)
    print(f"  Word-level → {word_path}")

    ipa_path = os.path.join(PHONEME_DIR, f"phone_{severity_label}_p{prompt_idx}.txt")
    with open(ipa_path, "w", encoding="utf-8") as f:
        f.write(dys_ipa + "\n")
    print(f"  IPA        → {ipa_path}")

    # Control: single-layer filler insertion
    print("\n[control] Running filler insertion...")
    ctrl_word_text, ctrl_ipa = generate_control_ipa(client, ref_text)

    word_path = os.path.join(PHONEME_DIR, f"word_control_p{prompt_idx}.txt")
    with open(word_path, "w", encoding="utf-8") as f:
        f.write(ctrl_word_text)
    print(f"  Word-level → {word_path}")

    ipa_path = os.path.join(PHONEME_DIR, f"phone_control_p{prompt_idx}.txt")
    with open(ipa_path, "w", encoding="utf-8") as f:
        f.write(ctrl_ipa + "\n")
    print(f"  IPA        → {ipa_path}")

    if args.dry_run:
        print("\n[dry-run] Skipping synthesis.")
        print(f"Dysfluent IPA preview: {dys_ipa[:120]}...")
        print(f"Control IPA preview:   {ctrl_ipa[:120]}...")
        return

    # ---- synthesis ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading VITS model (device={device})...")
    model, hps = load_model(device)
    sr = hps.data.sampling_rate

    # Synthesize dysfluent
    print("\n[dysfluent] Synthesizing per-sentence audio...")
    dys_results = synthesize_sentences(model, hps, dys_ipa, speaker_id, device)
    print(f"  {len(dys_results)} sentences produced")

    for r in dys_results:
        fname = f"dys_{severity_label}_p{prompt_idx}_spk{speaker_id}_s{r['sentence_idx']:03d}.wav"
        wav_path = os.path.join(dys_audio_dir, fname)
        save_wav(r["audio"], wav_path, sr)

        row = build_metadata_row(
            file_path=os.path.relpath(wav_path, BASE_DIR),
            label="dysfluent",
            severity=severity_label,
            speaker_id=speaker_id,
            result=r,
            prompt_idx=prompt_idx,
            ground_truth_text=ref_text,
        )
        log_metadata(METADATA_CSV, row)
        print(f"  [s{r['sentence_idx']:03d}] {r['duration_sec']:.1f}s → {fname}")

    # Synthesize control
    print("\n[control] Synthesizing per-sentence audio...")
    ctrl_results = synthesize_sentences(model, hps, ctrl_ipa, speaker_id, device)
    print(f"  {len(ctrl_results)} sentences produced")

    for r in ctrl_results:
        fname = f"ctrl_p{prompt_idx}_spk{speaker_id}_s{r['sentence_idx']:03d}.wav"
        wav_path = os.path.join(ctrl_audio_dir, fname)
        save_wav(r["audio"], wav_path, sr)

        row = build_metadata_row(
            file_path=os.path.relpath(wav_path, BASE_DIR),
            label="control",
            severity="none",
            speaker_id=speaker_id,
            result=r,
            prompt_idx=prompt_idx,
            ground_truth_text=ref_text,
        )
        log_metadata(METADATA_CSV, row)
        print(f"  [s{r['sentence_idx']:03d}] {r['duration_sec']:.1f}s → {fname}")

    total = len(dys_results) + len(ctrl_results)
    print(f"\nDone. {total} WAVs ({len(dys_results)} dysfluent + {len(ctrl_results)} control)")
    print(f"Metadata → {METADATA_CSV}")


if __name__ == "__main__":
    main()
