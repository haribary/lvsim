"""Synthesize audio from a dysfluent IPA phoneme string using VITS (VCTK model).

Input format: space-separated IPA phones with inline dysfluency markers, e.g.:
    aɪ w oʊ k ʌ p ɜː [PRO] l ɪ ɚ | b ɪ k ʌ [DEL] z ...

Dysfluency markers handled:
    [PRO]  prolongation  — stripped (duration already encoded in vowel)
    [PAU]  block/pause   — replaced with … (VITS pause character)
    [DEL]  deletion      — stripped (phone was deleted, nothing to say)
    [INS]  insertion     — stripped (inserted phone already in stream)
    [REP]  repetition    — stripped (the preceding ... already signals repetition)
    ...                  — kept as … (pause / repetition ellipsis)
    |      sentence boundary — chunk split point with inter-sentence silence

Usage:
    python3 synth_phonemes.py phonemes.txt [output.wav] [--speaker N]
    python3 synth_phonemes.py -                          # read from stdin
"""

import os
import sys
import re
import random
import argparse

# Add the vits/ directory to sys.path so that bare imports used by vits
# internals (e.g. `import commons`, `import modules`) resolve correctly.
_VITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vits")
sys.path.insert(0, _VITS_DIR)

import numpy as np
import torch
from scipy.io.wavfile import write as write_wav

# monotonic_align requires a Cython extension that may not be built.
# It is only used during training, not inference, so mock it if missing.
import types
try:
    import monotonic_align  # noqa: F401
except (ImportError, ModuleNotFoundError):
    _ma = types.ModuleType("monotonic_align")
    _ma.__path__ = []
    sys.modules["monotonic_align"] = _ma

from models import SynthesizerTrn
import commons
import utils

# Load symbols directly from text/symbols.py, bypassing text/__init__.py
# which pulls in cleaners → unidecode, phonemizer (not needed for inference).
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("text.symbols", os.path.join(_VITS_DIR, "text", "symbols.py"))
_sym_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_sym_mod)
symbols = _sym_mod.symbols

_symbol_to_id = {s: i for i, s in enumerate(symbols)}

def text_to_sequence_phn(phonemes: str) -> list[int]:
    """Convert a cleaned IPA string to a sequence of symbol IDs."""
    return [_symbol_to_id[c] for c in phonemes if c in _symbol_to_id]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = os.path.join(_VITS_DIR, "pretrained_vctk.pth")
CONFIG_PATH     = os.path.join(_VITS_DIR, "configs", "vctk_base.json")

SILENCE_SECS        = 0.30   # silence between sentence-boundary chunks
PAUSE_SECS          = 0.15   # silence appended after an in-word [PAU]
MAX_CHUNK_CHARS     = 500    # safety upper limit on IPA chars per inference call

PAUSE_CHAR = "…"             # U+2026, in VITS punctuation symbol set

_VALID = set(symbols)

# ---------------------------------------------------------------------------
# Marker / token normalisation
# ---------------------------------------------------------------------------

# Dysfluency markers to remove outright
_STRIP_MARKERS = re.compile(r'\[(PRO|DEL|INS|REP)\]')

def preprocess_ipa(raw: str) -> str:
    """Strip dysfluency markers and normalise pause tokens in an IPA string.

    Rules:
      [PAU]  → … (VITS pause character)
      [PRO]  → (removed)
      [DEL]  → (removed)
      [INS]  → (removed)
      [REP]  → (removed)
      ...    → … (three ASCII dots → Unicode ellipsis)
    """
    # [PAU] → pause char
    s = raw.replace("[PAU]", PAUSE_CHAR)
    # strip all other markers
    s = _STRIP_MARKERS.sub("", s)
    # three ASCII dots → ellipsis
    s = s.replace("...", PAUSE_CHAR)
    # collapse runs of whitespace left by removed markers
    s = re.sub(r' +', ' ', s).strip()
    return s


def filter_to_valid(s: str) -> str:
    """Drop any character not in the VITS symbol table."""
    return "".join(c for c in s if c in _VALID)

# ---------------------------------------------------------------------------
# Chunk splitting  (split on | sentence boundaries)
# ---------------------------------------------------------------------------

def split_into_chunks(ipa: str) -> list[str]:
    """Split the full IPA string at '|' sentence boundaries.

    Each chunk is cleaned and size-capped at MAX_CHUNK_CHARS.
    """
    raw_chunks = ipa.split("|")
    chunks = []
    for raw in raw_chunks:
        chunk = filter_to_valid(raw.strip())
        if not chunk:
            continue
        # Hard cap: split overlong chunks at the nearest pause char
        while len(chunk) > MAX_CHUNK_CHARS:
            cut = chunk.rfind(PAUSE_CHAR, 0, MAX_CHUNK_CHARS)
            if cut == -1:
                cut = MAX_CHUNK_CHARS
            chunks.append(chunk[:cut].strip())
            chunk = chunk[cut:].lstrip(PAUSE_CHAR).strip()
        if chunk:
            chunks.append(chunk)
    return chunks

# ---------------------------------------------------------------------------
# VITS model
# ---------------------------------------------------------------------------

def load_model(device: torch.device):
    hps = utils.get_hparams_from_file(CONFIG_PATH)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)
    net_g.eval()
    utils.load_checkpoint(CHECKPOINT_PATH, net_g, None)
    return net_g, hps


def synthesize_chunk(model, hps, chunk: str, speaker_id: int, device: torch.device) -> np.ndarray:
    """Run VITS inference on one IPA chunk → float32 audio array."""
    add_blank = getattr(hps.data, "add_blank", True)

    ids = text_to_sequence_phn(chunk)
    if not ids:
        return np.array([], dtype=np.float32)

    if add_blank:
        ids = commons.intersperse(ids, 0)

    x = torch.LongTensor(ids).unsqueeze(0).to(device)
    x_lengths = torch.LongTensor([len(ids)]).to(device)
    sid = torch.LongTensor([speaker_id]).to(device)

    length_scale = random.uniform(1.2, 1.5)

    with torch.no_grad():
        audio = model.infer(
            x, x_lengths,
            sid=sid,
            noise_scale=0.6,
            noise_scale_w=0.6,
            length_scale=length_scale,
        )[0][0, 0].cpu().float().numpy()

    return audio

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Synthesize dysfluent IPA with VITS (VCTK)")
    parser.add_argument("input", help="Path to phonemes.txt, or '-' for stdin")
    parser.add_argument("output", nargs="?", default="phonemes.wav", help="Output WAV path")
    parser.add_argument("--speaker", type=int, default=None,
                        help="VCTK speaker ID 0–108 (default: random)")
    args = parser.parse_args()

    # ---- checkpoint check ----
    if not os.path.isfile(CHECKPOINT_PATH):
        print(f"ERROR: VCTK checkpoint not found at {CHECKPOINT_PATH}")
        sys.exit(1)

    # ---- read input ----
    if args.input == "-":
        raw = sys.stdin.read()
    else:
        if not os.path.isfile(args.input):
            print(f"ERROR: file not found: {args.input}")
            sys.exit(1)
        raw = open(args.input, encoding="utf-8").read()

    if not raw.strip():
        print("ERROR: no phoneme input found")
        sys.exit(1)

    # ---- preprocess markers ----
    ipa = preprocess_ipa(raw)

    # ---- split into sentence chunks ----
    chunks = split_into_chunks(ipa)
    if not chunks:
        print("ERROR: no valid IPA chunks after preprocessing")
        sys.exit(1)
    print(f"Chunks: {len(chunks)}")

    # ---- speaker ----
    speaker_id = args.speaker if args.speaker is not None else random.randint(0, 108)
    print(f"Speaker ID: {speaker_id}")

    # ---- device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- load model ----
    print("Loading VITS (VCTK) model...")
    model, hps = load_model(device)
    sr = hps.data.sampling_rate
    silence = np.zeros(int(sr * SILENCE_SECS), dtype=np.float32)

    # ---- synthesize ----
    all_audio = []
    for i, chunk in enumerate(chunks):
        preview = chunk[:60] + ("…" if len(chunk) > 60 else "")
        print(f"  [{i+1:02d}/{len(chunks)}] {preview}")

        audio = synthesize_chunk(model, hps, chunk, speaker_id, device)
        if audio.size == 0:
            print(f"           (empty — skipped)")
            continue

        all_audio.append(audio)
        if i < len(chunks) - 1:
            all_audio.append(silence)

    if not all_audio:
        print("ERROR: no audio produced")
        sys.exit(1)

    # ---- save ----
    full = np.concatenate(all_audio)
    write_wav(args.output, sr, (full * 32767).astype(np.int16))
    duration = len(full) / sr
    print(f"\nSaved {duration:.1f}s → {args.output}")


if __name__ == "__main__":
    main()
