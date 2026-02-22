"""Synthesize audio from a dysfluent IPA phoneme string using VITS (VCTK model).

Input format: space-separated IPA phones with inline dysfluency markers, e.g.:
    aɪ w oʊ k ʌ p ɜː [PRO] l ɪ ɚ | b ɪ k ʌ [DEL] z ...

Dysfluency markers handled:
    [PRO]  prolongation  — extends the preceding phoneme's duration by 0.17-0.8s
                           inside the VITS duration matrix
    [PAU]  block/pause   — splits synthesis and inserts 0.3-1.5s silence
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

SILENCE_SECS        = 0.30           # silence between sentence-boundary chunks
PAU_SECS_RANGE      = (0.3, 1.5)    # silence range for [PAU] blocks
PRO_SECS_RANGE      = (0.17, 0.8)   # extra duration range for [PRO] prolongation
MAX_CHUNK_CHARS     = 500            # safety upper limit on IPA chars per inference call

PAUSE_CHAR = "…"             # U+2026, in VITS punctuation symbol set

_VALID = set(symbols)

# ---------------------------------------------------------------------------
# Marker / token normalisation
# ---------------------------------------------------------------------------

# Dysfluency markers to remove outright (DEL, INS, REP only).
# [PAU] and [PRO] are preserved for synthesis-time handling.
_STRIP_MARKERS = re.compile(r'\[(DEL|INS|REP)\]')

def preprocess_ipa(raw: str) -> str:
    """Normalise an IPA string, keeping [PAU] and [PRO] markers for synthesis.

    Rules:
      [PAU]  → kept (synthesis splits here and inserts silence)
      [PRO]  → kept (synthesis extends phoneme duration)
      [DEL]  → (removed)
      [INS]  → (removed)
      [REP]  → (removed)
      ...    → … (three ASCII dots → Unicode ellipsis)
    """
    # strip DEL, INS, REP markers
    s = _STRIP_MARKERS.sub("", raw)
    # three ASCII dots → ellipsis (for repetition constructs)
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

    [PAU] and [PRO] markers are preserved for synthesis-time handling.
    filter_to_valid is deferred until after marker parsing in synthesize_chunk.
    """
    raw_chunks = ipa.split("|")
    chunks = []
    for raw in raw_chunks:
        chunk = raw.strip()
        if not chunk:
            continue
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


# ---------------------------------------------------------------------------
# [PRO] marker parsing
# ---------------------------------------------------------------------------

def parse_pro_markers(chunk: str) -> tuple[str, list[int]]:
    """Extract [PRO] markers and return clean IPA with prolongation positions.

    Returns:
        clean: chunk with [PRO] stripped
        pro_char_indices: character indices (in clean) of the phonemes to prolong
    """
    pro_indices = []
    clean = ""
    i = 0
    while i < len(chunk):
        if chunk[i:i+5] == "[PRO]":
            # The phoneme to prolong is the last valid symbol char added
            for j in range(len(clean) - 1, -1, -1):
                if clean[j] in _symbol_to_id:
                    pro_indices.append(j)
                    break
            i += 5
        else:
            clean += chunk[i]
            i += 1
    return clean, pro_indices


def text_to_sequence_with_pro(phonemes: str, pro_char_indices: list[int]) -> tuple[list[int], list[int]]:
    """Convert IPA to symbol IDs, mapping prolongation char positions to ID positions.

    Characters not in the symbol table are skipped (same as filter_to_valid +
    text_to_sequence_phn in one pass), and the pro indices are remapped to
    account for any skipped characters.
    """
    pro_set = set(pro_char_indices)
    ids = []
    pro_id_indices = []
    for i, c in enumerate(phonemes):
        if c in _symbol_to_id:
            if i in pro_set:
                pro_id_indices.append(len(ids))
            ids.append(_symbol_to_id[c])
    return ids, pro_id_indices

# ---------------------------------------------------------------------------
# Synthesis helpers
# ---------------------------------------------------------------------------

def _synthesize_ids(model, hps, ids: list[int], speaker_id: int,
                    device: torch.device, length_scale: float) -> np.ndarray:
    """Standard VITS inference from pre-computed symbol IDs (no prolongation)."""
    add_blank = getattr(hps.data, "add_blank", True)
    if add_blank:
        ids = commons.intersperse(ids, 0)

    x = torch.LongTensor(ids).unsqueeze(0).to(device)
    x_lengths = torch.LongTensor([len(ids)]).to(device)
    sid = torch.LongTensor([speaker_id]).to(device)

    with torch.no_grad():
        audio = model.infer(
            x, x_lengths, sid=sid,
            noise_scale=0.6, noise_scale_w=0.6, length_scale=length_scale,
        )[0][0, 0].cpu().float().numpy()
    return audio


def _synthesize_with_prolongation(model, hps, ids: list[int],
                                  pro_id_indices: list[int],
                                  speaker_id: int, device: torch.device,
                                  length_scale: float) -> np.ndarray:
    """VITS inference with duration extension at prolongation positions.

    Inlines the model.infer() logic so we can modify w_ceil (the per-phoneme
    duration in mel frames) before the decoder runs.
    """
    add_blank = getattr(hps.data, "add_blank", True)
    sr = hps.data.sampling_rate
    hop_length = hps.data.hop_length

    if add_blank:
        pro_interspersed = [2 * idx + 1 for idx in pro_id_indices]
        ids = commons.intersperse(ids, 0)
    else:
        pro_interspersed = list(pro_id_indices)

    x = torch.LongTensor(ids).unsqueeze(0).to(device)
    x_lengths = torch.LongTensor([len(ids)]).to(device)
    sid = torch.LongTensor([speaker_id]).to(device)

    with torch.no_grad():
        # --- text encoder ---
        x_enc, m_p, logs_p, x_mask = model.enc_p(x, x_lengths)

        # --- speaker embedding ---
        g = model.emb_g(sid).unsqueeze(-1) if model.n_speakers > 0 else None

        # --- duration prediction ---
        if model.use_sdp:
            logw = model.dp(x_enc, x_mask, g=g, reverse=True, noise_scale=0.6)
        else:
            logw = model.dp(x_enc, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)

        # --- PROLONGATION: extend duration at marked positions ---
        for idx in pro_interspersed:
            if idx < w_ceil.shape[2]:
                extra_secs = random.uniform(*PRO_SECS_RANGE)
                extra_frames = int(extra_secs * sr / hop_length)
                w_ceil[0, 0, idx] += extra_frames

        # --- alignment + decoder (rest of model.infer) ---
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(
            commons.sequence_mask(y_lengths, None), 1
        ).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * 0.6
        z = model.flow(z_p, y_mask, g=g, reverse=True)
        o = model.dec((z * y_mask), g=g)
        audio = o[0, 0].cpu().float().numpy()

    return audio

# ---------------------------------------------------------------------------
# Top-level chunk synthesis (handles [PAU] + [PRO])
# ---------------------------------------------------------------------------

def synthesize_chunk(model, hps, chunk: str, speaker_id: int,
                     device: torch.device) -> np.ndarray:
    """Synthesize one sentence chunk, handling [PAU] and [PRO] markers.

    [PAU]  — splits synthesis here and inserts 0.3-1.5s silence.
    [PRO]  — extends the preceding phoneme's duration inside VITS.
    """
    sr = hps.data.sampling_rate
    length_scale = random.uniform(1.2, 1.5)  # consistent within the chunk

    # Split on [PAU] → sub-chunks with silence between them
    sub_chunks = re.split(r'\s*\[PAU\]\s*', chunk)

    audio_parts = []
    for sc in sub_chunks:
        sc = sc.strip()
        if not sc:
            continue

        # Parse [PRO] markers → clean IPA + prolongation positions
        clean_ipa, pro_char_indices = parse_pro_markers(sc)

        # Convert to IDs (also filters invalid chars)
        ids, pro_id_indices = text_to_sequence_with_pro(clean_ipa, pro_char_indices)
        if not ids:
            continue

        # Synthesize — use prolongation path only when needed
        if pro_id_indices:
            audio = _synthesize_with_prolongation(
                model, hps, ids, pro_id_indices, speaker_id, device, length_scale)
        else:
            audio = _synthesize_ids(
                model, hps, ids, speaker_id, device, length_scale)

        if audio.size > 0:
            audio_parts.append(audio)

    if not audio_parts:
        return np.array([], dtype=np.float32)

    # Concatenate sub-chunks with [PAU] silence in between
    result = []
    for i, part in enumerate(audio_parts):
        result.append(part)
        if i < len(audio_parts) - 1:
            pause_secs = random.uniform(*PAU_SECS_RANGE)
            result.append(np.zeros(int(sr * pause_secs), dtype=np.float32))

    return np.concatenate(result)

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
