"""Minimal VITS test: phoneme file in, wav file out."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vits"))

from synth_phonemes import (
    preprocess_ipa, split_into_chunks, load_model,
    synthesize_chunk, SILENCE_SECS,
)
import numpy as np
import torch
from scipy.io.wavfile import write as write_wav


def run_vits(phoneme_path: str, output_path: str, speaker_id: int = 0):
    """Read IPA phonemes from a text file and synthesize to a wav file."""
    raw = open(phoneme_path, encoding="utf-8").read()
    ipa = preprocess_ipa(raw)
    chunks = split_into_chunks(ipa)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, hps = load_model(device)
    sr = hps.data.sampling_rate
    silence = np.zeros(int(sr * SILENCE_SECS), dtype=np.float32)

    audio_parts = []
    for i, chunk in enumerate(chunks):
        audio = synthesize_chunk(model, hps, chunk, speaker_id, device)
        if audio.size > 0:
            audio_parts.append(audio)
            if i < len(chunks) - 1:
                audio_parts.append(silence)

    full = np.concatenate(audio_parts)
    write_wav(output_path, sr, (full * 32767).astype(np.int16))
    print(f"Saved {len(full)/sr:.1f}s -> {output_path}")


if __name__ == "__main__":
    phoneme_file = 
    out_file = "test_output.wav"
    run_vits(phoneme_file, out_file)
