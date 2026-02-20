import os
import sys

import numpy as np
import torch
from scipy.io.wavfile import write as write_wav

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "VITS"))

from models import SynthesizerTrn
from text import cleaned_text_to_sequence
from text.symbols import symbols
import commons
import utils

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "pretrained_ljs.pth")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "ljs_base.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_audio")
INPUT_FILE = os.path.join(os.path.dirname(__file__), "output.txt")
SILENCE_DURATION = 0.3

def text_to_ids(vits_string, add_blank=True):
    ids = cleaned_text_to_sequence(vits_string)
    if add_blank:
        ids = commons.intersperse(ids, 0)
    return torch.LongTensor(ids)

def load_model(config_path, checkpoint_path):
    hps = utils.get_hparams_from_file(config_path)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    )
    net_g.eval()
    utils.load_checkpoint(checkpoint_path, net_g, None)
    return net_g, hps

def synthesize(model, hps, vits_string):
    ids = text_to_ids(vits_string, add_blank=getattr(hps.data, "add_blank", True))
    x = ids.unsqueeze(0)
    x_lengths = torch.LongTensor([ids.size(0)])

    with torch.no_grad():
        audio = model.infer(
            x, x_lengths,
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1.00,
        )[0][0, 0].cpu().float().numpy()

    return audio

def main():
    if not os.path.isfile(CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    raw = open(INPUT_FILE, encoding='utf-8').read().strip()
    sentences = [s.strip() for s in raw.split('|') if s.strip()]
    print(f"Found {len(sentences)} sentences")

    print("Loading VITS model...")
    model, hps = load_model(CONFIG_PATH, CHECKPOINT_PATH)
    sr = hps.data.sampling_rate

    silence = np.zeros(int(sr * SILENCE_DURATION), dtype=np.float32)
    all_audio = []

    for i, sentence in enumerate(sentences):
        print(f"  [{i+1:02d}] {sentence[:60]}...")

        audio = synthesize(model, hps, sentence)

        out_path = os.path.join(OUTPUT_DIR, f"sentence_{i+1:02d}.wav")
        write_wav(out_path, sr, (audio * 32767).astype(np.int16))
        print(f"       → {out_path}")

        all_audio.append(audio)
        all_audio.append(silence)

    if all_audio:
        full = np.concatenate(all_audio)
        full_path = os.path.join(OUTPUT_DIR, "full3.wav")
        write_wav(full_path, sr, (full * 32767).astype(np.int16))
        print(f"\nFull audio → {full_path}")

if __name__ == "__main__":
    main()