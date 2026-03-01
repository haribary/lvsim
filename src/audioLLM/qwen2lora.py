import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import re, random, glob, copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from scipy.signal import fftconvolve
from collections import Counter
from tqdm.auto import tqdm

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR = 16000
WINDOW_SEC = 15
WINDOW_SAMPLES = WINDOW_SEC * TARGET_SR
SAMPLES_PER_STREAM = 3

print(f"Device: {DEVICE}")

# Build dataframe from concat streams (moderate + severe only)
records = []

for f in sorted(glob.glob(os.path.join(BASE_DIR, "data", "concat_control", "*.wav"))):
    m = re.match(r"ctrl_gt(\d+)_spk(\d+)\.wav", os.path.basename(f))
    records.append({
        "abs_path": f, "label": "healthy", "label_id": 0,
        "severity": "none", "gt_idx": int(m.group(1)), "speaker_id": int(m.group(2)),
    })

for sev in ["moderate", "severe"]:
    for f in sorted(glob.glob(os.path.join(BASE_DIR, "data", "concat_dysfluent", sev, "*.wav"))):
        m = re.match(r"dys_\w+_gt(\d+)_spk(\d+)\.wav", os.path.basename(f))
        records.append({
            "abs_path": f, "label": "dysfluent", "label_id": 1,
            "severity": sev, "gt_idx": int(m.group(1)), "speaker_id": int(m.group(2)),
        })

df = pd.DataFrame(records)
print(f"Total streams: {len(df)}")
print(df["severity"].value_counts().to_string())

train_full_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label_id"])
train_df, val_df = train_test_split(train_full_df, test_size=0.15, random_state=SEED, stratify=train_full_df["label_id"])
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"\nTrain: {len(train_df)} streams ({train_df['label_id'].mean():.1%} dysfluent)")
print(f"Val:   {len(val_df)} streams ({val_df['label_id'].mean():.1%} dysfluent)")
print(f"Test:  {len(test_df)} streams ({test_df['label_id'].mean():.1%} dysfluent)")

SYSTEM_PROMPT = "You are an expert speech-language pathologist."

CONTROL_DESC = (
    "CONTROL — Speech is fluent with no pathological features. "
    "Normal hesitations, filler words, self-corrections, and brief pauses "
    "may be present but are typical of healthy speech.\n"
    "    Example: aɪ səpˈoʊz ðə tɹˈɪp ðæt stˈeɪz wɪð mˌiː mˈoʊst wʌz, "
    "wˈɛl, ɐ kwˈaɪət sˈʌmɚ wiː spˈɛnt ˌʌp æt lˈeɪk plˈæsɪd"
)
LVPPA_DESC = (
    "lvPPA — Speech shows features of logopenic variant primary progressive aphasia, "
    "including abnormally prolonged pauses during word retrieval, "
    "phonological errors (sound-level distortions, substitutions, or false starts), "
    "repetitive attempts at words, and fragmented sentence production.\n"
    "    Example: aɪ səpˈoʊ[PROLONG]z ðə, ðə, ðə θˈɪŋ wiː dˈɪd, "
    "ðə tɹˈɪp ðæt stˈeɪz wɪð mˌiː mˈoʊst, ɪt wʌzɐ kwˈaɪ...kwˈaɪət"
)


def get_prompt(counterbalance: bool = None):
    if counterbalance is None:
        counterbalance = random.random() < 0.5

    if counterbalance:
        a_desc, b_desc = CONTROL_DESC, LVPPA_DESC
        a_label, b_label = "CONTROL", "lvPPA"
    else:
        a_desc, b_desc = LVPPA_DESC, CONTROL_DESC
        a_label, b_label = "lvPPA", "CONTROL"

    prompt = (
        "Listen carefully to this audio of a person speaking.\n\n"
        "Which best describes this speaker?\n"
        f"A. {a_desc}\n\n"
        f"B. {b_desc}\n\n"
        "Respond with only the letter: A or B."
    )

    option_map = {"A": a_label, "B": b_label}
    return prompt, option_map


def augment_waveform(wav):
    if random.random() < 0.5:
        factor = random.uniform(0.9, 1.1)
        new_len = int(len(wav) / factor)
        wav = F.interpolate(wav.view(1, 1, -1), size=new_len, mode="linear", align_corners=False).squeeze()
    if random.random() < 0.5:
        snr_db = random.uniform(10, 20)
        sig_pow = wav.pow(2).mean()
        noise = torch.randn_like(wav) * (sig_pow / (10 ** (snr_db / 10))).sqrt()
        wav = wav + noise
    if random.random() < 0.5:
        wav = wav * (10 ** (random.uniform(-6, 6) / 20))
    if random.random() < 0.3:
        rt60 = random.uniform(0.2, 0.8)
        n = int(TARGET_SR * rt60)
        t = torch.arange(n, dtype=torch.float32) / TARGET_SR
        rir = torch.randn(n) * torch.exp(-3.0 * t / rt60)
        rir = rir / rir.abs().max()
        out = fftconvolve(wav.numpy(), rir.numpy(), mode="full")[:len(wav)]
        wav = torch.from_numpy(out.astype(np.float32))
    return wav


class QwenConcatDataset(Dataset):
    def __init__(self, dataframe, processor, samples_per_stream=3, augment=False):
        self.processor = processor
        self.samples_per_stream = samples_per_stream
        self.augment = augment

        self.audio = []
        self.labels = []
        self.label_ids = []
        for _, row in dataframe.iterrows():
            wav, sr = torchaudio.load(row["abs_path"])
            if sr != TARGET_SR:
                wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
            self.audio.append(wav.squeeze(0))
            self.labels.append(row["label_id"])
            self.label_ids.append(row["label_id"])
        print(f"  Loaded {len(self.audio)} streams, {len(self)} samples")

    def __len__(self):
        return len(self.audio) * self.samples_per_stream

    def __getitem__(self, idx):
        stream_idx = idx // self.samples_per_stream
        wav = self.audio[stream_idx]
        label_id = self.label_ids[stream_idx]

        # Random 15s crop
        if len(wav) <= WINDOW_SAMPLES:
            chunk = F.pad(wav, (0, WINDOW_SAMPLES - len(wav)))
        else:
            start = random.randint(0, len(wav) - WINDOW_SAMPLES)
            chunk = wav[start:start + WINDOW_SAMPLES]

        if self.augment:
            chunk = augment_waveform(chunk)

        if len(chunk) > WINDOW_SAMPLES:
            chunk = chunk[:WINDOW_SAMPLES]
        elif len(chunk) < WINDOW_SAMPLES:
            chunk = F.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)))

        audio_np = chunk.numpy()

        # Build prompt with randomized A/B order
        prompt, option_map = get_prompt()

        # Determine correct answer letter based on label
        if label_id == 1:  # dysfluent / lvPPA
            answer_letter = "A" if option_map["A"] == "lvPPA" else "B"
        else:  # control / healthy
            answer_letter = "A" if option_map["A"] == "CONTROL" else "B"

        # Build prompt-only conversation (for label masking)
        conv_prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "file"},
                {"type": "text", "text": prompt},
            ]},
        ]
        # Full conversation with response
        conv_full = conv_prompt + [{"role": "assistant", "content": answer_letter}]

        prompt_text = self.processor.apply_chat_template(
            conv_prompt, add_generation_prompt=True, tokenize=False
        )
        full_text = self.processor.apply_chat_template(
            conv_full, add_generation_prompt=False, tokenize=False
        )

        # Get prompt token count for label masking
        prompt_ids = self.processor.tokenizer(prompt_text, add_special_tokens=False).input_ids
        prompt_len = len(prompt_ids)

        # Process full conversation + audio
        inputs = self.processor(text=full_text, audio=[audio_np], return_tensors="pt")

        input_ids = inputs.input_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        input_features = inputs.input_features.squeeze(0)
        feature_attention_mask = inputs.feature_attention_mask.squeeze(0)

        # Labels: mask prompt tokens with -100
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,
            "feature_attention_mask": feature_attention_mask,
            "labels": labels,
        }


def collate_fn(batch):
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id

    max_len = max(b["input_ids"].size(0) for b in batch)

    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, b in enumerate(batch):
        seq_len = b["input_ids"].size(0)
        input_ids[i, :seq_len] = b["input_ids"]
        attention_mask[i, :seq_len] = b["attention_mask"]
        labels[i, :seq_len] = b["labels"]

    input_features = torch.stack([b["input_features"] for b in batch])
    feature_attention_mask = torch.stack([b["feature_attention_mask"] for b in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "input_features": input_features,
        "feature_attention_mask": feature_attention_mask,
        "labels": labels,
    }


print("Dataset and collate ready.")

# Load model + LoRA on decoder only
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Freeze audio encoder + projector
for param in model.audio_tower.parameters():
    param.requires_grad = False
for param in model.multi_modal_projector.parameters():
    param.requires_grad = False

# LoRA on decoder attention
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Enable gradient checkpointing for memory
model.enable_input_require_grads()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

print("Model ready.")

BATCH_SIZE = 1
GRAD_ACCUM = 8
EPOCHS = 10

print("Loading train set...")
train_ds = QwenConcatDataset(train_df, processor, samples_per_stream=SAMPLES_PER_STREAM, augment=True)
print("Loading val set...")
val_ds = QwenConcatDataset(val_df, processor, samples_per_stream=SAMPLES_PER_STREAM, augment=False)

# WeightedRandomSampler for class balance
train_snippet_labels = np.array([train_ds.labels[i // SAMPLES_PER_STREAM] for i in range(len(train_ds))])
class_counts = np.bincount(train_snippet_labels)
sample_weights = (1.0 / class_counts)[train_snippet_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_snippet_labels), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=2e-4, weight_decay=0.01,
)

num_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
num_warmup = int(0.1 * num_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup, num_steps)

print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
print(f"Steps: {num_steps}, Warmup: {num_warmup}")

@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        # Count non-masked label tokens
        n_tokens = (batch["labels"] != -100).sum().item()
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens
    return total_loss / max(total_tokens, 1)

best_val_loss = float("inf")
best_epoch = 0
best_state = None
history = []

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    total_loss, n_batches = 0.0, 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:2d}/{EPOCHS}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / GRAD_ACCUM
        loss.backward()

        if (batch_idx + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += outputs.loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

    # Flush remaining gradients
    if (batch_idx + 1) % GRAD_ACCUM != 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_train_loss = total_loss / n_batches
    val_loss = evaluate(val_loader)
    history.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": val_loss})

    marker = ""
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items() if v.requires_grad or "lora" in k})
        marker = "  ** best **"

    print(
        f"Epoch {epoch+1:2d}/{EPOCHS}  train_loss={avg_train_loss:.4f}  "
        f"val_loss={val_loss:.4f}{marker}"
    )

print(f"\nBest val loss: {best_val_loss:.4f} at epoch {best_epoch}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MISC_DIR = os.path.join(BASE_DIR, "data", "misc")
os.makedirs(MISC_DIR, exist_ok=True)

h = pd.DataFrame(history)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(h["epoch"], h["train_loss"], "o-", label="Train")
ax.plot(h["epoch"], h["val_loss"], "s-", label="Val")
ax.axvline(best_epoch, color="red", ls=":", alpha=0.7, label=f"Best (ep {best_epoch})")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Qwen2-Audio LoRA Training")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(MISC_DIR, "qwen2_lora_training_curves.png"), dpi=150)
plt.close(fig)
print(f"Saved to {MISC_DIR}/qwen2_lora_training_curves.png")

# Save LoRA adapters
save_dir = os.path.join(BASE_DIR, "data", "models", "qwen2_audio_lora")
model.save_pretrained(save_dir)
processor.save_pretrained(save_dir)
print(f"LoRA adapters saved to: {save_dir}")

# Test on real clinical data
MAX_SAMPLES = 30 * TARGET_SR

def classify_audio(audio_path):
    wav, sr = torchaudio.load(audio_path)
    wav = torchaudio.functional.resample(wav, sr, TARGET_SR).mean(0)
    if wav.shape[0] > MAX_SAMPLES:
        wav = wav[:MAX_SAMPLES]
    audio_np = wav.numpy()

    prompt, option_map = get_prompt()

    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "audio", "audio_url": audio_path},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=[audio_np], sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=10)
    ids = ids[:, inputs["input_ids"].size(1):]
    response = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    return response, option_map


def parse_response(response, option_map):
    r = response.strip().upper()
    if r.startswith("A"):
        return 1 if option_map["A"] == "lvPPA" else 0
    if r.startswith("B"):
        return 1 if option_map["B"] == "lvPPA" else 0
    # Fallback: scan for keywords
    r_lower = response.lower()
    if "lvppa" in r_lower or "dysfluent" in r_lower:
        return 1
    if any(w in r_lower for w in ["control", "healthy", "normal", "fluent"]):
        return 0
    return -1


model.eval()

REAL_DIR = os.path.join(BASE_DIR, "data", "real")
groups = {
    "lvPPA":     (sorted(glob.glob(os.path.join(REAL_DIR, "*lvPPA*", "**", "*.wav"), recursive=True)), 1),
    "JHU":       (sorted(glob.glob(os.path.join(REAL_DIR, "*jhu*", "**", "*.wav"), recursive=True)), 1),
    "Control":   (sorted(glob.glob(os.path.join(REAL_DIR, "*segmentedcc*", "**", "*.wav"), recursive=True)), 0),
    "Capilouto": (sorted(glob.glob(os.path.join(REAL_DIR, "*Capilouto*", "**", "*.wav"), recursive=True)), 0),
}

all_true, all_pred, all_raw = [], [], []

for name, (files, true_label) in groups.items():
    preds, raw_responses = [], []
    for f in tqdm(files, desc=name):
        resp, option_map = classify_audio(f)
        raw_responses.append(resp)
        pred = parse_response(resp, option_map)
        if pred == -1:
            pred = 0
        preds.append(pred)

    correct = sum(1 for p in preds if p == true_label)
    total = len(preds)
    acc = correct / total if total > 0 else 0
    dys_count = sum(1 for p in preds if p == 1)

    all_true.extend([true_label] * total)
    all_pred.extend(preds)
    all_raw.extend(raw_responses)

    print(
        f"{name:>10s}  ({total:3d} clips)  acc={acc:.3f}  "
        f"dys={dys_count}  healthy={total - dys_count}"
    )

all_true = np.array(all_true)
all_pred = np.array(all_pred)

print("=" * 60)
print("Overall Classification Report (LoRA Fine-Tuned Qwen2-Audio)")
print("=" * 60)
print(classification_report(all_true, all_pred, target_names=["healthy", "dysfluent"]))
print(f"F1 Macro: {f1_score(all_true, all_pred, average='macro'):.4f}")
print(f"Accuracy: {(all_true == all_pred).mean():.4f}")

print("\nRaw response distribution:")
for resp, count in Counter(all_raw).most_common(15):
    print(f"  '{resp}': {count}")

    