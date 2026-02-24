from google import genai
from dotenv import load_dotenv
import os
from src.phoneme import *
from src.prompts import get_prompts, get_ref_text

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

SEVERITY_LABELS = ["mild", "moderate", "severe"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "phonemes")
os.makedirs(OUT_DIR, exist_ok=True)

ref_texts = get_ref_text()


severity = 1
label = SEVERITY_LABELS[severity]
system_prompt_l1, system_prompt_l2 = get_prompts(severity)
ref_text = ref_texts[0]
prompt_idx = 0
tag = f"severity={label} prompt={prompt_idx}"
print(f"\n{'='*60}")
print(f"  {tag}")
print(f"{'='*60}")

# L1: word-level dysfluency
response_l1 = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[ref_text],
    config={"system_instruction": system_prompt_l1},
)
word_dys = response_l1.text.replace("...", "")
print(word_dys)

# save word-level output
word_path = os.path.join(OUT_DIR, f"word_{label}_p{prompt_idx}.txt")
with open(word_path, "w", encoding="utf-8") as f:
    f.write(word_dys)
print(f"  Wrote word-level  -> {word_path}")

# L2: phoneme-level dysfluency
user_l2 = f"""
DYS_REF_TEXT:
{word_dys}

IPA_CORRECT:
{phonemize_text(word_dys)}

"""
print('____PROMPT L2_____')
print(user_l2)

response_l2 = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[user_l2],
    config={"system_instruction": system_prompt_l2},
)
dysfluent_ipa = response_l2.text.strip()
print('___OUT___')
print(dysfluent_ipa)

# save phoneme-level output
ipa_path = os.path.join(OUT_DIR, f"phone_{label}_p{prompt_idx}.txt")
with open(ipa_path, "w", encoding="utf-8") as f:
    f.write(dysfluent_ipa + "\n")
print(f"  Wrote phoneme-level -> {ipa_path}")

print(f"\nDone. All outputs in {OUT_DIR}")
