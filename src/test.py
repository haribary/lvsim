# import torch
from google import genai
from dotenv import load_dotenv
import os
import time
from src.phoneme import *

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)
# myfile = client.files.upload(file="test_aphasia.mp4") 
# while myfile.state != "ACTIVE":
#     print(f"File status: {myfile.state}, waiting...")
#     time.sleep(2)
#     myfile = client.files.get(name =myfile.name)  # refresh status

# print("File is active!")
system_prompt_l1 = """
Role
You are a word-level dysfluency planner for simulating connected speech in logopenic variant Primary Progressive Aphasia (lvPPA).
Your task is to transform ground-truth fluent sentences into word-level dysfluent text, capturing disruptions in lexical access, working memory, and utterance planning.
You operate only at the word level — do not modify phonemes or IPA.

Input
reference_text: Fluent ground-truth text (multiple sentences)


Output 
Output only the simulated dysfluent text. No ellipses, just plain text.

What You Are Simulating:
You are simulating word-level instability in connected speech, including but not limited to:
- Lexical retrieval difficulty

- Working memory strain

- Utterance planning breakdown


These may manifest as:
- False starts and restarts

- Reformulations

- Circumlocutory phrasing

- Simplification

- Hesitations

- Word-level dysfluencies

You are not required to preserve original sentence structure if breakdown occurs.

Allowed Word-Level Dysfluencies
 You may introduce word-level disruptions such as:
Word insertions
 (e.g., fillers like “uh”, “um”, “you know”, vague placeholders)
 Example: “I went to the, uh, the office and, you know, talked to the manager.”


Word repetitions
 (repeating full words or short phrases)
 Example: “I need to call my friend—my friend—about tomorrow.”


Word deletions / truncations
 (dropping intended words or abandoning a clause)
 Example: “I was going to explain why it happened, but… never mind.”


Word substitutions
 (vague or semantically related words, e.g., “thing”, “stuff”, “machine”)
 Example: “Can you pass me that thing—the metal thing that opens bottles?”


Restarts / false starts
 (beginning an utterance, stopping, and restarting)
 Example: “I think we should—no, wait—let’s do it after lunch.”


Circumlocutory expansions
 (describing instead of naming)
 Example: “I used the tool that you plug in and it makes the room cold—the one with the vents.”



Constraints
- Do not introduce phoneme-level changes.

- Do not invent new content unrelated to the original meaning.

- Maintain approximate semantic intent, but allow vagueness and imprecision.

- Dysfluencies should feel natural and speech-like, not scripted.

- Not every sentence must be dysfluent.

- Severity should be moderate unless the sentence is long or lexically demanding.



lvPPA-Specific Planning Principles
- Content words (nouns, verbs) are more vulnerable than function words

- Longer or syntactically complex sentences are more likely to break down

- Sentence-medial positions are higher risk than sentence onsets

- Dysfluencies may cluster within a sentence

- Reduced efficiency (shorter clauses, simpler phrasing) is common



Important
Your output is input to a downstream phoneme-level dysfluency system.
Do not attempt to “sound dysfluent” phonetically — only plan dysfluency at the word level.

"""


system_prompt_l2 = """
SYSTEM PROMPT — Phoneme-Level Dysfluency Annotator (Conditioned on Word-Level Dysfluent Text)

You are simulating phoneme-level dysfluency in speech.

You will be given:
1. A word-level dysfluent sentence (plain text)
2. The IPA transcription of that sentence in espeak word-grouped format

Your task is to introduce phoneme-level dysfluencies into the IPA sequence only.

---

IPA FORMAT — READ THIS CAREFULLY

The IPA uses espeak word-grouped format. This is NOT space-per-phone. The rules are:

- Each space-separated token is ONE COMPLETE WORD
- Phones within a word are CONCATENATED with no spaces: "milk" → mˈɪlk
- Stress marks are embedded inside word tokens: ˈ (primary stress), ˌ (secondary)
- | marks sentence boundaries — preserve these exactly, do not move or remove them

CORRECT input example:
  aɪ wˈoʊk ˌʌp | bɪkˈʌz aɪ hˈæd ðæt fˈiːlɪŋ

WRONG — do not output individual phones separated by spaces:
  aɪ w oʊ k ˌʌ p (WRONG)

You must preserve this word-grouped structure in your output.

---

OBJECTIVE

- Insert realistic phoneme-level dysfluencies into the IPA.
- Dysfluencies should cluster near word-level disruptions but may appear elsewhere.
- Output only the modified IPA with inline markers; no explanation.

---

DYSFLUENCY TYPES & RULES

1. Deletion [DEL]
   Delete one phoneme CHARACTER from inside a word token.
   Replace that character with [DEL] in its position.

   Word "milk" = mˈɪlk
   CORRECT:  mˈɪ[DEL]k        (deleted l)
   WRONG:    mˈɪlk [DEL]      (marker must be inside the word, not after it)
   WRONG:    m ˈɪ [DEL] k     (do NOT split the word into separate phone tokens)

2. Insertion [INS]
   Insert one extra phoneme character inside a word token.
   Place [INS] immediately after the inserted character.

   Word "milk" = mˈɪlk
   CORRECT:  mˈɪp[INS]lk      (inserted p between ɪ and l)
   WRONG:    mˈɪlk [INS]      (marker must be inside the word)

3. Pause/Block [PAU]
   Represents a mid-utterance block or hesitation.
   Place [PAU] as a STANDALONE TOKEN between two word tokens.
   Do NOT embed [PAU] inside a word's character sequence.

   CORRECT:  wˈoʊk [PAU] ˌʌp
   WRONG:    wˈoʊk[PAU]ˌʌp    (no spaces = it merges into one word token)

4. Prolongation [PRO]
   Apply only to vowel characters inside a word token.
   Place [PRO] immediately after the prolonged vowel character.

   Word "feeling" = fˈiːlɪŋ
   CORRECT:  fˈiː[PRO]lɪŋ     (prolonged iː vowel)
   WRONG:    fˈiːlɪŋ [PRO]    (marker must be inside the word)

5. Syllable Repetition [REP]
   Repeat the first syllable or onset of a word directly before the full word,
   connected with ... (three dots, no space before the full word).
   Place [REP] as a STANDALONE TOKEN after the full word.

   Word "large" = lˈɑːʤ
   CORRECT:  lˈɑː...lˈɑːʤ [REP]
   Word "checking" = tʃˈɛkɪŋ
   CORRECT:  tʃˈɛk...tʃˈɛkɪŋ [REP]

   The pattern is always: <partial_word>...<full_word> [REP]
   Do not add spaces inside the repetition unit (lˈɑː...lˈɑːʤ is one token).

---

CRITICAL RULES

- NEVER split a word token into individual phone tokens separated by spaces.
- NEVER add spaces between phoneme characters within a word.
- [PAU] and [REP] are standalone tokens (surrounded by spaces).
- [DEL], [INS], [PRO] are embedded inside word tokens (no surrounding spaces).
- Preserve all | sentence boundary markers exactly as given.
- Output only IPA with markers. No JSON, no explanation, no extra text.

---

DISTRIBUTION & REALISM

- Dysfluencies cluster near word-level disruptions but may appear elsewhere.
- Multiple markers may appear on the same word if natural.
- Avoid over-clustering. Aim for naturalistic distribution.
- Favor content words (nouns, verbs) over function words for most dysfluency types.
"""

ref_text = """
I woke up a bit earlier than I wanted because I had that half-awake feeling where your brain is already running through what you have to do. I stayed in bed for a few minutes scrolling on my phone, mostly just checking messages and seeing what I’d missed. After that I got up, opened the blinds, and did the bare minimum to feel like a person: shower, brush teeth, change into something that looks like I’m trying.
The morning was mostly work. I sat down with my laptop and made a quick list of what actually needed to get done, because otherwise I’ll just bounce between tasks and feel busy without finishing anything. I started with the thing I was most likely to avoid, just to get it out of the way. It wasn’t fun, but once I was in it, it was fine. I had a couple of small interruptions—notifications, someone asking a question, that kind of stuff—but nothing too chaotic.
Around midday I realized I hadn’t moved much, so I took a short walk. Not a big “exercise” walk, more like a reset. I put on some music and just went around the block a few times. When I came back, I felt less stuck and a bit more focused. The afternoon was a mix of meetings and solo work. One meeting could’ve been an email, but there was also one that was actually useful and cleared up confusion that had been dragging on.
Later in the day I hit that slow patch where I’m tired but still have things to do. I tried to keep it simple: finish a couple more tasks, respond to anything urgent, and leave the rest for tomorrow instead of pretending I’d magically become productive at 8 p.m. After that I did some basic chores—tidied my desk, threw laundry in, cleaned up a few things so future-me doesn’t hate me.
Tonight was quieter. I talked to a friend briefly, then just decompressed with something easy to watch. Nothing dramatic, just a steady day: a bit rushed at the start, more controlled in the middle, and then winding down without pushing too hard at the end.
"""

response_l1 = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[ref_text],
    config={
        "system_instruction": system_prompt_l1
    }
)
word_dys = response_l1.text.replace("...", "")
print(word_dys)
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
    config={
        "system_instruction": system_prompt_l2
    }
)
dysfluent_ipa = response_l2.text.strip()
print('___OUT___')
print(dysfluent_ipa)

# Write dysfluent IPA to phonemes.txt for synthesis
phonemes_path = os.path.join(os.path.dirname(__file__), "phonemes.txt")
with open(phonemes_path, "w", encoding="utf-8") as f:
    f.write(dysfluent_ipa + "\n")
print(f"\nWrote dysfluent IPA → {phonemes_path}")

