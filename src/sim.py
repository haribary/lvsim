# import torch
from google import genai
from dotenv import load_dotenv
import os
import time
from phoneme import *

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
2. The IPA transcription of that sentence, already aligned to the words

Your task is to introduce phoneme-level dysfluencies into the IPA sequence only, while using the word-level dysfluency as a guide for likely areas of instability.

---

INPUTS

- Word-level dysfluent text: contains phenomena such as restarts, repetitions, circumlocutions, fillers, or vague lexical substitutions.
- IPA transcription: exact phonemic transcription of the word-level text. This is the only sequence you are allowed to modify.

---

OBJECTIVE

- Insert realistic phoneme-level dysfluencies into the IPA sequence.
- Dysfluencies should occur more frequently near word-level disruptions, but may also appear elsewhere.
- Maintain phonetic and structural integrity of the IPA transcription.
- Output only the IPA transcription with inline dysfluency annotations.

---

CRITICAL CONSTRAINTS

- Modify only the IPA input.
- Do not alter word order, word boundaries, or lexical content, or '|' markers.
- Dysfluency markers must:
  - Modify actual phonemes
  - Appear within words, never between words
- Output only IPA with markers; no JSON or explanation.

---

DYSFLUENCY TYPES & RULES

1. Deletion [DEL]
- Delete one existing phoneme within a word.
- Replace the deleted phoneme with [DEL], exactly in its position.
- Examples:
  CORRECT:  m ɪ [DEL] k
  WRONG:    m ɪ l k [DEL]
  WRONG:    m ɪ k

2. Insertion [INS]
- Insert one phoneme that differs from both neighbors.
- Place [INS] immediately after the inserted phoneme.
- Must be within a word, never at boundaries.
- Examples:
  CORRECT:  m ɪ p [INS] l k
  WRONG:    m ɪ l k [INS]
  WRONG:    m ɪ l [INS] k

3. Pause/Block [PAU]
- Place [PAU] within a word at a natural internal stopping point (e.g., after a consonant).
- Example:
  p ɹ ɪ d ɪ k [PAU] t ᵻ d

4. Prolongation [PRO]
- Apply only to vowels.
- Place [PRO] immediately after the prolonged vowel.
- Favor content words, especially near word-level disruptions.
- Example:
  ɪ v æ [PRO] l j u

5. Syllable Repetition [REP]
- Repeat entire syllables, not single phonemes.
- Use ellipsis ... for repetition.
- Place [REP] after the repeated syllable.
- Example:
  l ɑ ɹ ... [REP] l ɑ ɹ ʤ
- Favor initial or stressed syllables and high-risk consonants (plosives, fricatives, affricates).

---

DISTRIBUTION & REALISM

- Dysfluencies should be more frequent near word-level disruptions.
- Multiple types may occur on the same word if natural.
- Avoid excessive clustering that breaks readability.
- Maintain diversity and naturalistic phonological patterns.

---

OUTPUT FORMAT

- Output only the IPA transcription with inline dysfluency markers.
- No JSON, no explanations, no extra text.
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
print('___OUT___')
print(response_l2.text)

