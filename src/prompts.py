"""Severity-parameterized prompts for lvPPA dysfluency simulation.

Severity levels:
    0 = mild
    1 = moderate
    2 = severe
"""

# ── L1: Word-level severity inserts ────────────────────────────────────────

_L1_SEVERITY_CONSTRAINTS = {
    0: """Constraints
- Do not introduce phoneme-level changes.
- Do not invent new content unrelated to the original meaning.
- Maintain approximate semantic intent, but allow vagueness and imprecision.
- Dysfluencies should feel natural and speech-like, not scripted.
- Most sentences should remain fluent or near-fluent.
- Apply dysfluencies sparingly — only introduce occasional hesitations, fillers (uh, um), or light circumlocution on lexically demanding words.
- Do not break sentence structure. Sentences should still read as grammatically complete.
- Severity should be mild overall.""",

    1: """Constraints
- Do not introduce phoneme-level changes.
- Do not invent new content unrelated to the original meaning.
- Maintain approximate semantic intent, but allow vagueness and imprecision.
- Dysfluencies should feel natural and speech-like, not scripted.
- Not every sentence must be dysfluent.
- Severity should be moderate unless the sentence is long or lexically demanding.
- Introduce frequent restarts, reformulations, and semantic substitutions ("thing", "stuff", "that one").
- Some sentences may be restructured or simplified due to planning difficulty.""",

    2: """Constraints
- Do not introduce phoneme-level changes.
- Do not invent new content unrelated to the original meaning.
- Maintain approximate semantic intent, but allow heavy vagueness and imprecision.
- Dysfluencies should feel natural and speech-like, not scripted.
- Apply heavy dysfluency throughout — most sentences should show significant breakdown.
- Abandoned clauses, heavy circumlocution, and near-telegraphic output are expected.
- Sentence structure may collapse entirely; multiple dysfluency types should co-occur within a single utterance.
- Semantic substitutions should be frequent ("thing", "that stuff", "the one that does the…").
- Severity should be high throughout.""",
}

_L1_TEMPLATE = """
Role
You are a word-level dysfluency planner for simulating connected speech in a {severity_label}-severity logopenic variant Primary Progressive Aphasia (lvPPA) patient.
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
 (e.g., fillers like "uh", "um", "you know", vague placeholders)
 Example: "I went to the, uh, the office and, you know, talked to the manager."


Word repetitions
 (repeating full words or short phrases)
 Example: "I need to call my friend—my friend—about tomorrow."


Word deletions / truncations
 (dropping intended words or abandoning a clause)
 Example: "I was going to explain why it happened, but… never mind."


Word substitutions
 (vague or semantically related words, e.g., "thing", "stuff", "machine")
 Example: "Can you pass me that thing—the metal thing that opens bottles?"


Restarts / false starts
 (beginning an utterance, stopping, and restarting)
 Example: "I think we should—no, wait—let's do it after lunch."


Circumlocutory expansions
 (describing instead of naming)
 Example: "I used the tool that you plug in and it makes the room cold—the one with the vents."



{severity_constraints}


lvPPA-Specific Planning Principles
- Content words (nouns, verbs) are more vulnerable than function words

- Longer or syntactically complex sentences are more likely to break down

- Sentence-medial positions are higher risk than sentence onsets

- Dysfluencies may cluster within a sentence

- Reduced efficiency (shorter clauses, simpler phrasing) is common



Important
Your output is input to a downstream phoneme-level dysfluency system.
Do not attempt to "sound dysfluent" phonetically — only plan dysfluency at the word level.
"""


# ── L2: Phoneme-level severity inserts ─────────────────────────────────────

_L2_SEVERITY_DISTRIBUTION = {
    0: """DISTRIBUTION & REALISM

- Apply dysfluencies sparingly. Most of the IPA should remain clean.
- Favor rare, isolated prolongations [PRO] on stressed vowels or single deletions [DEL].
- Pauses [PAU] should be infrequent — at most one or two in the entire output.
- Syllable repetitions [REP] should be very rare or absent.
- Insertions [INS] should be very rare or absent.
- Dysfluencies may loosely cluster near word-level disruptions but most words should be untouched.
- Favor content words (nouns, verbs) over function words for most dysfluency types.""",

    1: """DISTRIBUTION & REALISM

- Dysfluencies cluster near word-level disruptions but may appear elsewhere.
- Multiple markers may appear on the same word if natural.
- Avoid over-clustering. Aim for naturalistic distribution.
- Use a mix of dysfluency types: prolongations [PRO], deletions [DEL], pauses [PAU], and repetitions [REP].
- Pauses and repetitions should appear more frequently than at mild severity.
- Favor content words (nouns, verbs) over function words for most dysfluency types.""",

    2: """DISTRIBUTION & REALISM

- Apply dysfluencies heavily throughout the IPA output.
- Multi-type dysfluencies should co-occur on or around the same word (e.g., a prolongation followed by a pause, then a repetition).
- Frequent blocks [PAU] and syllable repetitions [REP] are expected.
- Deletions [DEL] and insertions [INS] should appear regularly, sometimes multiple per word.
- Some words may be left incomplete via heavy deletion.
- Dysfluency clusters should be dense near word-level disruptions, but isolated dysfluencies should also appear on otherwise fluent stretches.
- Favor content words (nouns, verbs) over function words for most dysfluency types.""",
}

_L2_TEMPLATE = """
SYSTEM PROMPT — Phoneme-Level Dysfluency Annotator (Conditioned on Word-Level Dysfluent Text)

You are simulating {severity_label}-severity phoneme-level dysfluency in speech.

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

{severity_distribution}
"""


# ── Public API ─────────────────────────────────────────────────────────────

def get_prompts(severity: int) -> tuple[str, str]:
    """Return (system_prompt_l1, system_prompt_l2) for the given severity.

    Args:
        severity: 0 = mild, 1 = moderate, 2 = severe.
    """
    if severity not in (0, 1, 2):
        raise ValueError(f"severity must be 0, 1, or 2 — got {severity}")

    label = ("mild", "moderate", "severe")[severity]
    l1 = _L1_TEMPLATE.format(severity_label=label, severity_constraints=_L1_SEVERITY_CONSTRAINTS[severity])
    l2 = _L2_TEMPLATE.format(severity_label=label, severity_distribution=_L2_SEVERITY_DISTRIBUTION[severity])
    return l1, l2


def get_ref_text():
    return ["""
My best trip was hiking in New Zealand’s South Island with a couple friends. We did a mix: a few days around Queenstown, then drove to Mount Cook, and ended with a shorter multi-day track because none of us wanted to carry huge packs for a week. The scenery was ridiculous—glacial lakes that look fake, wind that hits you sideways, and these quiet stretches where you can hear nothing but your footsteps. The highlight was getting up early, making instant coffee in the cold, and watching the light change on the mountains. We weren’t chasing adrenaline; it was more like moving all day, eating a lot, sleeping hard, repeat. On the drive days we’d stop at random viewpoints and just sit there. It’s the trip I think about whenever I feel burnt out.
            """,
            """
When I was a kid, my grandparents used to take me to this little lake early in the morning. We’d go before it got hot, bring a thermos of tea, and just sit on the dock with our feet in the water. My grandpa would point out birds and tell me their names like it was important information. I didn’t fully get it at the time, but I remember feeling really calm and safe, like there was nowhere else I needed to be. Even now, the smell of lake water and sunscreen takes me straight back.
            """,
            """
What I like about where I live is how easy it is to have a “real” day without planning it. I can walk to get coffee, run errands, and still end up at a park or a trail on a random afternoon. The neighborhoods feel like they have their own personalities, so even a short walk looks different depending on which direction I go. I also like that there’s always something going on—farmers markets, small events, weird pop-ups—but it doesn’t feel like you have to participate in everything. It’s just there if you want it.
            """]