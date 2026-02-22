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
- Retrieval failures should be brief and quickly resolved — the speaker finds the word after a short delay or one restart.
- Metacognitive comments (e.g., "what's the word") should be rare or absent.
- Do not break sentence structure. Sentences should still read as grammatically complete.
- Severity should be mild overall.""",

    1: """Constraints
- Do not introduce phoneme-level changes.
- Do not invent new content unrelated to the original meaning.
- Maintain approximate semantic intent, but allow vagueness and imprecision.
- Dysfluencies should feel natural and speech-like, not scripted.
- Not every sentence must be dysfluent, but most lexically demanding sentences should show some breakdown.
- Severity should be moderate unless the sentence is short and syntactically simple.
- Retrieval failures should sometimes cascade: a failed retrieval on one word destabilizes the rest of the utterance, causing restarts, abandoned clauses, or simplified re-attempts.
- The speaker may circle back to the same clause multiple times, each time getting slightly closer or giving up and simplifying.
- Introduce frequent restarts, reformulations, and semantic substitutions ("thing", "stuff", "that one", "the one that...").
- Metacognitive comments should appear occasionally (e.g., "I know what I mean", "what do you call it").
- Some sentences may be restructured or simplified due to planning difficulty.
- Abandoned utterances are allowed — the speaker may trail off and start a new thought.""",

    2: """Constraints
- Do not introduce phoneme-level changes.
- Do not invent new content unrelated to the original meaning.
- Maintain approximate semantic intent, but allow heavy vagueness and imprecision.
- Dysfluencies should feel natural and speech-like, not scripted.
- Apply heavy dysfluency throughout — most sentences should show significant breakdown.
- Retrieval failures should frequently cascade: one failed word derails the entire utterance plan, causing the speaker to restart the same clause multiple times, each attempt slightly different, often abandoning and restarting from scratch.
- "Revolving door" restarts are expected — the speaker loops through the same clause 2-4 times before resolving or abandoning it entirely.
- Metacognitive frustration should surface regularly (e.g., "I know what I wanna say but...", "what's the—you know what I mean", "the word won't come").
- Abandoned clauses, heavy circumlocution, and near-telegraphic output are expected.
- Sentence structure may collapse entirely; multiple dysfluency types should co-occur within a single utterance.
- Semantic substitutions should be frequent ("thing", "that stuff", "the one that does the…", "type of thing").
- Temporal and sequential confusion may appear — the speaker may lose track of the order of events or mix up details.
- The speaker may insert filler phrases to hold conversational ground while searching for words (e.g., "and uh... and uh...").
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
You are simulating word-level instability in connected speech caused by lexical retrieval failure and its downstream effects. In lvPPA, the core deficit is difficulty accessing words during speech. This is NOT a grammar or comprehension problem — the speaker knows what they want to say but cannot retrieve the words in time. When a word fails to come, the consequences cascade:

- The speaker stalls, fills time with hesitations, and searches for the word.
- If the word still doesn't come, they may try a vaguer substitute, describe it instead of naming it, or abandon the attempt entirely.
- The delay and cognitive effort of searching for one word disrupts working memory, causing the speaker to lose track of the sentence they were building.
- This forces restarts — the speaker goes back to the beginning of the clause and tries again, sometimes multiple times.
- Each restart may differ slightly as the speaker tries a new route to the same idea.
- Longer and more complex sentences are more likely to collapse because they place greater demands on working memory during the retrieval delays.

These cascading failures are the hallmark of lvPPA connected speech. The dysfluencies are not random — they are consequences of retrieval breakdowns propagating through the utterance plan.


These may manifest as:
- Hesitations and filled pauses (uh, um) while searching for a word
- False starts and restarts — beginning an utterance, failing to retrieve a word, and restarting from earlier in the clause
- Revolving-door restarts — cycling through the same clause multiple times, each attempt slightly different
- Reformulations — switching to a simpler sentence structure after a complex one fails
- Circumlocutory phrasing — describing a word instead of naming it ("the thing you use to... you know, the...")
- Semantic substitutions — using a vaguer or related word when the target won't come ("thing", "stuff", "the one")
- Abandoned utterances — trailing off when retrieval fails completely and the sentence plan is lost
- Metacognitive comments — the speaker acknowledges the difficulty ("I know what I mean", "what's the word", "it won't come")
- Filler stalling — using repeated fillers to hold conversational ground while searching ("and uh... and uh...")

You are not required to preserve original sentence structure if breakdown occurs.

Allowed Word-Level Dysfluencies
 You may introduce word-level disruptions such as:
Word insertions
 (e.g., fillers like "uh", "um", "you know", vague placeholders)
 Example: "I went to the, uh, the office and, you know, talked to the manager."


Word repetitions
 (repeating full words or short phrases as the speaker re-anchors after a retrieval delay)
 Example: "I need to call my friend—my friend—about tomorrow."


Word deletions / truncations
 (dropping intended words or abandoning a clause when retrieval fails)
 Example: "I was going to explain why it— never mind."


Word substitutions
 (vague or semantically related words when the target word won't come)
 Example: "Can you pass me that thing—the metal thing that opens bottles?"


Restarts / false starts
 (beginning an utterance, failing to retrieve a word, backing up and restarting — possibly multiple times)
 Example: "I think we should—no, wait—let's do it after lunch."
 Example (revolving-door): "She had to go—uh, be there at—she had to—she had to go at twelve o'clock."


Circumlocutory expansions
 (describing instead of naming when the word won't come)
 Example: "I used the—the thing that you plug in and it makes the room cold—the one with the vents."


Abandoned utterances
 (trailing off when retrieval failure causes complete loss of the sentence plan)
 Example: "And then she was going to—uh—I know what I mean but—anyway, so then they left."


Metacognitive comments
 (the speaker acknowledges their retrieval difficulty)
 Example: "It's the—what do you call it—the thing on the wall."
 Example: "I know what I wanna say but I can't get it out."


{severity_constraints}


lvPPA-Specific Planning Principles
- Content words (nouns, verbs) are the primary targets of retrieval failure — function words are largely spared.
- A single retrieval failure often destabilizes the entire utterance, causing cascading restarts and reformulations.
- Longer or syntactically complex sentences are more likely to break down because retrieval delays overload working memory.
- Sentence-medial positions are higher risk than sentence onsets — the speaker has committed to a structure but cannot retrieve the next content word.
- Dysfluencies cluster around retrieval failure sites — they are not randomly distributed.
- Reduced efficiency (shorter clauses, simpler phrasing) emerges as a compensatory strategy, not a primary deficit.
- The speaker retains awareness of their difficulty and may comment on it or show frustration.
- The speaker's semantic knowledge is intact — circumlocutions and descriptions are accurate even when the target word won't come.


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
- Substitutions [SUB] should be very rare or absent.
- Dysfluencies may loosely cluster near word-level disruptions but most words should be untouched.
- Favor content words (nouns, verbs) over function words for most dysfluency types.""",

    1: """DISTRIBUTION & REALISM

- Dysfluencies cluster near word-level disruptions but may appear elsewhere.
- Multiple markers may appear on the same word if natural.
- Avoid over-clustering. Aim for naturalistic distribution.
- Use a mix of dysfluency types: prolongations [PRO], deletions [DEL], pauses [PAU], repetitions [REP], and substitutions [SUB].
- Pauses, repetitions, and substitutions should appear more frequently than at mild severity.
- Substitutions should reflect phonological neighborhood errors — the substituted phoneme should be articulatorily close to the target (e.g., similar place or manner of articulation).
- Favor content words (nouns, verbs) over function words for most dysfluency types.""",

    2: """DISTRIBUTION & REALISM

- Apply dysfluencies heavily throughout the IPA output.
- Multi-type dysfluencies should co-occur on or around the same word (e.g., a prolongation followed by a pause, then a repetition).
- Frequent blocks [PAU] and syllable repetitions [REP] are expected.
- Deletions [DEL], insertions [INS], and substitutions [SUB] should appear regularly, sometimes multiple per word.
- Substitutions may be more phonologically distant at this severity — reflecting greater phonological instability.
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
   Repeat the first FULL SYLLABLE (onset + nucleus, or nucleus alone if no onset) of a word directly before the full word, connected with ... (three dots, no space before the full word).
   Place [REP] as a STANDALONE TOKEN after the full word.

   The repeated portion must be a complete syllable — not just the onset consonant(s) alone.
   Include the onset consonant(s) AND the vowel nucleus of the first syllable.

   Word "large" = lˈɑːʤ
   CORRECT:  lˈɑː...lˈɑːʤ [REP]        (repeated first syllable: onset l + nucleus ɑː)
   WRONG:    l...lˈɑːʤ [REP]            (onset only — missing the vowel nucleus)

   Word "checking" = tʃˈɛkɪŋ
   CORRECT:  tʃˈɛk...tʃˈɛkɪŋ [REP]     (repeated first syllable: onset tʃ + nucleus ɛ + coda k)
   WRONG:    tʃ...tʃˈɛkɪŋ [REP]         (onset only — missing the vowel)

   Word "open" = ˈoʊpən
   CORRECT:  ˈoʊ...ˈoʊpən [REP]         (no onset; repeated nucleus oʊ of first syllable)
   WRONG:    ˈoʊpən...ˈoʊpən [REP]      (repeated entire word, not just first syllable)

   The pattern is always: <first_syllable>...<full_word> [REP]
   Do not add spaces inside the repetition unit (lˈɑː...lˈɑːʤ is one token).

6. Substitution [SUB]
   Replace one phoneme CHARACTER inside a word token with a different phoneme.
   Place [SUB] immediately after the substituted (new) phoneme.

   The substituted phoneme must be DIFFERENT from the original phoneme.
   The substitution should be a real phoneme that could plausibly result from phonological error.

   Word "milk" = mˈɪlk
   CORRECT:  mˈɪn[SUB]k       (substituted n for l — both alveolar, plausible error)
   WRONG:    mˈɪlk[SUB]       (no phoneme was actually substituted)
   WRONG:    mˈɪl[SUB]k       (tag must follow the NEW phoneme, not the original)
   WRONG:    m ˈɪ n[SUB] k    (do NOT split the word into separate phone tokens)

   Word "feeling" = fˈiːlɪŋ
   CORRECT:  fˈiːlɪn[SUB]     (substituted n for ŋ — both nasals, plausible error)

   Prefer articulatorily close substitutions (same place or manner of articulation) at mild and moderate severity.

---

CRITICAL RULES

- NEVER split a word token into individual phone tokens separated by spaces.
- NEVER add spaces between phoneme characters within a word.
- [PAU] and [REP] are standalone tokens (surrounded by spaces).
- [DEL], [INS], [PRO], [SUB] are embedded inside word tokens (no surrounding spaces).
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
My best trip was hiking in New Zealand's South Island with a couple friends. We did a mix: a few days around Queenstown, then drove to Mount Cook, and ended with a shorter multi-day track because none of us wanted to carry huge packs for a week. The scenery was ridiculous—glacial lakes that look fake, wind that hits you sideways, and these quiet stretches where you can hear nothing but your footsteps. The highlight was getting up early, making instant coffee in the cold, and watching the light change on the mountains. We weren't chasing adrenaline; it was more like moving all day, eating a lot, sleeping hard, repeat. On the drive days we'd stop at random viewpoints and just sit there. It's the trip I think about whenever I feel burnt out.
            """,
            """
When I was a kid, my grandparents used to take me to this little lake early in the morning. We'd go before it got hot, bring a thermos of tea, and just sit on the dock with our feet in the water. My grandpa would point out birds and tell me their names like it was important information. I didn't fully get it at the time, but I remember feeling really calm and safe, like there was nowhere else I needed to be. Even now, the smell of lake water and sunscreen takes me straight back.
            """,
            """
What I like about where I live is how easy it is to have a "real" day without planning it. I can walk to get coffee, run errands, and still end up at a park or a trail on a random afternoon. The neighborhoods feel like they have their own personalities, so even a short walk looks different depending on which direction I go. I also like that there's always something going on—farmers markets, small events, weird pop-ups—but it doesn't feel like you have to participate in everything. It's just there if you want it.
            """]