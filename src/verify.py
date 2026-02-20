import os
os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", "/opt/homebrew/lib/libespeak-ng.dylib")

import json
import re
from phonemizer import phonemize
from phonemizer.separator import Separator

# ---------------------------------------------------------------------------
# 1. Parse output.txt
# ---------------------------------------------------------------------------

def parse_output(filepath):
    raw = open(filepath).read()

    # ref text
    ref_start = raw.index("ref text: ") + len("ref text: ")
    ref_end = raw.index("\n\nipa correct: ")
    ref_text = raw[ref_start:ref_end].strip()

    # events JSON
    events_start = raw.index("events: ") + len("events: ")
    # find the balanced JSON array
    events_json_str = _extract_json_array(raw[events_start:])
    events = json.loads(events_json_str)

    # dysfluent IPA JSON – embedded in SDK repr as text="""..."""
    l2_match = re.search(r'text="""\{(.*?)\}"""', raw, re.DOTALL)
    l2_json_str = "{" + l2_match.group(1) + "}"
    dysfluent = json.loads(l2_json_str)["sentences"]

    return ref_text, events, dysfluent


def _extract_json_array(text):
    depth = 0
    start = text.index("[")
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("unbalanced JSON array")

# ---------------------------------------------------------------------------
# 2. Phonemize with word boundaries
# ---------------------------------------------------------------------------

def phonemize_with_words(sentences, language="en-us"):
    return phonemize(
        sentences,
        language=language,
        backend="espeak",
        separator=Separator(phone=" ", word="|", syllable=""),
        strip=True,
        words_mismatch="warn",
    )


def build_word_phone_map(ipa_with_words):
    """Return list of {phones, phone_start, phone_end} per word."""
    word_groups = ipa_with_words.split("|")
    mapping = []
    offset = 0
    for group in word_groups:
        phones = [p for p in group.strip().split() if p]
        if not phones:
            continue
        mapping.append({
            "phones": phones,
            "phone_start": offset,
            "phone_end": offset + len(phones) - 1,
        })
        offset += len(phones)
    return mapping


def get_sentence_words(sentence_text):
    return [w.lower() for w in re.findall(r"[a-zA-Z']+", sentence_text)]

# ---------------------------------------------------------------------------
# 3. Tokenize dysfluent IPA
# ---------------------------------------------------------------------------

def tokenize_ipa(ipa_string):
    tokens = []
    for t in ipa_string.split():
        if t == ":" and tokens:
            tokens[-1] += ":"
        else:
            tokens.append(t)
    return tokens

# ---------------------------------------------------------------------------
# 4. Levenshtein alignment with backtrace
# ---------------------------------------------------------------------------

def levenshtein_align(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = "I"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                bt[i][j] = "M"
            else:
                choices = [
                    (dp[i - 1][j - 1] + 1, "S"),
                    (dp[i - 1][j] + 1, "D"),
                    (dp[i][j - 1] + 1, "I"),
                ]
                dp[i][j], bt[i][j] = min(choices, key=lambda x: x[0])

    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and bt[i][j] == "M":
            ops.append(("match", i - 1, j - 1))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and bt[i][j] == "S":
            ops.append(("substitute", i - 1, j - 1))
            i -= 1; j -= 1
        elif i > 0 and bt[i][j] == "D":
            ops.append(("delete", i - 1, None))
            i -= 1
        elif j > 0 and bt[i][j] == "I":
            ops.append(("insert", None, j - 1))
            j -= 1
        else:
            break
    ops.reverse()
    return ops

# ---------------------------------------------------------------------------
# 5. Classify and map ops to words
# ---------------------------------------------------------------------------

def classify_ops(ops, ref, hyp):
    result = []
    for op_type, ri, hi in ops:
        entry = {"op": op_type, "ref_idx": ri, "hyp_idx": hi}
        if op_type == "substitute" and ri is not None and hi is not None:
            if hyp[hi] == ref[ri] + ":":
                entry["op"] = "prolongation"
            entry["ref_phone"] = ref[ri]
            entry["hyp_phone"] = hyp[hi]
        elif op_type == "insert" and hi is not None:
            entry["hyp_phone"] = hyp[hi]
        elif op_type == "delete" and ri is not None:
            entry["ref_phone"] = ref[ri]
        result.append(entry)
    return result


def map_ops_to_words(ops, word_phone_map):
    def word_for_ref(idx):
        for i, w in enumerate(word_phone_map):
            if w["phone_start"] <= idx <= w["phone_end"]:
                return i
        return None

    last_ref = None
    for op in ops:
        if op["ref_idx"] is not None:
            op["word_idx"] = word_for_ref(op["ref_idx"])
            last_ref = op["ref_idx"]
        else:
            op["word_idx"] = word_for_ref(last_ref) if last_ref is not None else 0
            op["insert_after_ref"] = last_ref
    return ops

# ---------------------------------------------------------------------------
# 6. Event verification
# ---------------------------------------------------------------------------

FILLERS = {"ə", "əm", "ɜr", "ɜːr", "ɜːɹ", "ɜr"}


def _strip_pause_suffix(phone):
    """Strip trailing '...' from a token: 'ə...' → 'ə', '...' → '...'."""
    if phone.endswith("...") and len(phone) > 3:
        return phone[:-3]
    return phone


def _is_filler_or_pause(phone):
    base = _strip_pause_suffix(phone)
    return base in FILLERS or phone == "..."


def find_word_idx(target, words, wmap, hint_after=None):
    """Find target word's wmap index by phonemizing it and matching phones."""
    t = target.lower().strip(".,;:!?-'\"")

    # phonemize the target word to get its expected phones
    try:
        target_ipa = phonemize(
            [target], language="en-us", backend="espeak",
            separator=Separator(phone=" ", word="", syllable=""),
            strip=True,
        )[0].split()
    except Exception:
        target_ipa = None

    # strategy 1: find wmap entries whose phones match the target IPA
    if target_ipa:
        candidates = []
        for i, w in enumerate(wmap):
            # exact match or high overlap (>= 60% phones shared)
            if w["phones"] == target_ipa:
                candidates.append((i, 1.0))
            else:
                common = sum(1 for p in target_ipa if p in w["phones"])
                ratio = common / max(len(target_ipa), 1)
                if ratio >= 0.6 and len(w["phones"]) <= len(target_ipa) + 2:
                    candidates.append((i, ratio))

        if candidates:
            if hint_after is not None:
                after_cands = [(i, r) for i, r in candidates if i >= hint_after]
                if after_cands:
                    return max(after_cands, key=lambda x: x[1])[0]
            return max(candidates, key=lambda x: x[1])[0]

    # strategy 2: fall back to word-list index alignment (clamped)
    max_idx = len(wmap) - 1
    matches = [min(i, max_idx) for i, w in enumerate(words) if w == t]
    if not matches:
        matches = [min(i, max_idx) for i, w in enumerate(words) if t in w or w in t]
    if not matches:
        return None
    if hint_after is not None:
        for m in matches:
            if m >= hint_after:
                return m
    return matches[-1] if len(matches) > 1 else matches[0]


def _inserts_near(ops, phone_start, phone_end, target_phones):
    """Check for filler/pause inserts or repetitions near a word boundary."""
    filler = False
    rep = False
    for op in ops:
        if op["op"] != "insert" or not op.get("hyp_phone"):
            continue
        after = op.get("insert_after_ref")
        if after is None:
            after = -1
        # within a generous window around the word
        if phone_start - 5 <= after <= phone_end:
            if _is_filler_or_pause(op["hyp_phone"]):
                filler = True
            base = _strip_pause_suffix(op["hyp_phone"])
            if base in target_phones or op["hyp_phone"] in target_phones:
                rep = True
    return filler, rep


def verify_lexical_retrieval_failure(target_word, ops, words, wmap):
    idx = find_word_idx(target_word, words, wmap)
    if idx is None:
        return {"pass": False, "reason": f"'{target_word}' not found in words"}
    start = wmap[idx]["phone_start"]
    end = wmap[idx]["phone_end"]
    target_phones = set(wmap[idx]["phones"])

    filler_before, repetition = _inserts_near(ops, start, end, target_phones)

    return {
        "pass": filler_before or repetition,
        "filler_before": filler_before,
        "repetition": repetition,
        "target": target_word,
    }


def verify_phonological_encoding_failure(target_word, ops, words, wmap):
    idx = find_word_idx(target_word, words, wmap)
    if idx is None:
        return {"pass": False, "reason": f"'{target_word}' not found in words"}
    start, end = wmap[idx]["phone_start"], wmap[idx]["phone_end"]
    target_phones = set(wmap[idx]["phones"])

    sub = del_ = prolong = rep = False
    for op in ops:
        ri = op.get("ref_idx")
        if ri is not None and start <= ri <= end:
            if op["op"] == "substitute":
                sub = True
            elif op["op"] == "delete":
                del_ = True
            elif op["op"] == "prolongation":
                prolong = True
        # check inserts: either assigned to this word OR near its boundary
        if op["op"] == "insert" and op.get("hyp_phone"):
            after = op.get("insert_after_ref")
            if after is None:
                after = -1
            if start - 3 <= after <= end:
                base = _strip_pause_suffix(op["hyp_phone"])
                if base in target_phones or op["hyp_phone"] in target_phones:
                    rep = True

    return {
        "pass": sub or del_ or prolong or rep,
        "substitution": sub,
        "deletion": del_,
        "prolongation": prolong,
        "repetition": rep,
        "target": target_word,
    }


def verify_word_finding_pause(event, ops, words, wmap, sibling_events=None):
    if "target_word" in event:
        anchor, pos = event["target_word"], "before"
    else:
        anchor, pos = event["after_word"], "after"

    # use sibling events to hint word position for disambiguation
    hint = None
    if sibling_events:
        for sib in sibling_events:
            if sib is event:
                continue
            sib_word = sib.get("target_word") or sib.get("after_word")
            if sib_word:
                sib_idx = find_word_idx(sib_word, words, wmap)
                if sib_idx is not None:
                    hint = max(0, sib_idx - 3)
                    break

    idx = find_word_idx(anchor, words, wmap, hint_after=hint)
    if idx is None:
        return {"pass": False, "reason": f"'{anchor}' not found in words"}
    start, end = wmap[idx]["phone_start"], wmap[idx]["phone_end"]

    found = False
    for op in ops:
        if op["op"] != "insert" or not op.get("hyp_phone"):
            continue
        if not _is_filler_or_pause(op["hyp_phone"]):
            continue
        after = op.get("insert_after_ref")
        if after is None:
            after = -1
        if pos == "after" and end - 1 <= after <= end + 3:
            found = True
        elif pos == "before" and start - 5 <= after < start:
            found = True

    return {"pass": found, "anchor": anchor, "position": pos}


def verify_sentence_abandonment(after_word, ops, words, wmap, hyp_tokens):
    idx = find_word_idx(after_word, words, wmap)
    if idx is None:
        return {"pass": False, "reason": f"'{after_word}' not found in words"}
    end = wmap[idx]["phone_end"]

    total_after = sum(len(w["phones"]) for i, w in enumerate(wmap) if i > idx)
    deleted_after = sum(
        1 for op in ops
        if op["op"] == "delete" and op.get("ref_idx") is not None and op["ref_idx"] > end
    )
    ratio = deleted_after / max(total_after, 1)
    ends_pause = len(hyp_tokens) > 0 and hyp_tokens[-1] == "..."

    return {
        "pass": ratio > 0.7 or ends_pause,
        "deletion_ratio": f"{ratio:.0%}",
        "ends_with_pause": ends_pause,
        "after_word": after_word,
    }

# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main():
    ref_text, events_list, dysfluent_list = parse_output("output.txt")

    sentences = [e["sentence_text"] for e in events_list]
    dysfluent_map = {d["sentence_text"]: d["ipa_dysfluent"] for d in dysfluent_list}

    # batch phonemize with word boundaries
    ipas_with_words = phonemize_with_words(sentences)

    total_events = 0
    total_pass = 0

    for sent_entry, ipa_ww in zip(events_list, ipas_with_words):
        sentence_text = sent_entry["sentence_text"]
        sent_events = sent_entry["events"]

        print(f"\n{'=' * 70}")
        print(f"SENTENCE: {sentence_text}")

        if not sent_events:
            print("  (no events — skip)")
            continue

        ipa_dysfluent = dysfluent_map.get(sentence_text)
        if ipa_dysfluent is None:
            print("  WARNING: no dysfluent IPA found")
            continue

        wmap = build_word_phone_map(ipa_ww)
        clean = []
        for w in wmap:
            clean.extend(w["phones"])
        dysfluent_tokens = tokenize_ipa(ipa_dysfluent)
        words = get_sentence_words(sentence_text)

        raw_ops = levenshtein_align(clean, dysfluent_tokens)
        classified = classify_ops(raw_ops, clean, dysfluent_tokens)
        word_ops = map_ops_to_words(classified, wmap)

        n_m = sum(1 for o in word_ops if o["op"] == "match")
        n_s = sum(1 for o in word_ops if o["op"] == "substitute")
        n_d = sum(1 for o in word_ops if o["op"] == "delete")
        n_i = sum(1 for o in word_ops if o["op"] == "insert")
        n_p = sum(1 for o in word_ops if o["op"] == "prolongation")
        print(f"  clean phones:     {' '.join(clean)}")
        print(f"  dysfluent phones: {' '.join(dysfluent_tokens)}")
        print(f"  alignment: {n_m} match, {n_s} sub, {n_d} del, {n_i} ins, {n_p} prolong")

        for event in sent_events:
            etype = event["event_type"]
            total_events += 1

            if etype == "lexical_retrieval_failure":
                r = verify_lexical_retrieval_failure(event["target_word"], word_ops, words, wmap)
            elif etype == "phonological_encoding_failure":
                r = verify_phonological_encoding_failure(event["target_word"], word_ops, words, wmap)
            elif etype == "word_finding_pause":
                r = verify_word_finding_pause(event, word_ops, words, wmap, sibling_events=sent_events)
            elif etype == "sentence_abandonment":
                r = verify_sentence_abandonment(event["after_word"], word_ops, words, wmap, dysfluent_tokens)
            else:
                r = {"pass": False, "reason": f"unknown event type: {etype}"}

            status = "PASS" if r["pass"] else "FAIL"
            total_pass += r["pass"]
            detail = ", ".join(f"{k}={v}" for k, v in r.items() if k != "pass")
            print(f"  [{status}] {etype}: {detail}")

    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {total_pass}/{total_events} events verified")


if __name__ == "__main__":
    main()
