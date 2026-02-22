import os
import re
#os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", "/opt/homebrew/lib/libespeak-ng.dylib")
os.environ.setdefault(
    "PHONEMIZER_ESPEAK_BINARY",
    "/home/liharrison/local/bin/espeak-ng"
)

from phonemizer import phonemize

def phonemize_text(text, language='en-us'):
    """Convert text to espeak-style IPA (word-grouped, with stress marks).

    Output format matches what VITS was trained on via english_cleaners2:
    - spaces separate words only, phones within a word are concatenated
    - stress markers (ˈ ˌ) are embedded within word tokens
    - sentences are separated by ' | ' for downstream chunking

    Example: "I woke up. She ran." -> "aɪ wˈoʊk ˌʌp | ʃiː ɹˈæn"
    """
    # Split on sentence-ending punctuation to preserve sentence boundaries
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    results = []
    for sentence in sentences:
        ipa = phonemize(
            sentence,
            language=language,
            backend='espeak',
            strip=True,
            preserve_punctuation=False,
            with_stress=True,
        )
        ipa = re.sub(r'\s+', ' ', ipa).strip()
        if ipa:
            results.append(ipa)
    return ' | '.join(results)
