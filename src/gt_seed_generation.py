import json
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Master list of connected speech prompts
PROMPTS = [
    "Tell me about the best trip you ever took",
    "Tell me about your favorite holiday as a child",
    "Tell me about your worst childhood memory",
    "Tell me about when you retired",
    "Tell me about the worst trip you ever took",
    "Tell me about a happy childhood memory",
    "Tell me about when you had your first child",
    "Tell me about what you like about where you live",
    "Tell me about when you got married",
    "Tell me about your first job",
    "Tell me about how you met your husband/wife/partner",
    "Tell me about a time you were really scared/embarrassed/angry",
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "gt_seed")

PERSONAS = [
    "a reserved, soft-spoken person who speaks slowly and carefully, often pausing to find the right words",
    "an energetic, talkative person who jumps between ideas quickly and uses lots of filler words and laughter",
    "a wistful, emotional person who speaks with warmth but trails off and backtracks frequently",
    "a pragmatic, matter-of-fact person who tells things plainly and moves on without much sentiment",
    "a storyteller who is very detailed and vivid, using rich descriptions and dramatic pacing",
]


def generate_gt(prompt: str, prompt_index: int, n: int) -> str:
    """
    Generate n naturalistic connected speech responses for a given prompt using the Gemini API.
    All n responses are generated in a single call and saved together in one JSON file.

    Args:
        prompt: The conversational prompt to respond to.
        prompt_index: Index of the prompt in the PROMPTS list (0-based).
        n: Number of distinct speech samples to generate.

    Returns:
        Path to the written JSON file.
    """
    system_message = (
        "You are simulating naturalistic connected speech from older adults being interviewed. You should talk like a healthy patient with no speech issues. "
        "Each response must be written in first person as if recounting a personal memory or experience. "
        "Responses should be healthy, fluent speech."
        "Each response must be clearly distinct from the others in personality, tone, vocabulary, "
        "emotional register, and the specific memory or detail recalled. "
        "Write around 5 sentences per response. Do not use bullet points or headers. "
        "Write only the spoken responses, nothing else."
    )

    persona_instructions = "\n".join(
        f"Response {i + 1}: Speak as {PERSONAS[i % len(PERSONAS)]}."
        for i in range(n)
    )

    user_message = (
        f"{prompt}.\n\n"
        f"Generate {n} completely different responses to this prompt, each from a distinct speaker persona:\n"
        f"{persona_instructions}\n\n"
        f"Separate each response with the exact marker: <<<RESPONSE_BREAK>>>"
    )

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        config=types.GenerateContentConfig(
            system_instruction=system_message,
            temperature=1.2,
        ),
        contents=user_message,
    )

    raw = response.text
    speeches = [s.strip() for s in raw.split("<<<RESPONSE_BREAK>>>")]
    speeches = [s for s in speeches if s]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    filename = f"gt_{prompt_index}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)

    output = {"prompt": prompt}
    for i, speech in enumerate(speeches, start=1):
        output[f"speech{i}"] = speech

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved: {filepath}")
    return filepath


if __name__ == "__main__":
    for i in PROMPTS:
        generate_gt(i, prompt_index=PROMPTS.index(i), n=5)
