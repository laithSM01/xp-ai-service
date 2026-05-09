import json
import os
import re

KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(__file__), "knowledge", "gym_injuries.md")

KNOWN_CATEGORIES = [
    "knee pain",
    "ACL injury/recovery",
    "lower back pain",
    "shoulder impingement",
    "rotator cuff injury",
    "tennis elbow",
    "wrist pain",
    "hip flexor strain",
    "hamstring strain",
    "IT band syndrome",
    "ankle sprain",
    "neck pain",
    "chest/rib injury",
    "herniated disc",
    "post-surgery general recovery",
]


def _strip_think(content: str) -> str:
    clean = content.strip()
    if "<think>" in clean:
        clean = clean.split("</think>")[-1].strip()
    return clean.strip()


async def classify_injury(injury_text: str, llm) -> list[str]:
    categories_list = json.dumps(KNOWN_CATEGORIES)
    prompt = (
        f"You are a fitness injury classifier. Given the injury description below, "
        f"return a JSON array of matching category names from this exact list:\n\n"
        f"{categories_list}\n\n"
        f'Injury description: "{injury_text}"\n\n'
        f"Return ONLY a valid JSON array of matched strings. "
        f"If nothing matches, return []. No explanation, no markdown, no extra text."
    )

    result = await llm.ainvoke(prompt)
    raw = _strip_think(result.content)

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    return []


def fetch_injury_context(categories: list[str]) -> str:
    if not categories:
        return ""

    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    sections = re.split(r"\n(?=## )", content)
    matched = []

    for section in sections:
        if not section.startswith("## "):
            continue
        heading_line = section.split("\n")[0].lower().replace("## ", "").strip()
        for cat in categories:
            if cat.lower().strip() in heading_line or heading_line in cat.lower().strip():
                matched.append(section.strip())
                break

    return "\n\n".join(matched)
