import json
import re

try:
    from json_repair import repair_json
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False

from langchain_ollama import ChatOllama
from chains.workout_suggestion import analysis_chain, enforce_rules
from chains.tiers import get_generation_chain
from rag import get_injury_context

llm = ChatOllama(model="deepseek-r1:8b")


def _calculate_body_shape(weight: float, height: float, body_fat: float, muscle_mass: float) -> str:
    bmi = weight / (height / 100) ** 2
    if muscle_mass > 40 and body_fat < 15:
        return "athletic"
    elif body_fat > 30 and muscle_mass < 30:
        return "endomorph"
    elif bmi < 18.5:
        return "ectomorph"
    else:
        return "mesomorph"


def _strip_raw(content: str) -> str:
    clean = content.strip()
    if "<think>" in clean:
        clean = clean.split("</think>")[-1].strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    return clean.strip()


def _parse_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass

    if HAS_JSON_REPAIR:
        try:
            return json.loads(repair_json(raw))
        except Exception:
            pass

    raise ValueError("Could not parse JSON from LLM output")


async def run(client) -> dict:
    tier = client.currentTier.capitalize()

    latest = client.measurements[0] if client.measurements else None
    body_shape = _calculate_body_shape(
        latest["weight"], client.height,
        latest["bodyFat"], latest["muscleMass"]
    ) if latest else "unknown"

    injury_context = await get_injury_context(client.injuryNotes or "", llm)

    analysis_result = await analysis_chain.ainvoke({
        "age": client.age,
        "goal": client.goal,
        "currentXP": client.currentXP,
        "currentTier": tier,
        "measurements": client.measurements,
        "xpLogs": client.xpLogs,
        "currentExercises": client.currentExercises,
        "completedChallenges": client.completedChallenges,
        "pastPrograms": client.pastPrograms,
        "bodyShape": body_shape,
        "height": client.height,
        "sportTypes": ", ".join(client.sportTypes),
        "trainerNotes": client.trainerNotes or "",
        "injuryContext": injury_context,
    })

    try:
        analysis_raw = _strip_raw(analysis_result.content)
        analysis = _parse_json(analysis_raw)
    except (ValueError, Exception) as e:
        raise ValueError(f"Analysis chain returned invalid JSON: {analysis_result.content}") from e

    generation_chain = get_generation_chain(tier)

    MAX_RETRIES = 2
    has_injury = bool(
        client.injuryNotes
        and client.injuryNotes.strip()
        and client.injuryNotes.strip().lower() != "none"
    )

    last_parsed = None
    last_failures = []

    for attempt in range(MAX_RETRIES + 1):
        generation_result = await generation_chain.ainvoke({
            "age": client.age,
            "goal": client.goal,
            "currentTier": tier,
            "fatTrend": analysis.get("fatTrend", "stable"),
            "muscleTrend": analysis.get("muscleTrend", "stable"),
            "trainingDays": analysis.get("trainingDays", 3),
            "cardioRatio": analysis.get("cardioRatio", 50),
            "strengthRatio": analysis.get("strengthRatio", 50),
            "focus": analysis.get("focus", "balanced"),
            "notes": analysis.get("notes", ""),
            "currentExercisesToAvoid": analysis.get("currentExercisesToAvoid", []),
            "bodyShape": body_shape,
            "sportTypes": ", ".join(client.sportTypes),
            "trainerNotes": client.trainerNotes or "",
            "height": client.height,
        })

        try:
            generation_raw = _strip_raw(generation_result.content)
            parsed = _parse_json(generation_raw)
            enforced = enforce_rules(parsed, tier)
        except (ValueError, Exception) as e:
            last_failures = [f"JSON parse failed: {str(e)}"]
            last_parsed = None
            continue

        from evaluator import evaluate
        result = evaluate(enforced, tier.lower(), injury_context)

        if result.passed:
            return enforced

        last_parsed = enforced
        last_failures = result.failures

        if attempt < MAX_RETRIES:
            continue

    if has_injury and last_failures:
        raise ValueError(
            f"Could not generate a safe workout after {MAX_RETRIES + 1} attempts. "
            f"Issues: {last_failures}"
        )

    if last_parsed is not None:
        return last_parsed

    raise ValueError("Generation failed after all retries")
