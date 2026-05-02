import json
import pytest
from chains.workout_suggestion import analysis_chain, generation_chain, enforce_rules

try:
    from json_repair import repair_json
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False


# ── Helpers ───────────────────────────────────────────────────────────────────

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

    raise ValueError(f"Could not parse JSON from LLM output: {raw[:200]}")


def _run(input_data: dict) -> dict:
    tier = input_data["currentTier"].capitalize()

    # Chain 1 — Analyze
    analysis_result = analysis_chain.invoke({
        "age": input_data["age"],
        "goal": input_data["goal"],
        "currentXP": input_data["currentXP"],
        "currentTier": tier,
        "measurements": input_data["measurements"],
        "xpLogs": input_data["xpLogs"],
        "currentExercises": input_data["currentExercises"],
        "completedChallenges": input_data["completedChallenges"],
        "pastPrograms": input_data["pastPrograms"],
    })
    analysis = _parse_json(_strip_raw(analysis_result.content))

    # Chain 2 — Generate
    generation_result = generation_chain.invoke({
        "age": input_data["age"],
        "goal": input_data["goal"],
        "currentTier": tier,
        "fatTrend": analysis.get("fatTrend", "stable"),
        "muscleTrend": analysis.get("muscleTrend", "stable"),
        "trainingDays": analysis.get("trainingDays", 3),
        "cardioRatio": analysis.get("cardioRatio", 50),
        "strengthRatio": analysis.get("strengthRatio", 50),
        "focus": analysis.get("focus", "balanced"),
        "notes": analysis.get("notes", ""),
        "currentExercisesToAvoid": analysis.get("currentExercisesToAvoid", []),
    })
    parsed = _parse_json(_strip_raw(generation_result.content))
    print(json.dumps(parsed, indent=2))
    return enforce_rules(parsed, tier)


def _assert_valid_program(data: dict, min_days: int, max_days: int) -> None:
    # 1. Top-level keys
    assert "title" in data, "Missing 'title'"
    assert "weeklySchedule" in data, "Missing 'weeklySchedule'"

    schedule = data["weeklySchedule"]

    # 2. Every day has required keys; rest days have empty exercises
    for entry in schedule:
        for key in ("day", "type", "exercises"):
            assert key in entry, f"Day entry missing '{key}': {entry}"
        if entry["type"] == "Rest":
            assert entry["exercises"] == [], (
                f"Rest day {entry['day']} must have an empty exercises array"
            )

    # 3. Training day count matches tier rules
    training_days = [d for d in schedule if d["type"] != "Rest"]
    count = len(training_days)
    assert min_days <= count <= max_days, (
        f"Expected {min_days}--{max_days} training days, got {count}"
    )

    # 4. No exercise name appears in more than one training day
    seen: dict[str, int] = {}
    for day in training_days:
        for ex in day["exercises"]:
            name = ex["name"].strip().lower()
            assert name not in seen, (
                f"Exercise '{ex['name']}' appears on day {seen[name]} "
                f"and again on day {day['day']}"
            )
            seen[name] = day["day"]

    # 5. Reps is always a number
    for day in training_days:
        for ex in day["exercises"]:
            assert isinstance(ex["reps"], (int, float)), (
                f"Exercise '{ex['name']}' has non-numeric reps: {ex['reps']}"
            )


# ── Test profiles ─────────────────────────────────────────────────────────────

PROFILES = [
    pytest.param(
        {
            "age": 30,
            "goal": "Fat Loss",
            "currentXP": 0,
            "currentTier": "Beginner",
            "measurements": [],
            "xpLogs": [],
            "currentExercises": [],
            "completedChallenges": [],
            "pastPrograms": [],
        },
        3, 3,
        id="beginner_fat_loss_age30",
    ),
    pytest.param(
        {
            "age": 25,
            "goal": "Muscle Gain",
            "currentXP": 500,
            "currentTier": "Novice",
            "measurements": [],
            "xpLogs": [],
            "currentExercises": [],
            "completedChallenges": [],
            "pastPrograms": [],
        },
        3, 4,
        id="novice_muscle_gain_age25",
    ),
    pytest.param(
        {
            "age": 35,
            "goal": "Strength Building",
            "currentXP": 1500,
            "currentTier": "Intermediate",
            "measurements": [],
            "xpLogs": [],
            "currentExercises": [],
            "completedChallenges": [],
            "pastPrograms": [
                {
                    "title": "Strength Phase 1",
                    "exercises": ["Barbell Squat", "Bench Press", "Barbell Row", "Overhead Press"],
                },
                {
                    "title": "Strength Phase 2",
                    "exercises": ["Romanian Deadlift", "Incline Bench Press", "Lat Pulldown", "Dumbbell Shoulder Press"],
                },
            ],
        },
        4, 4,
        id="intermediate_strength_age35_2programs",
    ),
    pytest.param(
        {
            "age": 28,
            "goal": "Body Recomposition",
            "currentXP": 2500,
            "currentTier": "Advanced",
            "measurements": [
                {"date": "2026-04-01", "weight": 88.5, "bodyFat": 26.0, "muscleMass": 38.0},
                {"date": "2026-03-01", "weight": 87.0, "bodyFat": 24.5, "muscleMass": 38.5},
                {"date": "2026-02-01", "weight": 86.0, "bodyFat": 23.0, "muscleMass": 39.0},
            ],
            "xpLogs": [],
            "currentExercises": [],
            "completedChallenges": [],
            "pastPrograms": [],
        },
        5, 6,
        id="advanced_recomp_fat_increasing_age28",
    ),
    pytest.param(
        {
            "age": 58,
            "goal": "Fat Loss",
            "currentXP": 0,
            "currentTier": "Beginner",
            "measurements": [
                {"date": "2026-04-01", "weight": 82.0, "bodyFat": 31.0, "muscleMass": 32.0},
            ],
            "xpLogs": [],
            "currentExercises": [],
            "completedChallenges": [],
            "pastPrograms": [],
        },
        3, 3,
        id="beginner_fat_loss_age58_edge",
    ),
]


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("input_data,min_days,max_days", PROFILES)
def test_workout_chain(input_data: dict, min_days: int, max_days: int) -> None:
    data = _run(input_data)
    _assert_valid_program(data, min_days, max_days)