import copy
import json
import pytest
from chains.workout_suggestion import workout_chain, enforce_rules


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse(result, tier: str) -> dict:
    raw = result.content.strip()
    if "<think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])
    return enforce_rules(parsed, tier)


def _run(input_data: dict) -> dict:
    result = workout_chain.invoke(input_data)
    return _parse(result, input_data["currentTier"])


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


# ── Test profiles ─────────────────────────────────────────────────────────────

PROFILES = [
    pytest.param(
        # Profile 1 -- Beginner, Fat Loss, 0 XP, age 30, no history
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
        # Profile 2 -- Novice, Muscle Gain, 500 XP, age 25, no history
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
        # Profile 3 -- Intermediate, Strength Building, 1500 XP, age 35, 2 past programs
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
                    "exercises": [
                        "Barbell Squat",
                        "Bench Press",
                        "Barbell Row",
                        "Overhead Press",
                    ],
                },
                {
                    "title": "Strength Phase 2",
                    "exercises": [
                        "Romanian Deadlift",
                        "Incline Bench Press",
                        "Lat Pulldown",
                        "Dumbbell Shoulder Press",
                    ],
                },
            ],
        },
        4, 4,
        id="intermediate_strength_age35_2programs",
    ),
    pytest.param(
        # Profile 4 -- Advanced, Body Recomposition, 2500 XP, age 28,
        #              measurements show body fat trending upward (warning sign)
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
        # Profile 5 -- Edge case: Beginner, Fat Loss, 0 XP, age 58, single measurement
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
