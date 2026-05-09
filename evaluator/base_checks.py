import re
from chains.workout_suggestion import TIER_TRAINING_DAYS

_TIER_SPLIT_RULES = {
    "beginner":     {"allowed": {"full body"}, "label": "Full Body"},
    "novice":       {"allowed": {"full body"}, "label": "Full Body"},
    "intermediate": {"allowed": {"upper body", "lower body"}, "label": "Upper Body / Lower Body"},
    "advanced":     {"allowed": {"push", "pull", "legs"}, "label": "Push / Pull / Legs"},
    # elite: any specific muscle group name is valid — no fixed set
}


def _normalize_name(name: str) -> str:
    name = re.sub(r'[^a-z0-9 ]', '', name.strip().lower())
    return re.sub(r'\s+', ' ', name).strip()


def check_reps_are_numbers(workout: dict) -> list[str]:
    failures = []
    for day in workout.get("weeklySchedule", []):
        if day.get("type") == "Rest":
            continue
        for ex in day.get("exercises", []):
            reps = ex.get("reps")
            if not isinstance(reps, (int, float)):
                failures.append(
                    f"Day {day.get('day')}: exercise '{ex.get('name')}' has non-numeric reps: {reps!r}"
                )
    return failures


def check_no_duplicate_exercises(workout: dict) -> list[str]:
    seen: set[str] = set()
    failures = []
    for day in workout.get("weeklySchedule", []):
        if day.get("type") == "Rest":
            continue
        for ex in day.get("exercises", []):
            name = _normalize_name(ex.get("name", ""))
            if name in seen:
                failures.append(
                    f"Day {day.get('day')}: duplicate exercise '{ex.get('name')}'"
                )
            else:
                seen.add(name)
    return failures


def check_training_day_count(workout: dict, tier: str) -> list[str]:
    expected = TIER_TRAINING_DAYS.get(tier.lower())
    if expected is None:
        return []
    training_days = [d for d in workout.get("weeklySchedule", []) if d.get("type") != "Rest"]
    actual = len(training_days)
    if actual != expected:
        return [f"Training day count is {actual}, expected {expected} for tier '{tier}'"]
    return []


def check_split_type(workout: dict, tier: str) -> list[str]:
    rule = _TIER_SPLIT_RULES.get(tier.lower())
    if rule is None:
        # elite tier — any named muscle group is acceptable
        return []

    allowed = rule["allowed"]
    label = rule["label"]
    failures = []
    for day in workout.get("weeklySchedule", []):
        if day.get("type") == "Rest":
            continue
        day_type = day.get("type", "").strip().lower()
        if day_type not in allowed:
            failures.append(
                f"Day {day.get('day')}: type '{day.get('type')}' is not valid for {tier} tier "
                f"(expected {label})"
            )
    return failures
