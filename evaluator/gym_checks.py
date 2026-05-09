import re


def _normalize_name(name: str) -> str:
    name = re.sub(r'[^a-z0-9 ]', '', name.strip().lower())
    return re.sub(r'\s+', ' ', name).strip()


def check_injury_protocol(workout: dict, injury_context: str) -> list[str]:
    if not injury_context or not injury_context.strip():
        return []

    avoided: list[str] = []
    for line in injury_context.splitlines():
        if "avoid" not in line.lower():
            continue
        # Extract the exercise name that follows "avoid"
        parts = re.split(r'avoid[:\s]+', line, flags=re.IGNORECASE, maxsplit=1)
        if len(parts) > 1:
            candidate = parts[1].strip().rstrip('.,:;')
            normalized = _normalize_name(candidate)
            if normalized:
                avoided.append(normalized)

    if not avoided:
        return []

    failures = []
    for day in workout.get("weeklySchedule", []):
        if day.get("type") == "Rest":
            continue
        for ex in day.get("exercises", []):
            ex_name = _normalize_name(ex.get("name", ""))
            for avoided_ex in avoided:
                if avoided_ex and avoided_ex in ex_name:
                    failures.append(
                        f"Day {day.get('day')}: exercise '{ex.get('name')}' is contraindicated "
                        f"(injury protocol: avoid '{avoided_ex}')"
                    )
                    break
    return failures
