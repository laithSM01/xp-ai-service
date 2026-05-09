from dataclasses import dataclass
from evaluator.base_checks import (
    check_reps_are_numbers,
    check_no_duplicate_exercises,
    check_training_day_count,
    check_split_type,
)
from evaluator.gym_checks import check_injury_protocol


@dataclass
class EvalResult:
    passed: bool
    failures: list[str]


def evaluate(workout: dict, tier: str, injury_context: str) -> EvalResult:
    failures = []
    failures += check_reps_are_numbers(workout)
    failures += check_no_duplicate_exercises(workout)
    failures += check_training_day_count(workout, tier)
    failures += check_split_type(workout, tier)
    failures += check_injury_protocol(workout, injury_context)
    return EvalResult(passed=len(failures) == 0, failures=failures)
