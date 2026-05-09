# Add a Sport Evaluator

## When to use
When adding an evaluator for swimming_expert or boxing_expert after the expert itself is built.

## What to tell Claude Code

I need to add an evaluator for [SPORT] following the exact same pattern as the gym evaluator.

Read CLAUDE.md first for full context.

Create `evaluator/[sport]_checks.py` with one function:

```python
def check_[sport]_injury_protocol(workout: dict, injury_context: str) -> list[str]:
    # Same pattern as gym_checks.py:check_injury_protocol
    # If injury_context is empty → return [] immediately
    # Parse avoided exercises from injury_context
    # Fuzzy match against workout exercises
    # Return list of failure messages
```

Then update `evaluator/__init__.py` to add a new evaluate function:

```python
def evaluate_[sport](workout: dict, tier: str, injury_context: str) -> EvalResult:
    failures = []
    failures += check_reps_are_numbers(workout)
    failures += check_no_duplicate_exercises(workout)
    failures += check_training_day_count(workout, tier)
    failures += check_split_type(workout, tier)
    failures += check_[sport]_injury_protocol(workout, injury_context)
    return EvalResult(passed=len(failures) == 0, failures=failures)
```

Then update `agents/[sport]_expert.py` to call `evaluate_[sport]()` instead of `evaluate()` in the retry loop. Follow the exact same retry pattern as gym_expert.py — MAX_RETRIES=2, injury-aware error vs best attempt return.

Do not touch gym_checks.py, gym_expert.py, or any existing evaluator logic.