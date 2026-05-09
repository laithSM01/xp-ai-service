# Add a New Test Profile

## When to use
When you want to test a new client type (new tier, goal, injury, sport)

## What to tell Claude Code

Add a new test profile to tests/test_workout_chain.py PROFILES list.

Profile details:
- age: [AGE]
- goal: [GOAL]
- currentTier: [TIER]
- injuryNotes: [INJURY OR ""]
- sportTypes: [LIST]
- measurements: [LIST OR []]
- expected min_days: [N]
- expected max_days: [N]
- id: [descriptive_snake_case_id]

Follow the exact same pytest.param structure as existing profiles.
Do not change _run(), _assert_valid_program(), or any existing profile.