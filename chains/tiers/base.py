OUTPUT_RULES = """
OUTPUT RULES:
1. Generate exactly {trainingDays} training days + enough rest days to total 7
2. Cardio appears as an exercise row inside the day, NOT a separate day
3. Every exercise must have: name, sets, reps, notes
4. reps is ALWAYS a number — never a string. For cardio reps = minutes as a number (e.g. 20 not "20 min")
5. Do NOT repeat the same exercise name across different training days
6. If fatTrend is "increasing", mention it in at least one exercise note
7. Exercise notes must include sets/rest info AND why it fits this client
8. If trainerNotes is not empty, treat it as high priority — override defaults if needed
9. Body shape must influence exercise selection: endomorph gets more cardio, ectomorph gets heavier compound lifts, athletic gets intensity focus
10. CRITICAL: JSON keys must be exactly "weeklySchedule", "day", "type", "exercises"
11. Respond ONLY with valid JSON, no extra text, no markdown

Use this exact format:
{{
  "title": "AI Program — {currentTier} | {goal}",
  "weeklySchedule": [
    {{
      "day": 1,
      "type": "Full Body",
      "exercises": [
        {{"name": "Exercise Name", "sets": 3, "reps": 10, "notes": "Structural info + why this fits the client"}},
        {{"name": "Treadmill Cardio", "sets": 1, "reps": 20, "notes": "20 min cardio. Placement + reason."}}
      ]
    }},
    {{
      "day": 2,
      "type": "Rest",
      "exercises": []
    }}
  ]
}}
"""
