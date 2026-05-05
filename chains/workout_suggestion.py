import copy
import json
import re
# from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# llm = ChatAnthropic(model="claude-sonnet-4-20250514")
llm = ChatOllama(model="deepseek-r1:8b")

# ═══════════════════════════════════════
# CHAIN 1 — ANALYSIS
# Small focused prompt. Analyzes client data and returns a short JSON.
# Output feeds directly into Chain 2.
# ═══════════════════════════════════════

analysis_prompt = ChatPromptTemplate.from_template("""
You are a fitness data analyst. Analyze the client profile below and return a short JSON summary.

CLIENT PROFILE:
- Age: {age}
- Goal: {goal}
- Current XP: {currentXP}
- Current Tier: {currentTier}
- Height: {height}cm
- Body Shape: {bodyShape}
- Sport Type: {sportType}
- Trainer Notes: {trainerNotes}

MEASUREMENTS (latest first, weight in kg, bodyFat and muscleMass in %):
{measurements}

XP LOGS (recent activity):
{xpLogs}

COMPLETED CHALLENGES:
{completedChallenges}

PAST PROGRAMS:
{pastPrograms}

CURRENT EXERCISES TO AVOID:
{currentExercises}

ANALYSIS RULES:
- Look at the last 3 measurements only
- If body fat is increasing: set fatTrend to "increasing", raise cardioRatio
- If body fat is decreasing: set fatTrend to "decreasing", maintain cardioRatio
- If muscle mass is increasing: set muscleTrend to "increasing"
- If muscle mass is decreasing: set muscleTrend to "decreasing", flag in notes
- Rising weight + rising body fat = warning, mention in notes
- Rising weight + rising muscle = good progress, mention in notes

TIER TRAINING DAYS:
- Beginner: 3
- Novice: 4
- Intermediate: 4
- Advanced: 5
- Elite: 6

GOAL CARDIO RATIO (apply on top of tier defaults):
- Fat Loss: push cardio to maximum for tier
- Muscle Gain: push cardio to minimum for tier
- Strength Building: cardio at minimum
- General Fitness: balanced, follow tier default
- Endurance / Cardio: cardio at maximum
- Body Recomposition: cardio at mid-range
- Athletic Performance: balanced
- Flexibility & Mobility: light cardio only
- Rehabilitation & Recovery: very light cardio only
- Lifestyle & Wellness: moderate cardio

BODY SHAPE RULES:
- athletic: maintain current balance, push intensity
- endomorph: prioritize fat loss, higher cardio regardless of goal
- ectomorph: prioritize muscle gain, minimize cardio
- mesomorph: balanced, follow goal as primary driver
- unknown: follow goal only

SPORT TYPE CONTEXT:
- gym: standard resistance training
- swimming: low impact, focus on endurance and full body
- football: explosive power, agility, conditioning
- rehab: low impact only, avoid heavy compound lifts

You MUST respond ONLY with valid JSON, no extra text, no markdown, no explanation.
Use this exact format:
{{
  "fatTrend": "increasing" | "decreasing" | "stable",
  "muscleTrend": "increasing" | "decreasing" | "stable",
  "trainingDays": 3,
  "cardioRatio": 65,
  "strengthRatio": 35,
  "focus": "cardio priority",
  "notes": "Short observation about the client trend and what to prioritize",
  "currentExercisesToAvoid": ["exercise1", "exercise2"]
}}
""")

analysis_chain = (analysis_prompt | llm).with_config({
    "run_name": "workout_analysis",
    "tags": ["gym-xp", "analysis"],
})


# ═══════════════════════════════════════
# CHAIN 2 — GENERATION
# Takes analysis output from Chain 1.
# Generates the weekly schedule JSON only.
# ═══════════════════════════════════════

generation_prompt = ChatPromptTemplate.from_template("""
You are an expert fitness coach. Generate a weekly workout schedule based on the analysis below.

CLIENT:
- Age: {age}
- Goal: {goal}
- Tier: {currentTier}
- Height: {height}cm
- Body Shape: {bodyShape}
- Sport Type: {sportType}
- Trainer Notes: {trainerNotes}

ANALYSIS SUMMARY:
- Fat Trend: {fatTrend}
- Muscle Trend: {muscleTrend}
- Training Days: {trainingDays}
- Cardio Ratio: {cardioRatio}%
- Strength Ratio: {strengthRatio}%
- Focus: {focus}
- Coach Notes: {notes}

EXERCISES TO AVOID (do not use any of these):
{currentExercisesToAvoid}

WORKOUT TYPE PER TIER:
- Beginner / Novice: "Full Body"
- Intermediate: alternate "Upper Body" and "Lower Body"
- Advanced: rotate "Push", "Pull", "Legs"
- Elite: specific muscle groups (e.g. "Chest & Triceps", "Back & Biceps", "Legs", "Shoulders", "Arms", "Core & Cardio")

EXERCISE COUNT PER DAY:
- Beginner: 3-4 exercises
- Novice: 4-5 exercises
- Intermediate: 4-6 exercises
- Advanced: 5-7 exercises
- Elite: 6-8 exercises

OUTPUT RULES:
1. Generate exactly {trainingDays} training days + enough rest days to total 7 days
2. Cardio appears as an exercise row inside the day, NOT as a separate day
3. Every exercise must have: "name", "sets", "reps", "notes"
4. "reps" is ALWAYS a number — never a string. For cardio, reps = minutes as a number (e.g. 20 not "20 min")
5. Do NOT repeat the same exercise name across different training days
6. 6. Each exercise name must be unique across the entire week. Before writing each exercise, check all exercises already written in previous days. If the name already exists, choose a different exercise.
7. If fatTrend is "increasing", mention it in at least one exercise note
8. Exercise notes must include: sets/rest info AND why it fits this client
9. If trainerNotes is not empty, treat it as high priority instruction — override defaults if needed
10. Body shape must influence exercise selection — endomorph gets more cardio, ectomorph gets heavier compound lifts, athletic gets intensity focus

CRITICAL: The JSON keys must be exactly "weeklySchedule", "day", "type", "exercises". Do not use "days", "rest", or any other key names.

CARDIO FORMAT EXAMPLE:
{{ "name": "Treadmill Cardio", "sets": 1, "reps": 20, "notes": "20 min moderate pace. Cardio finisher matching fat loss goal." }}

You MUST respond ONLY with valid JSON, no extra text, no markdown, no explanation.
Use this exact format:
{{
  "title": "AI Program — {currentTier} | {goal}",
  "weeklySchedule": [
    {{
      "day": 1,
      "type": "Full Body",
      "exercises": [
        {{ "name": "Exercise Name", "sets": 3, "reps": 10, "notes": "Structural info + why this fits the client" }},
        {{ "name": "Treadmill Cardio", "sets": 1, "reps": 20, "notes": "20 min cardio. Placement + reason." }}
      ]
    }},
    {{
      "day": 2,
      "type": "Rest",
      "exercises": []
    }}
  ]
}}
""")

generation_chain = (generation_prompt | llm).with_config({
    "run_name": "workout_generation",
    "tags": ["gym-xp", "generation"],
})


# ═══════════════════════════════════════
# TIER RULES
# ═══════════════════════════════════════

TIER_TRAINING_DAYS = {
    "beginner": 3,
    "novice": 4,
    "intermediate": 4,
    "advanced": 5,
    "elite": 6,
}


# ═══════════════════════════════════════
# ENFORCE RULES — post-processing safety net
# ═══════════════════════════════════════

def enforce_rules(schedule: dict, tier: str) -> dict:
    tier_key = tier.strip().lower()
    expected = TIER_TRAINING_DAYS.get(tier_key, 4)

    # Deduplicate exercises across all training days
    seen_exercises = set()
    for day in schedule["weeklySchedule"]:
        if day["type"] != "Rest":
            unique = []
            for ex in day["exercises"]:
                name = re.sub(r'[^a-z0-9 ]', '', ex["name"].strip().lower())
                name = re.sub(r'\s+', ' ', name).strip()
                if name not in seen_exercises:
                    seen_exercises.add(name)
                    unique.append(ex)
            day["exercises"] = unique

    # Fix training day count
    training_days = [d for d in schedule["weeklySchedule"] if d["type"] != "Rest"]
    rest_days = [d for d in schedule["weeklySchedule"] if d["type"] == "Rest"]

    if len(training_days) > expected:
        training_days = training_days[:expected]
    elif len(training_days) < expected:
        while len(training_days) < expected:
            clone = copy.deepcopy(training_days[-1])
            clone["exercises"] = []
            training_days.append(clone)

    combined = training_days + rest_days
    for i, day in enumerate(combined):
        day["day"] = i + 1
    schedule["weeklySchedule"] = combined

    return schedule