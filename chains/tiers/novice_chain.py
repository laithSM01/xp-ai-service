from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from .base import OUTPUT_RULES

llm = ChatOllama(model="deepseek-r1:8b")

_PROMPT = """
You are an expert fitness coach specializing in novice trainees. Generate a weekly workout schedule for the client below.

CLIENT:
- Age: {age}
- Goal: {goal}
- Tier: {currentTier}
- Height: {height}cm
- Body Shape: {bodyShape}
- Sport Types: {sportTypes}
- Trainer Notes: {trainerNotes}

ANALYSIS SUMMARY:
- Fat Trend: {fatTrend}
- Muscle Trend: {muscleTrend}
- Training Days: {trainingDays}
- Cardio Ratio: {cardioRatio}%
- Strength Ratio: {strengthRatio}%
- Focus: {focus}
- Coach Notes: {notes}

EXERCISES TO AVOID (do not include any of these):
{currentExercisesToAvoid}

NOVICE PROGRAM RULES:
- Split: Full Body — every training day targets the entire body
- Training days: exactly 4 per week
- Exercises per day: 4-5 only — do not exceed 5
- Equipment: introduce dumbbells and barbells for basic movements; machines are still acceptable
- Acceptable barbell movements: barbell bench press, barbell overhead press, Romanian deadlift — no Olympic lifts
- Rest between sets: 75-90 seconds — always state rest time in exercise notes
- Sets per exercise: 3 sets
- Intensity: moderate to moderately high — technique should be solid before adding load
- Progress from machines toward free weights where appropriate

""" + OUTPUT_RULES

novice_prompt = ChatPromptTemplate.from_template(_PROMPT)

novice_chain = (novice_prompt | llm).with_config({
    "run_name": "workout_generation_novice",
    "tags": ["gym-xp", "generation", "novice"],
})
