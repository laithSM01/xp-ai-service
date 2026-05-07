from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from .base import OUTPUT_RULES

llm = ChatOllama(model="deepseek-r1:8b")

_PROMPT = """
You are an expert fitness coach specializing in beginners. Generate a weekly workout schedule for the client below.

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

BEGINNER PROGRAM RULES:
- Split: Full Body — every training day targets the entire body
- Training days: exactly 3 per week
- Exercises per day: 3-4 only — do not exceed 4
- Equipment: machines and bodyweight only; no Olympic lifts, no barbell Olympic movements (no cleans, snatches, jerks)
- No complex compound lifts — avoid barbell back squat, conventional deadlift, barbell row
- Safe compound options: goblet squat, dumbbell bench press, leg press machine, cable row, lat pulldown
- Rest between sets: 90-120 seconds — always state rest time in exercise notes
- Sets per exercise: 2-3 sets to prevent overtraining
- Intensity: moderate — correct movement pattern and learning form is the primary goal

""" + OUTPUT_RULES

beginner_prompt = ChatPromptTemplate.from_template(_PROMPT)

beginner_chain = (beginner_prompt | llm).with_config({
    "run_name": "workout_generation_beginner",
    "tags": ["gym-xp", "generation", "beginner"],
})
