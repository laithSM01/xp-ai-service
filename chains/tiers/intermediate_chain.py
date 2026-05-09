from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from .base import OUTPUT_RULES

llm = ChatOllama(model="deepseek-r1:14b")

_PROMPT = """
You are an expert fitness coach specializing in intermediate trainees. Generate a weekly workout schedule for the client below.

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

INTERMEDIATE PROGRAM RULES:
- Split: alternate Upper Body / Lower Body across training days (e.g. Upper, Lower, Upper, Lower)
- Training days: exactly 4 per week
- Exercises per day: 4-6
- Equipment: free weights are standard — barbell and dumbbell compounds, cables, machines as accessory
- Supersets are optional — use them to increase density when cardio ratio is high or time is limited
- Rest between sets: 60-90 seconds for hypertrophy work, up to 2 minutes for heavy compound sets
- Sets per exercise: 3-4 sets
- Intensity: moderately high — progressive overload is the primary driver; notes must reflect this
- Separate upper and lower days clearly; do not mix push and pull into random order

""" + OUTPUT_RULES

intermediate_prompt = ChatPromptTemplate.from_template(_PROMPT)

intermediate_chain = (intermediate_prompt | llm).with_config({
    "run_name": "workout_generation_intermediate",
    "tags": ["gym-xp", "generation", "intermediate"],
})
