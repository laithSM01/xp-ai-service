from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from .base import OUTPUT_RULES

llm = ChatOllama(model="deepseek-r1:8b")

_PROMPT = """
You are an expert fitness coach specializing in advanced trainees. Generate a weekly workout schedule for the client below.

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

ADVANCED PROGRAM RULES:
- Split: Push / Pull / Legs rotation across 5 days (e.g. Push, Pull, Legs, Push, Pull)
- Training days: exactly 5 per week
- Exercises per day: 5-7
- Structure: compound movements first, isolation and accessory work after
- Equipment: full free weight access — barbell compounds, dumbbells, cables, machines
- Advanced techniques are expected and allowed: drop sets, rest-pause, tempo manipulation, mechanical drop sets
- Rest between sets: 60-90 seconds for isolation work, 2-3 minutes for heavy compound sets — state clearly in notes
- Sets per exercise: 3-5 sets depending on technique and position in session
- Intensity: high — maximize mechanical tension and metabolic stress; notes must justify exercise selection and technique
- Push days: chest, shoulders, triceps; Pull days: back, biceps; Legs: quads, hamstrings, glutes, calves

""" + OUTPUT_RULES

advanced_prompt = ChatPromptTemplate.from_template(_PROMPT)

advanced_chain = (advanced_prompt | llm).with_config({
    "run_name": "workout_generation_advanced",
    "tags": ["gym-xp", "generation", "advanced"],
})
