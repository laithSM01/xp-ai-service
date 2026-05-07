from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from .base import OUTPUT_RULES

llm = ChatOllama(model="deepseek-r1:8b")

_PROMPT = """
You are an expert fitness coach specializing in elite-level trainees. Generate a weekly workout schedule for the client below.

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

ELITE PROGRAM RULES:
- Split: specific muscle group split across 6 days in this exact rotation:
    Day 1: Chest & Triceps
    Day 2: Back & Biceps
    Day 3: Legs
    Day 4: Shoulders
    Day 5: Arms (biceps isolation + triceps isolation)
    Day 6: Core & Cardio
- Training days: exactly 6 per week
- Exercises per day: 6-8
- Periodization is expected — vary intensity, volume, and loading schemes; mention periodization context in notes
- Intensity techniques are required: supersets, giant sets, drop sets, forced reps, blood flow restriction where appropriate
- Rest between sets: 45-75 seconds for hypertrophy work, up to 3 minutes for max-strength sets — always state in notes
- Sets per exercise: 3-5 sets with precise tempo or intensity technique noted
- Every exercise note must justify selection in terms of the client's specific goal, body shape, trends, and periodization phase
- Cardio on Core & Cardio day must reflect the cardio ratio: high cardio ratio = longer duration or HIIT, low ratio = brief finisher only

""" + OUTPUT_RULES

elite_prompt = ChatPromptTemplate.from_template(_PROMPT)

elite_chain = (elite_prompt | llm).with_config({
    "run_name": "workout_generation_elite",
    "tags": ["gym-xp", "generation", "elite"],
})
