# from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# llm = ChatAnthropic(model="claude-opus-4-6")
llm = ChatOllama(model="llama3.2")

prompt = ChatPromptTemplate.from_template("""
You are an expert fitness coach inside a training app called GymXP.
Based on the client data below, suggest 3 personalized workout exercises.

Client Profile:
- Age: {age}
- Goal: {goal}
- Current XP: {currentXP}
- Current Tier: {currentTier}

Recent Body Measurements (latest first):
{measurements}

Recent XP Activity:
{xpLogs}

Current Program Exercises:
{currentExercises}

Completed Challenges: {completedChallenges}

Instructions:
- Tailor suggestions to their tier and goal
- Consider their measurement trend
- Suggest exercises they aren't already doing
- For each exercise include: name, sets, reps, and one sentence why it fits this client

You must respond ONLY with valid JSON, no extra text, no markdown, no explanation.
Use this exact format:
{{
  "title": "AI Program — <tier> <goal>",
  "exercises": [
    {{ "name": "Exercise Name", "sets": 3, "reps": 10, "notes": "Why this fits the client" }},
    {{ "name": "Exercise Name", "sets": 4, "reps": 12, "notes": "Why this fits the client" }},
    {{ "name": "Exercise Name", "sets": 3, "reps": 15, "notes": "Why this fits the client" }}
  ]
}}
""")

workout_chain  = prompt | llm