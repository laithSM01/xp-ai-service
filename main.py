import json
from unittest import result

from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
from chains.workout_suggestion import workout_chain

from fastapi.middleware.cors import CORSMiddleware


load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClientData(BaseModel):
    age: int
    goal: str
    currentXP: int
    currentTier: str
    measurements: list
    xpLogs: list
    currentExercises: list
    completedChallenges: list  # was int, now list of titles
    pastPrograms: list         # NEW

@app.get("/")
def root():
    return {"message": "GymXP AI Service is running"}

@app.post("/suggest/workout")
async def suggest_workout(client: ClientData):
    result = await workout_chain.ainvoke({
        "age": client.age,
        "goal": client.goal,
        "currentXP": client.currentXP,
        "currentTier": client.currentTier,
        "measurements": client.measurements,
        "xpLogs": client.xpLogs,
        "currentExercises": client.currentExercises,
        "completedChallenges": client.completedChallenges,
        "pastPrograms": client.pastPrograms,
    })
    
    try:
        clean = result.content.strip()
        # Gemma often wraps response in ```json ... ```
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        clean = clean.strip()
        parsed = json.loads(clean)
        return {"suggestions": parsed}
    except json.JSONDecodeError:
        return {"error": "AI returned invalid JSON", "raw": result.content}