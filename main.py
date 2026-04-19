import json

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
    completedChallenges: int

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
    })
    
    try:
        parsed = json.loads(result.content)
        return {"suggestions": parsed}
    except json.JSONDecodeError:
        return {"suggestions": result.content}