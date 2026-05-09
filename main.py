import json
from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
from chains.workout_suggestion import analysis_chain, enforce_rules
from chains.tiers import get_generation_chain
from fastapi.middleware.cors import CORSMiddleware

try:
    from json_repair import repair_json
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def calculate_body_shape(weight: float, height: float, body_fat: float, muscle_mass: float) -> str:
    bmi = weight / (height / 100) ** 2
    if muscle_mass > 40 and body_fat < 15:
        return "athletic"
    elif body_fat > 30 and muscle_mass < 30:
        return "endomorph"
    elif bmi < 18.5:
        return "ectomorph"
    else:
        return "mesomorph"
    

SPORT_CHAINS = {
    "gym": analysis_chain,
    # "swimming": swimming_analysis_chain,  # future
    # "football": football_analysis_chain,  # future
    # "rehab": rehab_analysis_chain,        # future
}

def get_chain_for_sport(sport_types: list[str]):
    primary = sport_types[0] if sport_types else 'gym'
    return SPORT_CHAINS.get(primary.lower(), analysis_chain)

class ClientData(BaseModel):
    age: int
    goal: str
    currentXP: int
    currentTier: str
    measurements: list
    xpLogs: list
    currentExercises: list
    completedChallenges: list
    pastPrograms: list
    height: float
    sportTypes: list[str]
    trainerNotes: str | None = None
    injuryNotes: str | None = None


def _strip_raw(content: str) -> str:
    """Strip think blocks and markdown fences from LLM output."""
    clean = content.strip()
    if "<think>" in clean:
        clean = clean.split("</think>")[-1].strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    return clean.strip()


def _parse_json(raw: str) -> dict:
    """
    Try to parse JSON from raw string.
    Falls back to json-repair if available and parsing fails.
    """
    # First attempt — direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Second attempt — find JSON boundaries
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass

    # Third attempt — json-repair as last resort
    if HAS_JSON_REPAIR:
        try:
            return json.loads(repair_json(raw))
        except Exception:
            pass

    raise ValueError("Could not parse JSON from LLM output")


@app.get("/")
def root():
    return {"message": "GymXP AI Service is running"}


@app.post("/suggest/workout")
async def suggest_workout(client: ClientData):
    from agents.orchestrator import run as orchestrator_run
    try:
        result = await orchestrator_run(client)
        return {"suggestions": result}
    except ValueError as e:
        msg = str(e)
        if "safe workout" in msg:
            return {"error": "Could not generate a safe program for this client's injury. Please review the injury notes and try again."}
        return {"error": msg}
    except Exception as e:
        return {"error": str(e)}