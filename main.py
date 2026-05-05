import json
from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
from chains.workout_suggestion import analysis_chain, generation_chain, enforce_rules
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
    tier = client.currentTier.capitalize()

    print(client.model_dump())

    # Calculate body shape from latest measurement
    latest = client.measurements[0] if client.measurements else None
    body_shape = calculate_body_shape(
        latest["weight"], client.height,
        latest["bodyFat"], latest["muscleMass"]
    ) if latest else "unknown"

    # Route to correct chain based on sport
    chain = get_chain_for_sport(client.sportTypes)

    # ── Chain 1: Analyze client data ──
    analysis_result = await chain.ainvoke({
        "age": client.age,
        "goal": client.goal,
        "currentXP": client.currentXP,
        "currentTier": tier,
        "measurements": client.measurements,
        "xpLogs": client.xpLogs,
        "currentExercises": client.currentExercises,
        "completedChallenges": client.completedChallenges,
        "pastPrograms": client.pastPrograms,
        "bodyShape": body_shape,
        "height": client.height,
        "sportTypes": ", ".join(client.sportTypes),
        "trainerNotes": client.trainerNotes or "",
    })

    try:
        analysis_raw = _strip_raw(analysis_result.content)
        analysis = _parse_json(analysis_raw)
    except (ValueError, Exception) as e:
        return {"error": "Analysis chain returned invalid JSON", "raw": analysis_result.content}

    # ── Chain 2: Generate workout from analysis ──
    generation_result = await generation_chain.ainvoke({
        "age": client.age,
        "goal": client.goal,
        "currentTier": tier,
        "fatTrend": analysis.get("fatTrend", "stable"),
        "muscleTrend": analysis.get("muscleTrend", "stable"),
        "trainingDays": analysis.get("trainingDays", 3),
        "cardioRatio": analysis.get("cardioRatio", 50),
        "strengthRatio": analysis.get("strengthRatio", 50),
        "focus": analysis.get("focus", "balanced"),
        "notes": analysis.get("notes", ""),
        "currentExercisesToAvoid": analysis.get("currentExercisesToAvoid", []),
        "bodyShape": body_shape,
        "sportTypes": ", ".join(client.sportTypes),
        "trainerNotes": client.trainerNotes or "",
        "height": client.height,
    })

    try:
        generation_raw = _strip_raw(generation_result.content)
        parsed = _parse_json(generation_raw)
        parsed = enforce_rules(parsed, tier)
        return {"suggestions": parsed}
    except (ValueError, Exception) as e:
        return {"error": "Generation chain returned invalid JSON", "raw": generation_result.content}