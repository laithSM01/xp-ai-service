# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Activate virtualenv (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Install dependencies:**
```
pip install -r requirements.txt
```

**Run the API server:**
```
uvicorn main:app --reload
```

**Run all tests:**
```
pytest
```

**Run a single test by ID:**
```
pytest tests/test_workout_chain.py -k "beginner_fat_loss_age30"
```

Tests are slow — each case fires two LLM calls against local Ollama. Run targeted test IDs when iterating.

## Architecture

This is a FastAPI service that generates personalized workout programs using a two-step LLM chain. The single endpoint is `POST /suggest/workout` which accepts a `ClientData` body and returns a weekly schedule JSON.

### Request flow

```
POST /suggest/workout (ClientData)
  → agents/orchestrator.py       # sport routing (currently only "gym")
    → agents/gym_expert.py       # main worker
        1. Calculate body_shape from measurements (BMI + fat/muscle thresholds)
        2. RAG: classify injuryNotes → fetch matching sections from gym_injuries.md
        3. analysis_chain.ainvoke() → JSON summary (fatTrend, muscleTrend, trainingDays, cardioRatio, etc.)
        4. get_generation_chain(tier).ainvoke() → weekly schedule JSON
        5. enforce_rules() → deterministic post-processing (day count fix, dedup exercises)
        6. Return dict
  → {"suggestions": result}
```

### Chain architecture (`chains/`)

There are two sequential chain steps, not one:

**Step 1 — Analysis** (`chains/workout_suggestion.py:analysis_chain`): A focused prompt that analyzes the client's measurement history, XP logs, goals, body shape, injury context, and sport types. Returns a compact JSON struct used as input to step 2. Also exports `enforce_rules()`.

**Step 2 — Generation** (`chains/tiers/`): Five tier-specific chains (beginner → novice → intermediate → advanced → elite), each with different split rules, exercise constraints, and volume prescriptions. All share `OUTPUT_RULES` from `chains/tiers/base.py`. `get_generation_chain(tier)` dispatches to the right chain; falls back to `intermediate_chain` for unknown tiers.

### RAG (`rag/`)

The injury pipeline is two steps:
1. `classify_injury(text, llm)` — uses the LLM to map free-text injury notes onto a fixed `KNOWN_CATEGORIES` list (returns a JSON array).
2. `fetch_injury_context(categories)` — pure string matching against `## Heading` sections in `rag/knowledge/gym_injuries.md`. No vector store — just regex section splitting.

The injury context string is injected into the analysis prompt, where it overrides default exercise selection rules.

### LLM

All chains currently use `ChatOllama(model="deepseek-r1:8b")` (local Ollama). A `ChatAnthropic` import is commented out in `chains/workout_suggestion.py` and `agents/gym_expert.py` — it was pointing to `claude-sonnet-4-20250514`.

### JSON parsing strategy

LLM outputs go through a three-layer fallback in `_parse_json()`:
1. Direct `json.loads()`
2. Boundary extraction (`find("{")` / `rfind("}")`)
3. `json_repair` library (last resort)

`_strip_raw()` must run first to remove DeepSeek-R1's `<think>...</think>` reasoning blocks and any markdown code fences before attempting JSON parsing.

### `enforce_rules()` (post-processing)

Runs deterministically after the generation chain — not an LLM call. It:
- Deduplicates exercises across training days (normalizes names, lowercased, stripped of special chars)
- Clamps or pads training day count to match tier defaults (`TIER_TRAINING_DAYS`)
- Re-numbers all days sequentially

### Tier training day rules

| Tier         | Training days |
|--------------|---------------|
| Beginner     | 3             |
| Novice       | 4             |
| Intermediate | 4             |
| Advanced     | 5             |
| Elite        | 6             |

### Environment

Requires a `.env` file with `GOOGLE_API_KEY` (for future Google Gemini support — not currently wired). Ollama must be running locally with `deepseek-r1:8b` pulled.

### Tests

Tests in `tests/test_workout_chain.py` call `gym_expert_run()` directly (bypass FastAPI). Each test profile is parametrized with `(input_data, min_days, max_days)` and validated by `_assert_valid_program()`, which checks: required JSON keys, rest day structure, training day count bounds, no duplicate exercise names across days, and numeric `reps` values.

`conftest.py` forces `WindowsSelectorEventLoopPolicy` on Windows to work around a Python 3.13 + httpx async loop issue.

### Extension points

- **New sports**: add a chain to `SPORT_CHAINS` in `main.py` and a corresponding agent under `agents/`, then update `orchestrator.py` to route by `client.sportTypes`.
- **New injury categories**: add to `KNOWN_CATEGORIES` in `rag/retriever.py` and add a matching `## Heading` section in `rag/knowledge/gym_injuries.md`.
- **Switching LLM**: replace `ChatOllama` with `ChatAnthropic` (already imported and commented) or any LangChain-compatible LLM in `chains/workout_suggestion.py` and each tier chain file.
