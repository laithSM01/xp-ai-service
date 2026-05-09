# Add a New Sport Expert

## When to use
When adding swimming_expert.py or boxing_expert.py to agents/

## What to tell Claude Code

I need to add a new sport expert called [SPORT]_expert.py to the agents/ folder.

Follow the exact same pattern as agents/gym_expert.py:
- Same pipeline: injury RAG → Chain 1 (analysis) → Chain 2 (generation via get_generation_chain)
- Same input: ClientData object
- Same output: dict from enforce_rules()
- LLM: ChatOllama(model="deepseek-r1:14b")

The only differences from gym_expert.py:
- Sport-specific context in the analysis prompt
- Import from rag/[SPORT]_knowledge.md instead of gym_injuries.md

Also:
1. Create rag/knowledge/[SPORT]_injuries.md — same structure as gym_injuries.md but for [SPORT]-specific injuries
2. Update agents/orchestrator.py to route clients whose sportTypes includes "[SPORT]" to [SPORT]_expert

Do not touch chains/tiers/, chains/workout_suggestion.py, or any existing expert.