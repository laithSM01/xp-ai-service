# Switch the LLM

## When to use
When switching from Ollama to Groq, Together.ai, or Claude API

## What to tell Claude Code

Switch the LLM across the entire xp-ai-service from [CURRENT] to [NEW].

Files that define the LLM and must be updated:
- chains/workout_suggestion.py
- agents/gym_expert.py
- rag/retriever.py (classify_injury uses the LLM)
- Any file in chains/tiers/ that defines its own LLM instance

For Groq: use ChatGroq from langchain_groq, model="llama-3.3-70b-versatile"
For Claude: use ChatAnthropic from langchain_anthropic, model="claude-sonnet-4-20250514"
For Ollama: use ChatOllama from langchain_ollama, model="[MODEL_NAME]"

Load API keys from .env — never hardcode them.
Do not change any prompt, any chain logic, or any test profile.