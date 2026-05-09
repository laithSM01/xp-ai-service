# Commands

Reusable Claude Code prompts. Each file is a ready-to-paste prompt.

## How to use

1. Open the relevant command file
2. Replace all [PLACEHOLDERS] with your actual values
3. Add any extra context specific to your task
4. Paste into Claude Code

## When to edit a command first

Edit the command before running if your task has something
extra that the template doesn't cover. Otherwise just fill
in the placeholders and run as-is.

## Adding a new sport (full workflow)

When adding swimming or boxing:

1. Run add_sport_expert.md     → creates [sport]_expert.py + rag/[sport]_knowledge.md
2. Run add_sport_evaluator.md  → creates evaluator/[sport]_checks.py + updates evaluator/__init__.py
3. Update orchestrator.py      → routes [sport] clients to new expert

## Adding a new injury

Run add_rag_knowledge.md → adds entry to rag/knowledge/gym_injuries.md

## Switching the LLM

Run switch_llm.md → updates all files that define the LLM in one pass