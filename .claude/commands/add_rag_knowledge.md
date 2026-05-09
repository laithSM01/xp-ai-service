# Add New Knowledge to RAG

## When to use
When adding new injuries, conditions, or sport-specific knowledge to the RAG knowledge base

## What to tell Claude Code

Add the following injury/condition to rag/knowledge/gym_injuries.md:

Injury name: [NAME]
Aliases: [how trainers might describe it]
Exercises to avoid: [list]
Exercises to prefer: [list]
Technique notes: [what the AI should know]

Follow the exact same markdown structure as existing entries in the file.
Do not change anything else — not retriever.py, not __init__.py, not any chain.