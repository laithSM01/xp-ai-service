
from rag.retriever import classify_injury, fetch_injury_context


async def get_injury_context(injury_text: str, llm) -> str:
    if not injury_text or not injury_text.strip():
        return ""
    categories = await classify_injury(injury_text, llm)
    return fetch_injury_context(categories)
