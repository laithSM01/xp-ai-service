import asyncio
import pytest

@pytest.fixture(scope="session")
def event_loop():
    """Force a single event loop for the entire test session.
    Fixes RuntimeError: Event loop is closed on Windows + Python 3.13
    with httpx/anyio async clients."""
    if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()