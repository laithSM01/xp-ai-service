from agents.gym_expert import run as gym_expert_run


async def run(client) -> dict:
    # Future: check client.sportTypes to route to swimming_expert, boxing_expert, etc.
    # Future: check if injuryNotes indicates full rehab needed → rehab_expert
    return await gym_expert_run(client)
