from .beginner_chain import beginner_chain
from .novice_chain import novice_chain
from .intermediate_chain import intermediate_chain
from .advanced_chain import advanced_chain
from .elite_chain import elite_chain

_TIER_CHAINS = {
    "beginner": beginner_chain,
    "novice": novice_chain,
    "intermediate": intermediate_chain,
    "advanced": advanced_chain,
    "elite": elite_chain,
}


def get_generation_chain(tier: str):
    return _TIER_CHAINS.get(tier.strip().lower(), intermediate_chain)
