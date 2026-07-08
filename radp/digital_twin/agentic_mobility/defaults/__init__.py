"""Scenario presets and parameter defaults for agentic mobility generation."""
from typing import Dict, Optional

from radp.digital_twin.agentic_mobility.defaults.rural import RURAL_PRESET
from radp.digital_twin.agentic_mobility.defaults.suburban import SUBURBAN_PRESET
from radp.digital_twin.agentic_mobility.defaults.urban import URBAN_PRESET

PRESETS: Dict[str, Dict] = {
    "urban": URBAN_PRESET,
    "suburban": SUBURBAN_PRESET,
    "rural": RURAL_PRESET,
}


def get_preset(name: Optional[str]) -> Optional[Dict]:
    """Look up a scenario preset by name (case-insensitive).

    Args:
        name: Scenario name (e.g., "urban", "Suburban", "RURAL"). May be None.

    Returns:
        The preset dict, or None if the name does not match a known preset.
    """
    if not name:
        return None
    return PRESETS.get(name.lower())


__all__ = [
    "PRESETS",
    "URBAN_PRESET",
    "SUBURBAN_PRESET",
    "RURAL_PRESET",
    "get_preset",
]
