"""LangGraph state definition."""
from typing import Dict, Optional, TypedDict


class MobilityGenerationState(TypedDict):
    """State schema for LangGraph workflow."""

    user_query: str  # Original user input
    current_query: str  # Updated with suggestions (for retries)
    query_intent: Optional[Dict]  # Parsed structured intent
    location_data: Optional[Dict]  # Geographic bounds from resolver
    gen_params: Optional[Dict]  # Generated mobility parameters
    validation_result: Optional[Dict]  # Validation outcome + errors
    retry_count: int  # Current retry attempt (0, 1, or 2)
