"""LangGraph node for location resolution."""
from typing import Dict

from radp.digital_twin.agentic_mobility.models.state import MobilityGenerationState
from radp.digital_twin.agentic_mobility.utils.geocoding import GeocodingService


def node(state: MobilityGenerationState) -> Dict:
    """Location resolver node - converts location string to geographic bounds.

    Args:
        state: Current graph state

    Returns:
        Dict with only location_data key updated
    """
    query_intent = state["query_intent"]

    if not query_intent:
        raise ValueError("query_intent must be populated before location resolution")

    geocoding_service = GeocodingService()

    # Extract location from query_intent dict
    location = query_intent["location"]

    # Geocode location
    location_data = geocoding_service.geocode_location(location)

    if not location_data:
        # Fallback to default bounds if geocoding fails
        print(f"Warning: Geocoding failed for '{location}', using default bounds")
        location_data = {
            "center": (0.0, 0.0),
            "min_lat": -0.05,
            "max_lat": 0.05,
            "min_lon": -0.05,
            "max_lon": 0.05,
            "area_type": "urban",
        }
    else:
        # Convert LocationData to dict
        location_data = location_data.dict()

    # Return only the keys we're updating
    return {"location_data": location_data}
