"""LangGraph node for parameter validation."""
from typing import Dict

from radp.digital_twin.agentic_mobility.chains.validation_chain import ValidationChain
from radp.digital_twin.agentic_mobility.models.generation_params import GenParams
from radp.digital_twin.agentic_mobility.models.query_intent import QueryIntent
from radp.digital_twin.agentic_mobility.models.state import MobilityGenerationState


def node(state: MobilityGenerationState) -> Dict:
    """Validation node - validates generated parameters.

    Args:
        state: Current graph state

    Returns:
        Dict with only validation_result key updated
    """
    query_intent_dict = state["query_intent"]
    gen_params_dict = state["gen_params"]

    if not query_intent_dict or not gen_params_dict:
        raise ValueError("query_intent and gen_params must be populated before validation")

    # Convert dicts back to objects
    gen_params = GenParams(**gen_params_dict)
    query_intent = QueryIntent(**query_intent_dict)

    validation_chain = ValidationChain()

    # Validate parameters
    validation_result = validation_chain.validate(gen_params, query_intent)

    # Return only the keys we're updating
    return {"validation_result": validation_result}
