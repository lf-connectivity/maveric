"""LangGraph node for query parsing."""
from typing import Dict

from radp.digital_twin.agentic_mobility.chains.parser_chain import ParserChain
from radp.digital_twin.agentic_mobility.models.state import MobilityGenerationState


def node(state: MobilityGenerationState) -> Dict:
    """Query parser node - parses natural language query into QueryIntent.

    Args:
        state: Current graph state

    Returns:
        Dict with only query_intent key updated
    """
    parser = ParserChain()

    # Use current_query (which may be augmented from suggestions)
    query_to_parse = state["current_query"]

    # Parse query
    query_intent = parser.parse(query_to_parse)

    # Return only the keys we're updating
    return {"query_intent": query_intent.dict()}
