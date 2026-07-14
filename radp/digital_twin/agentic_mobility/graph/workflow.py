"""LangGraph workflow construction."""
from langgraph.graph import END, StateGraph

from radp.digital_twin.agentic_mobility.graph.routing import should_retry
from radp.digital_twin.agentic_mobility.models.state import MobilityGenerationState
from radp.digital_twin.agentic_mobility.nodes import (
    location_resolver,
    parameter_agent,
    query_parser,
    suggestion,
    validation,
)


def create_workflow() -> StateGraph:
    """Create and compile the mobility generation workflow.

    Returns:
        Compiled StateGraph workflow
    """
    workflow = StateGraph(MobilityGenerationState)

    # Add nodes
    workflow.add_node("query_parser", query_parser.node)
    workflow.add_node("location_resolver", location_resolver.node)
    workflow.add_node("parameter_agent", parameter_agent.node)
    workflow.add_node("validation", validation.node)
    workflow.add_node("suggestion", suggestion.node)

    # Define flow
    workflow.set_entry_point("query_parser")

    # Fork: 1.1 → (1.2 | 1.3) parallel
    workflow.add_edge("query_parser", "location_resolver")
    workflow.add_edge("query_parser", "parameter_agent")

    # Merge: (1.2 + 1.3) → 1.4
    workflow.add_edge("location_resolver", "validation")
    workflow.add_edge("parameter_agent", "validation")

    # Conditional: 1.4 → success/retry/max_retries
    workflow.add_conditional_edges(
        "validation", should_retry, {"success": END, "retry": "suggestion", "max_retries": END}
    )

    # Loop: 1.5 → 1.1
    workflow.add_edge("suggestion", "query_parser")

    return workflow.compile()
