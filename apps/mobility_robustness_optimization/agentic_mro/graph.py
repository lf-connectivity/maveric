"""
LangGraph Construction for Agentic MRO

Builds the multi-agent workflow graph with conditional looping.

Based on architecture document Section 4.4 (Graph Construction)
"""

from langgraph.graph import StateGraph, END
from state import AgenticMROState
from nodes.analyzer_node import analyzer_node
from nodes.strategy_node import strategy_node
from nodes.coordinator_node import coordinator_node
from nodes.finalize_node import finalize_node
from utils.stop_conditions import should_continue_optimization


def create_agentic_mro_graph():
    """
    Build LangGraph for Agentic MRO system.

    Graph Structure:
        START → Analyzer → Strategy → Coordinator ⟲ (loops max 3x) → Finalize → END

    The Coordinator node loops based on stop conditions:
    - Continue if: iteration < max_iterations AND score < target AND not plateaued
    - Finalize if: any stop condition met

    Returns:
        Compiled LangGraph application

    Example:
        >>> graph = create_agentic_mro_graph()
        >>> result = graph.invoke(initial_state)
        >>> print(result["final_output"])
    """
    print("\n" + "="*60)
    print("Building Agentic MRO LangGraph...")
    print("="*60)

    # Initialize graph with state schema
    workflow = StateGraph(AgenticMROState)

    # Add nodes
    print("→ Adding nodes:")
    workflow.add_node("analyzer", analyzer_node)
    print("  ✓ Analyzer")

    workflow.add_node("strategy", strategy_node)
    print("  ✓ Strategy")

    workflow.add_node("coordinator", coordinator_node)
    print("  ✓ Coordinator")

    workflow.add_node("finalize", finalize_node)
    print("  ✓ Finalize")

    # Define sequential edges
    print("\n→ Adding edges:")
    workflow.set_entry_point("analyzer")
    print("  ✓ START → Analyzer")

    workflow.add_edge("analyzer", "strategy")
    print("  ✓ Analyzer → Strategy")

    workflow.add_edge("strategy", "coordinator")
    print("  ✓ Strategy → Coordinator")

    # Conditional edge for coordinator loop
    workflow.add_conditional_edges(
        "coordinator",
        should_continue_optimization,
        {
            "continue": "coordinator",  # Loop back
            "finalize": "finalize"      # Exit loop
        }
    )
    print("  ✓ Coordinator → [continue: Coordinator | finalize: Finalize]")

    # Final edge to END
    workflow.add_edge("finalize", END)
    print("  ✓ Finalize → END")

    # Compile graph
    print("\n→ Compiling graph...")
    app = workflow.compile()
    print("✓ Graph compiled successfully")

    print("="*60 + "\n")

    return app


if __name__ == "__main__":
    """Test graph construction."""
    graph = create_agentic_mro_graph()
    print("Graph created successfully!")
    print(f"Graph type: {type(graph)}")
