"""
Finalize Node for Agentic MRO

Packages best parameters as final output.

Based on architecture document Section 4.2 (Node Definitions)
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import AgenticMROState
from utils.stop_conditions import print_stop_summary


def finalize_node(state: AgenticMROState) -> AgenticMROState:
    """
    Finalize: Package best parameters as output JSON.

    Updates state with:
    - final_output: {best_hysteresis, best_ttt, best_score}

    Args:
        state: Current AgenticMROState after optimization

    Returns:
        Updated AgenticMROState with final output
    """
    print("\n" + "="*60)
    print("FINALIZE NODE: Packaging results...")
    print("="*60)

    # Print stop summary
    print_stop_summary(state)

    # Package final output
    final_output = {
        "best_hysteresis": state.get("best_hyst"),
        "best_ttt": state.get("best_ttt"),
        "best_score": state.get("best_score", 0.0),
        "total_iterations": state.get("iteration_count", 0),
        "target_score": state.get("target_score", 0.80),
        "tested_parameters": state.get("tested_parameters", [])
    }

    state["final_output"] = final_output

    # Display final output
    print("\n" + "-"*60)
    print("FINAL OUTPUT:")
    print("-"*60)

    if final_output['best_hysteresis'] is not None and final_output['best_ttt'] is not None:
        print(f"Best Hysteresis: {final_output['best_hysteresis']:.4f} dB")
        print(f"Best TTT: {final_output['best_ttt']} ticks")
        print(f"Best Score: {final_output['best_score']:.4f}")
        print(f"Total Iterations: {final_output['total_iterations']}")
    else:
        print("WARNING: No optimal parameters found!")
        print(f"Best Hysteresis: {final_output['best_hysteresis']}")
        print(f"Best TTT: {final_output['best_ttt']}")
        print(f"Best Score: {final_output['best_score']}")
        print(f"Total Iterations: {final_output['total_iterations']}")
        print("\nPossible reasons:")
        print("  - Coordinator node encountered errors")
        print("  - LLM failed to provide valid parameters")
        print("  - Evaluation function failed")

    print("-"*60)

    print("\n✓ Finalize Node Complete")
    print("="*60)

    return state
