"""
Stop Conditions for Agentic MRO Optimization Loop

Implements plateau detection and continuation logic for the Coordinator Agent.

Based on architecture document Section 7 (Stop Conditions Summary)
"""

from typing import Literal
import sys
import os

# Add parent directory to path for state imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import AgenticMROState


def should_continue_optimization(state: AgenticMROState) -> Literal["continue", "finalize"]:
    """
    Determine if Coordinator should continue optimization or stop.

    Stop conditions (evaluated in order):
    1. Max iterations reached (hard limit)
    2. Target score achieved (goal met)
    3. Plateau detected (< 1% improvement for 2 iterations)

    Args:
        state: Current AgenticMROState

    Returns:
        "continue" if optimization should proceed
        "finalize" if optimization should stop

    Example:
        >>> state["iteration_count"] = 3
        >>> state["max_iterations"] = 3
        >>> result = should_continue_optimization(state)
        >>> assert result == "finalize"
    """
    # Check 1: Max iterations (hard limit)
    if state["iteration_count"] >= state["max_iterations"]:
        print(f"[STOP] Maximum iterations reached: {state['iteration_count']}/{state['max_iterations']}")
        return "finalize"

    # Check 2: Target score achieved
    if state["best_score"] >= state["target_score"]:
        print(f"[STOP] Target score achieved: {state['best_score']:.4f} >= {state['target_score']}")
        return "finalize"

    # Check 3: Plateau detection
    if state.get("plateau_detected", False):
        print(f"[STOP] Plateau detected - improvement < 1% for 2 consecutive iterations")
        return "finalize"

    # Continue optimization
    print(f"[CONTINUE] Iteration {state['iteration_count']}/{state['max_iterations']}, "
          f"Best score: {state['best_score']:.4f}, Target: {state['target_score']}")
    return "continue"


def detect_plateau(state: AgenticMROState) -> bool:
    """
    Detect if optimization has plateaued.

    Plateau condition:
    - Last 2 iterations show < 1% improvement in score

    Args:
        state: Current AgenticMROState with tested_parameters history

    Returns:
        True if plateau detected, False otherwise

    Example:
        >>> state["tested_parameters"] = [
        ...     {"score": 0.70},
        ...     {"score": 0.705},  # 0.71% improvement
        ... ]
        >>> assert detect_plateau(state) == True
    """
    tested_params = state.get("tested_parameters", [])

    # Need at least 2 iterations to detect plateau
    if len(tested_params) < 2:
        return False

    # Get last 2 scores
    recent_scores = [p["score"] for p in tested_params[-2:]]

    # Calculate improvement
    prev_score = recent_scores[0]
    curr_score = recent_scores[1]

    # Avoid division by zero
    if prev_score == 0:
        return False

    # Calculate percentage improvement
    improvement = (curr_score - prev_score) / abs(prev_score)

    # Plateau if improvement < 1%
    is_plateau = abs(improvement) < 0.01

    if is_plateau:
        print(f"[PLATEAU] Improvement: {improvement * 100:.2f}% (< 1% threshold)")
        print(f"  Previous score: {prev_score:.4f}, Current score: {curr_score:.4f}")

    return is_plateau


def update_stop_conditions(state: AgenticMROState) -> AgenticMROState:
    """
    Update stop condition flags in state.

    This should be called after each Coordinator iteration to:
    1. Detect plateau
    2. Update plateau_detected flag

    Args:
        state: Current AgenticMROState

    Returns:
        Updated state with stop condition flags
    """
    # Detect plateau
    plateau = detect_plateau(state)
    state["plateau_detected"] = plateau

    return state


def get_stop_reason(state: AgenticMROState) -> str:
    """
    Get human-readable reason for stopping optimization.

    Args:
        state: Current AgenticMROState

    Returns:
        String describing why optimization stopped

    Example:
        >>> state["iteration_count"] = 3
        >>> state["max_iterations"] = 3
        >>> reason = get_stop_reason(state)
        >>> print(reason)
        "Maximum iterations reached (3/3)"
    """
    # Check stop conditions in priority order
    if state["iteration_count"] >= state["max_iterations"]:
        return f"Maximum iterations reached ({state['iteration_count']}/{state['max_iterations']})"

    if state["best_score"] >= state["target_score"]:
        return f"Target score achieved ({state['best_score']:.4f} >= {state['target_score']})"

    if state.get("plateau_detected", False):
        return "Plateau detected (improvement < 1% for 2 consecutive iterations)"

    # Still running
    return "Optimization in progress"


def print_stop_summary(state: AgenticMROState) -> None:
    """
    Print a summary of why optimization stopped.

    Args:
        state: Final AgenticMROState after optimization
    """
    print("\n" + "=" * 60)
    print("OPTIMIZATION STOPPED")
    print("=" * 60)
    print(f"Reason: {get_stop_reason(state)}")
    print(f"Total Iterations: {state['iteration_count']}")
    print(f"Best Score: {state['best_score']:.4f}")
    print(f"Best Parameters: hyst={state.get('best_hyst')}, ttt={state.get('best_ttt')}")
    print("=" * 60 + "\n")
