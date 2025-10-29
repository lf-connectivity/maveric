"""
Coordinator Node for Agentic MRO

Executes intelligent parameter search loop with LLM guidance.

Based on architecture document Section 4.2 (Node Definitions)
"""

import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import AgenticMROState
from llm.llm_provider import create_llm_provider
from llm.prompt_templates import (
    COORDINATOR_PROMPT_TEMPLATE,
    format_tested_parameters,
    format_strategy_summary
)
from utils.evaluation import evaluate_parameters, validate_parameters, format_evaluation_result
from utils.stop_conditions import update_stop_conditions


def coordinator_node(state: AgenticMROState) -> AgenticMROState:
    """
    Coordinator Agent: Execute intelligent parameter search.

    Steps:
    1. Format prompt with context + history
    2. Call LLM for next parameter suggestion
    3. Parse hyst, ttt values
    4. Evaluate score using MRO metric function
    5. Update tested_parameters buffer
    6. Track best score
    7. Check stop conditions
    8. Increment iteration_count

    Updates state with:
    - iteration_count
    - tested_parameters
    - best_score, best_hyst, best_ttt
    - plateau_detected

    Args:
        state: Current AgenticMROState with strategy outputs

    Returns:
        Updated AgenticMROState after one iteration

    Raises:
        ValueError: If required inputs are missing
    """
    print("\n" + "="*60)
    print(f"COORDINATOR NODE: Iteration {state['iteration_count'] + 1}/{state['max_iterations']}")
    print("="*60)

    # Check inputs
    if not state.get("analyzer_markdown"):
        raise ValueError("analyzer_markdown required")
    if state.get("hyst_range") is None or state.get("ttt_range") is None:
        raise ValueError("hyst_range and ttt_range required from Strategy Node")
    if state.get("raw_dataframe") is None:
        raise ValueError("Processed DataFrame required")

    # Get context
    network_analysis = state["analyzer_markdown"]
    hyst_range = state["hyst_range"]
    ttt_range = state["ttt_range"]
    strategy_json = state.get("strategy_json", {})
    tested_params = state.get("tested_parameters", [])
    processed_df = state["raw_dataframe"]
    rlf_threshold = state.get("rlf_threshold", -4.0)

    # Step 1: Format prompt
    prompt = COORDINATOR_PROMPT_TEMPLATE.format(
        network_analysis=network_analysis,
        hyst_min=hyst_range[0],
        hyst_max=hyst_range[1],
        ttt_min=ttt_range[0],
        ttt_max=ttt_range[1],
        strategy_summary=format_strategy_summary(strategy_json),
        tested_parameters_history=format_tested_parameters(tested_params),
        iteration=state["iteration_count"] + 1,
        max_iterations=state["max_iterations"],
        best_score=state["best_score"],
        target_score=state["target_score"]
    )

    # Step 2: Call LLM for suggestion
    print("\n→ Calling LLM for next parameter suggestion...")
    try:
        llm = create_llm_provider(state["llm_config"])
        suggestion_json = llm.generate_json(prompt)
        print(f"✓ Suggestion received")

        # Parse parameters
        suggested_hyst = float(suggestion_json.get("suggested_hyst", hyst_range[0]))
        suggested_ttt = int(suggestion_json.get("suggested_ttt", ttt_range[0]))
        reasoning = suggestion_json.get("reasoning", "No reasoning provided")

    except Exception as e:
        print(f"✗ LLM call failed: {e}")
        # Fallback: use predicted optimal or mid-range
        suggested_hyst, suggested_ttt, reasoning = _generate_fallback_suggestion(
            state, strategy_json, hyst_range, ttt_range
        )
        print(f"✓ Using fallback suggestion")

    # Step 3: Validate parameters
    is_valid, error_msg = validate_parameters(suggested_hyst, suggested_ttt, hyst_range, ttt_range)

    if not is_valid:
        print(f"⚠ Invalid parameters: {error_msg}")
        # Clamp to valid ranges
        suggested_hyst = max(hyst_range[0], min(hyst_range[1], suggested_hyst))
        suggested_ttt = max(ttt_range[0], min(ttt_range[1], int(suggested_ttt)))
        print(f"→ Clamped to: hyst={suggested_hyst:.4f}, ttt={suggested_ttt}")

    print(f"\n→ Testing parameters:")
    print(f"  Hysteresis: {suggested_hyst:.4f} dB")
    print(f"  TTT: {suggested_ttt} ticks")
    print(f"  Reasoning: {reasoning}")

    # Step 4: Evaluate parameters
    print(f"\n→ Evaluating MRO metric...")
    try:
        attached_df, score = evaluate_parameters(
            df=processed_df,
            hyst=suggested_hyst,
            ttt=suggested_ttt,
            rlf_threshold=rlf_threshold
        )
        print(f"✓ Score: {score:.4f}")
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        print(f"✗ Error type: {type(e).__name__}")
        import traceback
        print(f"✗ Traceback:\n{traceback.format_exc()}")
        score = 0.0
        attached_df = processed_df

    # Step 5: Format result
    result = format_evaluation_result(suggested_hyst, suggested_ttt, score, attached_df)
    result["reasoning"] = reasoning

    print(f"  Handovers: {result.get('num_handovers', 'N/A')}")
    print(f"  RLFs: {result.get('num_rlfs', 'N/A')}")

    # Step 6: Update tested parameters
    tested_params.append(result)
    state["tested_parameters"] = tested_params

    # Step 7: Update best score
    if score > state["best_score"]:
        state["best_score"] = score
        state["best_hyst"] = suggested_hyst
        state["best_ttt"] = suggested_ttt
        print(f"\n🎉 NEW BEST SCORE: {score:.4f}")

    # Step 8: Increment iteration
    state["iteration_count"] += 1

    # Step 9: Update stop conditions
    state = update_stop_conditions(state)

    print(f"\n✓ Coordinator Iteration Complete")
    if state.get('best_hyst') is not None:
        print(f"  Best so far: {state['best_score']:.4f} (hyst={state.get('best_hyst'):.4f}, ttt={state.get('best_ttt')})")
    else:
        # Handle -inf case
        if state['best_score'] == float('-inf'):
            print(f"  Best so far: No valid parameters found yet")
        else:
            print(f"  Best so far: {state['best_score']:.4f} (No valid parameters found yet)")
    print("="*60)

    return state


def _generate_fallback_suggestion(state, strategy_json, hyst_range, ttt_range):
    """
    Generate fallback suggestion if LLM fails.

    Args:
        state: Current state
        strategy_json: Strategy recommendations
        hyst_range: Valid hysteresis range
        ttt_range: Valid TTT range

    Returns:
        Tuple of (hyst, ttt, reasoning)
    """
    # Try to use predicted optimal from strategy
    predicted = strategy_json.get("optimization_strategy", {}).get("predicted_optimal", {})

    if predicted and "hysteresis" in predicted and "ttt" in predicted:
        hyst = predicted["hysteresis"]
        ttt = int(predicted["ttt"])
        reasoning = "Using predicted optimal from strategy (LLM fallback)"
    else:
        # Use mid-range
        hyst = (hyst_range[0] + hyst_range[1]) / 2
        ttt = int((ttt_range[0] + ttt_range[1]) / 2)
        reasoning = "Using mid-range values (LLM fallback)"

    return hyst, ttt, reasoning
