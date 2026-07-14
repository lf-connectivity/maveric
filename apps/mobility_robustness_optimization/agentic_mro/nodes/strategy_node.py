"""
Strategy Node for Agentic MRO

Plans optimization strategy and recommends parameter ranges based on network analysis.

Based on architecture document Section 4.2 (Node Definitions)
"""

import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import AgenticMROState
from llm.llm_provider import create_llm_provider
from llm.prompt_templates import STRATEGY_PROMPT_TEMPLATE, format_network_statistics
from utils.evaluation import get_parameter_ranges


def strategy_node(state: AgenticMROState) -> AgenticMROState:
    """
    Strategy Agent: Plan optimization approach.

    Steps:
    1. Read analyzer markdown + insights DF
    2. Calculate valid parameter ranges from data
    3. Format strategic planning prompt
    4. Call LLM via llm_provider abstraction
    5. Parse JSON response
    6. Extract hyst_range and ttt_range

    Updates state with:
    - strategy_json
    - hyst_range
    - ttt_range

    Args:
        state: Current AgenticMROState with analyzer outputs

    Returns:
        Updated AgenticMROState with strategy outputs

    Raises:
        ValueError: If analyzer_markdown or insights_dataframe is missing
    """
    print("\n" + "="*60)
    print("STRATEGY NODE: Planning optimization strategy...")
    print("="*60)

    # Step 1: Check inputs
    if not state.get("analyzer_markdown"):
        raise ValueError("analyzer_markdown is required from Analyzer Node")

    if state.get("insights_dataframe") is None:
        raise ValueError("insights_dataframe is required from Analyzer Node")

    network_analysis = state["analyzer_markdown"]
    insights_df = state["insights_dataframe"]

    print(f"✓ Received network analysis ({len(network_analysis)} characters)")
    print(f"✓ Received insights ({len(insights_df.columns)} metrics)")

    # Step 2: Calculate parameter ranges
    print("\n→ Calculating parameter ranges from data...")
    processed_df = state.get("raw_dataframe")

    if processed_df is None:
        raise ValueError("Processed DataFrame not available from Analyzer Node")

    hyst_range, ttt_range = get_parameter_ranges(processed_df)
    print(f"✓ Hysteresis range: [{hyst_range[0]:.2f}, {hyst_range[1]:.2f}] dB")
    print(f"✓ TTT range: [{ttt_range[0]}, {ttt_range[1]}] ticks")

    # Step 3: Format network statistics
    network_statistics = format_network_statistics(insights_df)
    rlf_threshold = state.get("rlf_threshold", -4.0)

    # Step 4: Create LLM prompt
    prompt = STRATEGY_PROMPT_TEMPLATE.format(
        network_analysis=network_analysis,
        network_statistics=network_statistics,
        max_hyst=hyst_range[1],
        max_ttt=ttt_range[1],
        rlf_threshold=rlf_threshold
    )

    # Step 5: Call LLM
    print("\n→ Calling LLM for strategy planning...")
    try:
        llm = create_llm_provider(state["llm_config"])
        strategy_json = llm.generate_json(prompt)
        print(f"✓ Strategy JSON received")
    except Exception as e:
        print(f"✗ LLM call failed: {e}")
        # Fallback: use heuristic strategy
        strategy_json = _generate_fallback_strategy(insights_df, hyst_range, ttt_range)
        print(f"✓ Using fallback strategy")

    # Step 6: Extract parameter ranges from strategy
    param_rec = strategy_json.get("parameter_recommendations", {})

    hyst_rec = param_rec.get("hysteresis", {})
    recommended_hyst_range = (
        hyst_rec.get("min", hyst_range[0]),
        hyst_rec.get("max", hyst_range[1])
    )

    ttt_rec = param_rec.get("time_to_trigger", {})
    recommended_ttt_range = (
        int(ttt_rec.get("min", ttt_range[0])),
        int(ttt_rec.get("max", ttt_range[1]))
    )

    # Step 7: Display strategy
    print("\n" + "-"*60)
    print("OPTIMIZATION STRATEGY:")
    print("-"*60)
    print(json.dumps(strategy_json, indent=2))
    print("-"*60)

    print(f"\n→ Recommended Hysteresis Range: [{recommended_hyst_range[0]:.2f}, {recommended_hyst_range[1]:.2f}] dB")
    print(f"→ Recommended TTT Range: [{recommended_ttt_range[0]}, {recommended_ttt_range[1]}] ticks")

    # Step 8: Update state
    state["strategy_json"] = strategy_json
    state["hyst_range"] = recommended_hyst_range
    state["ttt_range"] = recommended_ttt_range

    print("\n✓ Strategy Node Complete")
    print("="*60)

    return state


def _generate_fallback_strategy(insights_df, hyst_range, ttt_range) -> dict:
    """
    Generate heuristic fallback strategy if LLM fails.

    Args:
        insights_df: DataFrame with network insights
        hyst_range: Full hysteresis range
        ttt_range: Full TTT range

    Returns:
        Strategy JSON dictionary
    """
    insights = insights_df.iloc[0].to_dict()

    # Extract key metrics
    sinr_mean = insights.get('sinr_mean', 0)
    velocity_mean = insights.get('velocity_mean', 0)

    # Heuristic strategy
    if sinr_mean < 0:
        # Poor signal → prioritize stability
        hyst_min = max(hyst_range[0], 3.0)
        hyst_max = min(hyst_range[1], 5.0)
        ttt_min = max(ttt_range[0], 6)
        ttt_max = min(ttt_range[1], 8)
        priority = "STABILITY_OVER_RESPONSIVENESS"
        predicted_hyst = 4.0
        predicted_ttt = 7
    elif sinr_mean > 5:
        # Good signal → allow responsiveness
        hyst_min = max(hyst_range[0], 1.0)
        hyst_max = min(hyst_range[1], 3.0)
        ttt_min = max(ttt_range[0], 3)
        ttt_max = min(ttt_range[1], 5)
        priority = "RESPONSIVENESS_OVER_STABILITY"
        predicted_hyst = 2.0
        predicted_ttt = 4
    else:
        # Moderate signal → balanced approach
        hyst_min = max(hyst_range[0], 2.0)
        hyst_max = min(hyst_range[1], 4.0)
        ttt_min = max(ttt_range[0], 4)
        ttt_max = min(ttt_range[1], 7)
        priority = "BALANCED"
        predicted_hyst = 3.0
        predicted_ttt = 5

    return {
        "parameter_recommendations": {
            "hysteresis": {
                "min": hyst_min,
                "max": hyst_max,
                "reasoning": f"Based on SINR mean of {sinr_mean:.2f} dB"
            },
            "time_to_trigger": {
                "min": ttt_min,
                "max": ttt_max,
                "reasoning": f"Based on mobility pattern (velocity: {velocity_mean:.4f})"
            }
        },
        "optimization_strategy": {
            "priority": priority,
            "test_sequence": "SIMULTANEOUS",
            "predicted_optimal": {
                "hysteresis": predicted_hyst,
                "ttt": predicted_ttt
            },
            "reasoning": "Heuristic fallback strategy based on signal quality and mobility"
        }
    }
