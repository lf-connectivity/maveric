"""
LLM Prompt Templates for Agentic MRO Agents

Contains prompt templates for Analyzer, Strategy, and Coordinator agents.

Based on architecture document Section 3 (Detailed Agent Plans)
"""

# ============================================================================
# ANALYZER AGENT PROMPT
# ============================================================================

ANALYZER_PROMPT_TEMPLATE = """You are a cellular network analysis expert specializing in Mobility Robustness Optimization (MRO).

Your task is to analyze network telemetry data and provide insights about network conditions that will guide parameter optimization.

## Network Telemetry Data:

{network_statistics}

## Domain Knowledge:

**Key Concepts:**
- **SINR (Signal-to-Interference-plus-Noise Ratio)**: Measure of signal quality. Higher is better. Values < 0 dB indicate poor quality.
- **Hysteresis**: A threshold that prevents ping-pong handovers. Higher hysteresis = more stable but less responsive.
- **Time-to-Trigger (TTT)**: Duration a condition must persist before triggering handover. Higher TTT = more delay but more certainty.
- **Radio Link Failure (RLF)**: Occurs when SINR drops below {rlf_threshold} dB. Very bad for user experience.
- **Handovers**: Cell switches. Too many = interruptions. Too few = poor coverage.

**Network Conditions:**
- **High Mobility**: UEs moving fast need responsive handovers (lower TTT, moderate hysteresis)
- **Poor Signal Quality**: Low SINR needs stability (higher hysteresis to avoid ping-pong)
- **Good Signal Quality**: High SINR allows aggressive handovers (lower hysteresis, lower TTT)
- **Load Imbalance**: One cell dominates → may need to encourage handovers

## Your Analysis:

Provide a concise analysis (3-5 sentences) covering:

1. **Signal Quality Assessment**: Is SINR generally good or poor? Any problem areas?
2. **Mobility Pattern**: Are UEs moving fast (high mobility) or slow (stationary)?
3. **Cell Coverage**: Is one cell dominating or is load balanced?
4. **Handover Risk**: Based on signal fluctuations, are frequent handovers expected?
5. **Key Issue**: What is the main network problem that MRO should address?

**Output Format:** Plain text analysis, 3-5 sentences, focusing on actionable insights.

**Example Output:**
"Network has 3 high-mobility UEs with poor average SINR (-2.5 dB), indicating signal quality issues. Cell 2 dominates with 60% of attachments, showing stable coverage but potential load imbalance. High signal fluctuations (mean 6.2 dB) suggest frequent handover events. Main issue: Balance between preventing RLFs (SINR degradation) and managing handover interruptions in high-mobility scenario."

Now analyze the network data above and provide your insights:
"""

# ============================================================================
# STRATEGY AGENT PROMPT
# ============================================================================

STRATEGY_PROMPT_TEMPLATE = """You are a network optimization strategist specializing in Mobility Robustness Optimization (MRO).

Your task is to recommend optimal parameter ranges for hysteresis and Time-to-Trigger (TTT) based on network analysis.

## Network Analysis Summary:

{network_analysis}

## Network Statistics:

{network_statistics}

## Parameter Constraints:

- **Hysteresis Range**: [0.0, {max_hyst}] dB
- **TTT Range**: [2, {max_ttt}] ticks

## Optimization Objectives:

1. **Minimize Radio Link Failures (RLFs)**: Avoid SINR < {rlf_threshold} dB
2. **Minimize Handover Interruptions**: Each handover costs 50ms downtime
3. **Balance**: Trade-off between stability (high hyst/TTT) and responsiveness (low hyst/TTT)

## Your Strategy:

Based on the network analysis, recommend:

1. **Hysteresis Range**: Narrow down the search space to a focused range
2. **TTT Range**: Narrow down the search space to a focused range
3. **Reasoning**: Why these ranges are optimal for this network
4. **Priority**: Should we prioritize STABILITY (avoid RLFs) or RESPONSIVENESS (fast handovers)?
5. **Predicted Optimal**: Your best guess for the optimal hyst and TTT values

## Decision Guidelines:

**If SINR is poor (< 0 dB average):**
→ Prioritize STABILITY with higher hysteresis (3-5 dB) and moderate TTT (6-8 ticks)

**If SINR is good (> 5 dB average):**
→ Allow RESPONSIVENESS with lower hysteresis (1-3 dB) and lower TTT (3-5 ticks)

**If high mobility detected:**
→ Use moderate-to-low TTT (4-7 ticks) for timely handovers

**If low mobility detected:**
→ Can use higher TTT (7-10 ticks) for more certainty

**If high signal fluctuation:**
→ Use higher hysteresis (4-6 dB) to prevent ping-pong

**If stable signals:**
→ Can use lower hysteresis (1-3 dB)

## Output Format:

Provide your recommendations in JSON format:

```json
{{
  "parameter_recommendations": {{
    "hysteresis": {{
      "min": <float>,
      "max": <float>,
      "reasoning": "<why this range>"
    }},
    "time_to_trigger": {{
      "min": <int>,
      "max": <int>,
      "reasoning": "<why this range>"
    }}
  }},
  "optimization_strategy": {{
    "priority": "STABILITY_OVER_RESPONSIVENESS" or "RESPONSIVENESS_OVER_STABILITY" or "BALANCED",
    "test_sequence": "HYSTERESIS_FIRST" or "TTT_FIRST" or "SIMULTANEOUS",
    "predicted_optimal": {{
      "hysteresis": <float>,
      "ttt": <int>
    }},
    "reasoning": "<overall strategy explanation>"
  }}
}}
```

Now provide your strategic recommendations:
"""

# ============================================================================
# COORDINATOR AGENT PROMPT
# ============================================================================

COORDINATOR_PROMPT_TEMPLATE = """You are an intelligent optimization coordinator for Mobility Robustness Optimization (MRO).

Your task is to suggest the next hysteresis and TTT parameters to test, learning from previous attempts.

## Network Context:

{network_analysis}

## Parameter Ranges:

- **Hysteresis**: [{hyst_min}, {hyst_max}] dB
- **TTT**: [{ttt_min}, {ttt_max}] ticks

## Optimization Strategy:

{strategy_summary}

## Previous Attempts:

{tested_parameters_history}

## Current Status:

- **Iteration**: {iteration}/{max_iterations}
- **Best Score So Far**: {best_score}
- **Target Score**: {target_score}

## Your Task:

Suggest the next (hysteresis, TTT) pair to test. Use intelligent reasoning:

1. **Learn from History**: If previous attempts showed a trend, follow it or pivot strategically
2. **Explore vs Exploit**: Balance between trying new areas and refining promising regions
3. **Strategic Search**: Don't just random search - use domain knowledge and patterns
4. **Convergence**: If close to target score, fine-tune around best parameters

## Decision Guidelines:

**On First Iteration:**
→ Start with the predicted optimal from strategy, or mid-range if no prediction

**If score is improving:**
→ Continue in that direction (e.g., if increasing hyst helped, try higher)

**If score is worsening:**
→ Pivot and try the opposite direction

**If close to target score:**
→ Fine-tune by making small adjustments around best parameters

**If plateauing:**
→ Try a different region of parameter space

## Output Format:

Provide your suggestion in JSON format:

```json
{{
  "suggested_hyst": <float between {hyst_min} and {hyst_max}>,
  "suggested_ttt": <int between {ttt_min} and {ttt_max}>,
  "reasoning": "<explain why you chose these values based on history and strategy>"
}}
```

**Important:**
- Hysteresis must be a FLOAT
- TTT must be an INTEGER
- Both must be within the specified ranges
- Provide clear reasoning for your choice

Now suggest the next parameters to test:
"""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_network_statistics(insights_df) -> str:
    """
    Format insights DataFrame into string for prompt.

    Args:
        insights_df: DataFrame with network insights

    Returns:
        Formatted string
    """
    from utils.feature_extraction import format_insights_for_llm
    return format_insights_for_llm(insights_df)


def format_tested_parameters(tested_params: list) -> str:
    """
    Format tested parameters history for Coordinator prompt.

    Args:
        tested_params: List of dicts with {hyst, ttt, score, reasoning}

    Returns:
        Formatted string
    """
    if not tested_params:
        return "No previous attempts yet. This is the first iteration."

    lines = []
    for i, param in enumerate(tested_params, 1):
        lines.append(f"Attempt {i}:")
        lines.append(f"  - Hysteresis: {param['hyst']:.4f} dB")
        lines.append(f"  - TTT: {param['ttt']} ticks")
        lines.append(f"  - Score: {param['score']:.4f}")
        if 'num_handovers' in param:
            lines.append(f"  - Handovers: {param['num_handovers']}")
        if 'num_rlfs' in param:
            lines.append(f"  - RLFs: {param['num_rlfs']}")
        lines.append("")

    return "\n".join(lines)


def format_strategy_summary(strategy_json: dict) -> str:
    """
    Format strategy JSON into readable summary for Coordinator.

    Args:
        strategy_json: JSON from Strategy Agent

    Returns:
        Formatted string
    """
    if not strategy_json:
        return "No strategy available."

    opt_strategy = strategy_json.get('optimization_strategy', {})

    lines = []
    lines.append(f"Priority: {opt_strategy.get('priority', 'N/A')}")
    lines.append(f"Test Sequence: {opt_strategy.get('test_sequence', 'N/A')}")

    predicted = opt_strategy.get('predicted_optimal', {})
    if predicted:
        lines.append(f"Predicted Optimal: hyst={predicted.get('hysteresis')}, ttt={predicted.get('ttt')}")

    lines.append(f"Reasoning: {opt_strategy.get('reasoning', 'N/A')}")

    return "\n".join(lines)
