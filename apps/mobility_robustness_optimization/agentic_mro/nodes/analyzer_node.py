"""
Analyzer Node for Agentic MRO

Processes raw network data, extracts features, and generates LLM-powered analysis.

Based on architecture document Section 4.2 (Node Definitions)
"""

import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import AgenticMROState
from llm.llm_provider import create_llm_provider
from llm.prompt_templates import ANALYZER_PROMPT_TEMPLATE, format_network_statistics
from utils.feature_extraction import extract_network_features


def analyzer_node(state: AgenticMROState) -> AgenticMROState:
    """
    Analyzer Agent: Process raw CSV and generate network intelligence.

    Steps:
    1. Load CSV data or use raw DataFrame
    2. Extract features using existing MRO preprocessing functions
    3. Create insights DataFrame with aggregated statistics
    4. Format LLM prompt with statistics
    5. Call LLM via llm_provider abstraction
    6. Save markdown context

    Updates state with:
    - insights_dataframe
    - analyzer_markdown

    Args:
        state: Current AgenticMROState

    Returns:
        Updated AgenticMROState with analyzer outputs

    Raises:
        ValueError: If neither input_csv_path nor raw_dataframe is provided
        ValueError: If topology is not provided
    """
    print("\n" + "="*60)
    print("ANALYZER NODE: Starting network analysis...")
    print("="*60)

    # Step 1: Load data
    if state.get("raw_dataframe") is not None:
        df = state["raw_dataframe"]
        print(f"✓ Using provided DataFrame: {len(df)} rows")
    elif state.get("input_csv_path"):
        print(f"✓ Loading CSV: {state['input_csv_path']}")
        df = pd.read_csv(state["input_csv_path"])
        print(f"  Loaded {len(df)} rows")
    else:
        raise ValueError("Either input_csv_path or raw_dataframe must be provided")

    # Step 2: Extract features (topology extracted automatically from CSV)
    print("\n→ Extracting network features...")
    processed_df, insights_df = extract_network_features(df)
    print(f"✓ Feature extraction complete")
    print(f"  Processed DataFrame: {len(processed_df)} rows")
    print(f"  Insights extracted: {len(insights_df.columns)} metrics")

    # Step 3: Format statistics for LLM
    print("\n→ Formatting insights for LLM...")
    network_statistics = format_network_statistics(insights_df)

    # Step 4: Create LLM prompt
    rlf_threshold = state.get("rlf_threshold", -4.0)

    prompt = ANALYZER_PROMPT_TEMPLATE.format(
        network_statistics=network_statistics,
        rlf_threshold=rlf_threshold
    )

    # Step 5: Call LLM
    print("\n→ Calling LLM for network analysis...")
    try:
        llm = create_llm_provider(state["llm_config"])
        analyzer_markdown = llm.generate(prompt)
        print(f"✓ LLM analysis received ({len(analyzer_markdown)} characters)")
    except Exception as e:
        print(f"✗ LLM call failed: {e}")
        # Fallback: use basic analysis
        analyzer_markdown = _generate_fallback_analysis(insights_df)
        print(f"✓ Using fallback analysis")

    # Step 6: Display analysis
    print("\n" + "-"*60)
    print("NETWORK ANALYSIS:")
    print("-"*60)
    print(analyzer_markdown)
    print("-"*60)

    # Step 7: Update state
    state["insights_dataframe"] = insights_df
    state["analyzer_markdown"] = analyzer_markdown
    state["raw_dataframe"] = processed_df  # Store processed data for later use

    print("\n✓ Analyzer Node Complete")
    print("="*60)

    return state


def _generate_fallback_analysis(insights_df: pd.DataFrame) -> str:
    """
    Generate basic fallback analysis if LLM fails.

    Args:
        insights_df: DataFrame with network insights

    Returns:
        Basic analysis string
    """
    if insights_df.empty:
        return "Network analysis unavailable due to missing data."

    insights = insights_df.iloc[0].to_dict()

    # Extract key metrics
    num_ues = insights.get('num_ues', 'N/A')
    num_cells = insights.get('num_cells', 'N/A')
    sinr_mean = insights.get('sinr_mean', None)
    velocity_mean = insights.get('velocity_mean', None)
    handover_risk = insights.get('handover_risk', 'UNKNOWN')

    # Generate basic analysis
    analysis_parts = []

    analysis_parts.append(f"Network has {num_ues} UEs across {num_cells} cells.")

    if sinr_mean is not None:
        if sinr_mean < 0:
            analysis_parts.append(f"Poor signal quality detected (avg SINR: {sinr_mean:.2f} dB).")
        elif sinr_mean < 5:
            analysis_parts.append(f"Moderate signal quality (avg SINR: {sinr_mean:.2f} dB).")
        else:
            analysis_parts.append(f"Good signal quality (avg SINR: {sinr_mean:.2f} dB).")

    if velocity_mean is not None:
        if velocity_mean > 0.01:
            analysis_parts.append(f"High mobility detected (avg velocity: {velocity_mean:.4f}).")
        else:
            analysis_parts.append(f"Low mobility (avg velocity: {velocity_mean:.4f}).")

    analysis_parts.append(f"Handover risk assessment: {handover_risk}.")

    return " ".join(analysis_parts)


if __name__ == "__main__":
    """Test analyzer node independently."""
    from state import create_initial_state

    # Test configuration
    llm_config = {
        "provider": "groq",
        "model": "llama-3.1-70b-versatile",
        "temperature": 0.2,
        "max_tokens": 2000
    }

    # Create test topology
    topology = pd.DataFrame({
        'cell_id': ['cell_0', 'cell_1', 'cell_2'],
        'cell_lat': [33.5, 33.6, 33.7],
        'cell_lon': [102.0, 102.1, 102.2],
        'cell_az_deg': [0, 120, 240],
        'cell_carrier_freq_mhz': [2100, 2100, 2100]
    })

    # Initialize state
    state = create_initial_state(
        llm_config=llm_config,
        input_csv_path="notebooks/simulation_data_initial.csv",
        topology=topology
    )

    # Run analyzer node
    updated_state = analyzer_node(state)

    # Display results
    print("\n\nRESULTS:")
    print(f"Insights shape: {updated_state['insights_dataframe'].shape}")
    print(f"Analysis length: {len(updated_state['analyzer_markdown'])} characters")
