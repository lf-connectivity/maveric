"""
State Schema for Agentic MRO System

Defines the state structure used by LangGraph to manage the multi-agent
MRO optimization pipeline.

Based on architecture document Section 4.1
"""

from typing import TypedDict, List, Dict, Optional
import pandas as pd


class AgenticMROState(TypedDict):
    """
    State schema for the Agentic MRO LangGraph workflow.

    This state is passed between nodes and tracks the entire optimization process.
    """

    # ============ Configuration ============
    llm_config: Dict
    """LLM provider configuration (provider, model, api_key, temperature, etc.)"""

    # ============ Input Data ============
    input_csv_path: Optional[str]
    """Path to input CSV file with preprocessed UE data"""

    raw_dataframe: Optional[pd.DataFrame]
    """Preprocessed input DataFrame (alternative to CSV path)"""

    # ============ Analyzer Outputs ============
    insights_dataframe: Optional[pd.DataFrame]
    """Aggregated insights from feature extraction (statistics, metrics)"""

    analyzer_markdown: Optional[str]
    """LLM-generated network analysis in markdown format"""

    # ============ Strategy Outputs ============
    strategy_json: Optional[Dict]
    """Complete strategy JSON response from Strategy Agent"""

    hyst_range: Optional[tuple]
    """Hysteresis parameter range (min, max)"""

    ttt_range: Optional[tuple]
    """Time-to-Trigger parameter range (min, max)"""

    # ============ Coordinator State ============
    iteration_count: int
    """Current iteration number in optimization loop"""

    tested_parameters: List[Dict]
    """History of tested parameters: [{hyst, ttt, score, reasoning}, ...]"""

    best_score: float
    """Best MRO metric score found so far"""

    best_hyst: Optional[float]
    """Best hysteresis value found so far"""

    best_ttt: Optional[int]
    """Best time-to-trigger value found so far"""

    # ============ Stop Conditions ============
    target_score: float
    """Target MRO metric score for early stopping (default: 0.80)"""

    max_iterations: int
    """Maximum number of optimization iterations (default: 3)"""

    plateau_detected: bool
    """Flag indicating if optimization has plateaued"""

    rlf_threshold: float
    """Radio Link Failure threshold in dB (default: -4)"""

    # ============ Final Output ============
    final_output: Optional[Dict]
    """Final output JSON: {best_hysteresis, best_ttt, best_score}"""


def create_initial_state(
    llm_config: Dict,
    input_csv_path: Optional[str] = None,
    raw_dataframe: Optional[pd.DataFrame] = None,
    target_score: float = 0.80,
    max_iterations: int = 3,
    rlf_threshold: float = -4.0
) -> AgenticMROState:
    """
    Create initial state for Agentic MRO pipeline.

    Args:
        llm_config: LLM provider configuration dictionary
        input_csv_path: Path to preprocessed CSV file (optional if raw_dataframe provided)
        raw_dataframe: Preprocessed DataFrame (optional if input_csv_path provided)
        target_score: Target MRO metric for early stopping (default: 0.80)
        max_iterations: Maximum optimization iterations (default: 3)
        rlf_threshold: RLF threshold in dB (default: -4.0)

    Returns:
        Initialized AgenticMROState

    Example:
        >>> llm_config = {
        ...     "provider": "groq",
        ...     "model": "llama-3.1-70b-versatile",
        ...     "temperature": 0.2,
        ...     "max_tokens": 2000
        ... }
        >>> state = create_initial_state(
        ...     llm_config=llm_config,
        ...     input_csv_path="simulation_data_initial.csv"
        ... )
    """
    return AgenticMROState(
        # Configuration
        llm_config=llm_config,

        # Input
        input_csv_path=input_csv_path,
        raw_dataframe=raw_dataframe,

        # Analyzer outputs (initialized as None)
        insights_dataframe=None,
        analyzer_markdown=None,

        # Strategy outputs (initialized as None)
        strategy_json=None,
        hyst_range=None,
        ttt_range=None,

        # Coordinator state (initialized)
        iteration_count=0,
        tested_parameters=[],
        best_score=0.0,
        best_hyst=None,
        best_ttt=None,

        # Stop conditions
        target_score=target_score,
        max_iterations=max_iterations,
        plateau_detected=False,
        rlf_threshold=rlf_threshold,

        # Final output
        final_output=None
    )
