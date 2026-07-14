"""
Evaluation Utilities for Agentic MRO

Wraps existing MRO evaluation functions for use in the Coordinator Agent.
Provides a clean interface to evaluate hysteresis and TTT parameters.

Based on architecture document Section 3.3 (Coordinator Agent - Step 6)
"""

import pandas as pd
from typing import Tuple

# Import existing MRO evaluation functions
from radp.digital_twin.utils.cell_selection import perform_attachment_hyst_ttt, find_hyst_diff
from apps.mobility_robustness_optimization.mobility_robustness_optimization import calculate_mro_metric


def evaluate_parameters(
    df: pd.DataFrame,
    hyst: float,
    ttt: int,
    rlf_threshold: float = -4.0
) -> Tuple[pd.DataFrame, float]:
    """
    Evaluate MRO parameters (hysteresis and TTT) and return the MRO metric score.

    This is a WRAPPER function that calls existing MRO evaluation code.

    Args:
        df: Simulation DataFrame with predicted Rx power and SINR
        hyst: Hysteresis value (dB)
        ttt: Time-to-Trigger value (ticks)
        rlf_threshold: Radio Link Failure threshold in dB (default: -4.0)

    Returns:
        Tuple of (attached_df, mro_metric_score)
        - attached_df: DataFrame with cell attachments
        - mro_metric_score: MRO metric in seconds (higher is better, max = total_time)

    Example:
        >>> attached_df, score = evaluate_parameters(sim_data, hyst=3.5, ttt=6)
        >>> print(f"Score: {score:.4f}")
        Score: 48.5234  # seconds of effective operational time
    """
    # Step 1: Perform cell attachment using existing function
    print(f"    [DEBUG] Calling perform_attachment_hyst_ttt with:")
    print(f"      - Input df shape: {df.shape}")
    print(f"      - hyst: {hyst}")
    print(f"      - ttt: {ttt}")
    print(f"      - rlf_threshold: {rlf_threshold}")

    attached_df = perform_attachment_hyst_ttt(
        ue_data=df,
        hyst=hyst,
        ttt=ttt,
        rlf_threshold=rlf_threshold
    )

    print(f"    [DEBUG] Attachment results:")
    print(f"      - Output df shape: {attached_df.shape}")
    print(f"      - Unique UEs: {attached_df['ue_id'].nunique()}")
    print(f"      - Unique ticks: {attached_df['tick'].nunique() if 'tick' in attached_df.columns else 'N/A'}")

    # Step 2: Calculate MRO metric using existing function (returns time in seconds)
    mro_metric = calculate_mro_metric(attached_df)
    print(f"    [DEBUG] MRO metric calculated: {mro_metric:.6f} seconds")

    return attached_df, mro_metric


def get_parameter_ranges(df: pd.DataFrame) -> Tuple[tuple, tuple]:
    """
    Get valid parameter ranges for hysteresis and TTT based on simulation data.

    Args:
        df: Simulation DataFrame with signal measurements

    Returns:
        Tuple of (hyst_range, ttt_range)
        - hyst_range: (min_hyst, max_hyst) in dB
        - ttt_range: (min_ttt, max_ttt) in ticks

    Example:
        >>> hyst_range, ttt_range = get_parameter_ranges(sim_data)
        >>> print(f"Hysteresis range: {hyst_range}")
        Hysteresis range: (0.0, 15.3)
        >>> print(f"TTT range: {ttt_range}")
        TTT range: (2, 101)
    """
    # Get hysteresis range using existing function
    max_diff = find_hyst_diff(df)
    hyst_range = (0.0, max_diff)

    # Get TTT range based on number of ticks
    num_ticks = df['tick'].nunique()
    ttt_range = (2, num_ticks + 1)  # Minimum TTT is 2

    return hyst_range, ttt_range


def validate_parameters(
    hyst: float,
    ttt: int,
    hyst_range: tuple,
    ttt_range: tuple
) -> Tuple[bool, str]:
    """
    Validate that parameters are within acceptable ranges.

    Args:
        hyst: Proposed hysteresis value
        ttt: Proposed TTT value
        hyst_range: Valid (min, max) for hysteresis
        ttt_range: Valid (min, max) for TTT

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if parameters are valid
        - error_message: Description of validation error (empty if valid)

    Example:
        >>> is_valid, msg = validate_parameters(3.5, 6, (0, 10), (2, 100))
        >>> assert is_valid == True
    """
    # Validate hysteresis
    if hyst < hyst_range[0] or hyst > hyst_range[1]:
        return False, f"Hysteresis {hyst} out of range {hyst_range}"

    # Validate TTT
    if ttt < ttt_range[0] or ttt > ttt_range[1]:
        return False, f"TTT {ttt} out of range {ttt_range}"

    # Validate TTT is integer
    if not isinstance(ttt, int) and ttt != int(ttt):
        return False, f"TTT must be an integer, got {ttt}"

    return True, ""


def format_evaluation_result(
    hyst: float,
    ttt: int,
    score: float,
    attached_df: pd.DataFrame
) -> dict:
    """
    Format evaluation result into a dictionary for tracking.

    Args:
        hyst: Hysteresis value tested
        ttt: TTT value tested
        score: MRO metric score achieved
        attached_df: DataFrame with attachment results

    Returns:
        Dictionary with evaluation details

    Example:
        >>> result = format_evaluation_result(3.5, 6, 0.7234, attached_df)
        >>> print(result)
        {
            'hyst': 3.5,
            'ttt': 6,
            'score': 0.7234,
            'num_handovers': 15,
            'num_rlfs': 2
        }
    """
    # Count handovers and RLFs from attached_df
    num_handovers = count_handovers(attached_df)
    num_rlfs = count_rlfs(attached_df)

    print(f"    [DEBUG] Detailed breakdown:")
    print(f"      - Handovers: {num_handovers}")
    print(f"      - RLFs: {num_rlfs}")
    print(f"      - Score: {score:.6f} seconds")

    return {
        'hyst': hyst,
        'ttt': ttt,
        'score': score,
        'num_handovers': num_handovers,
        'num_rlfs': num_rlfs
    }


def count_handovers(attached_df: pd.DataFrame) -> int:
    """
    Count number of handovers in attached DataFrame.

    A handover occurs when a UE switches from one cell to another.

    Args:
        attached_df: DataFrame with cell attachments per UE per tick

    Returns:
        Number of handovers detected
    """
    handover_count = 0

    # Group by UE
    for ue_id, ue_df in attached_df.groupby('ue_id'):
        ue_df = ue_df.sort_values('tick')

        # Count cell changes
        cell_ids = ue_df['cell_id'].values
        for i in range(1, len(cell_ids)):
            if cell_ids[i] != cell_ids[i-1] and cell_ids[i] != 'RLF':
                handover_count += 1

    return handover_count


def count_rlfs(attached_df: pd.DataFrame) -> int:
    """
    Count number of Radio Link Failures in attached DataFrame.

    Args:
        attached_df: DataFrame with cell attachments per UE per tick

    Returns:
        Number of RLFs detected
    """
    # Count rows where cell_id is 'RLF'
    if 'cell_id' in attached_df.columns:
        rlf_count = (attached_df['cell_id'] == 'RLF').sum()
    else:
        rlf_count = 0

    return rlf_count
