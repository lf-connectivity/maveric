"""
Feature Extraction Utilities for Agentic MRO

Wraps existing MRO preprocessing functions and extracts statistical features
for LLM analysis. Uses existing code from notebooks/radp_library.py.

Based on architecture document Section 3.1 (Analyzer Agent)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

# Import existing MRO preprocessing functions
from notebooks.radp_library import (
    preprocess_ue_data,
    calc_log_distance,
    calc_relative_bearing,
    add_cell_info,
    normalize_cell_ids,
    check_cartesian_format
)


def extract_network_features(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract features from preprocessed network data.

    This function expects PREPROCESSED CSV data that already contains:
    - Cell information (cell_id, cell_lat, cell_lon, cell_az_deg, cell_carrier_freq_mhz)
    - Computed features (distance_km, relative_bearing, cell_rxpower_dbm, sinr_db)

    Args:
        df: Preprocessed DataFrame with UE and cell data

    Returns:
        Tuple of (processed_df, insights_df)
        - processed_df: Full DataFrame (copy of input)
        - insights_df: Aggregated statistics for LLM analysis

    Raises:
        ValueError: If required columns are missing

    Example:
        >>> df = pd.read_csv("simulation_data_initial.csv")
        >>> processed_df, insights = extract_network_features(df)
    """
    # Verify data is preprocessed
    required_cols = ['cell_id', 'cell_rxpower_dbm', 'sinr_db']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Please provide preprocessed CSV with cell information and computed features."
        )

    print("  ✓ Preprocessed data validated - all required columns present")

    # Use data directly
    processed_df = df.copy()

    # Extract topology from data
    topology = _extract_topology_from_data(df)
    print(f"  ✓ Extracted topology: {len(topology)} unique cells")

    # Aggregate statistics for LLM
    insights_df = _compute_insights(processed_df, topology)

    return processed_df, insights_df


def _extract_topology_from_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract unique cell information from preprocessed data to create topology.

    Args:
        df: Preprocessed DataFrame with cell columns

    Returns:
        DataFrame with unique cell information (topology)
    """
    cell_cols = ['cell_id', 'cell_lat', 'cell_lon', 'cell_az_deg', 'cell_carrier_freq_mhz']

    # Check which columns exist
    available_cols = [col for col in cell_cols if col in df.columns]

    if not available_cols:
        # Minimal topology with just cell IDs
        return pd.DataFrame({'cell_id': df['cell_id'].unique()})

    # Extract unique cell records
    topology = df[available_cols].drop_duplicates().reset_index(drop=True)

    return topology


def _compute_insights(df: pd.DataFrame, topology: pd.DataFrame) -> pd.DataFrame:
    """
    Compute aggregated insights from processed data.

    Extracts key statistics for LLM analysis:
    - UE mobility patterns
    - Signal quality metrics
    - Handover patterns
    - Problem areas

    Args:
        df: Processed DataFrame with all features
        topology: Network topology

    Returns:
        DataFrame with aggregated insights
    """
    insights = {}

    # === Basic Network Info ===
    insights['num_ues'] = df['ue_id'].nunique()
    insights['num_cells'] = topology['cell_id'].nunique()
    insights['num_ticks'] = df['tick'].nunique()
    insights['total_measurements'] = len(df)

    # === SINR Statistics ===
    if 'sinr_db' in df.columns:
        insights['sinr_mean'] = df['sinr_db'].mean()
        insights['sinr_std'] = df['sinr_db'].std()
        insights['sinr_min'] = df['sinr_db'].min()
        insights['sinr_max'] = df['sinr_db'].max()
        insights['sinr_p10'] = df['sinr_db'].quantile(0.10)
        insights['sinr_p50'] = df['sinr_db'].quantile(0.50)
        insights['sinr_p90'] = df['sinr_db'].quantile(0.90)

        # Poor signal quality detection (SINR < 0 dB)
        insights['poor_sinr_percent'] = (df['sinr_db'] < 0).sum() / len(df) * 100

    # === Rx Power Statistics ===
    if 'cell_rxpower_dbm' in df.columns:
        insights['rxpower_mean'] = df['cell_rxpower_dbm'].mean()
        insights['rxpower_std'] = df['cell_rxpower_dbm'].std()
        insights['rxpower_min'] = df['cell_rxpower_dbm'].min()
        insights['rxpower_max'] = df['cell_rxpower_dbm'].max()

    # === Mobility Metrics ===
    ue_mobility = _compute_ue_mobility(df)
    insights.update(ue_mobility)

    # === Cell Dominance ===
    cell_stats = _compute_cell_statistics(df, topology)
    insights.update(cell_stats)

    # === Handover Risk ===
    handover_metrics = _estimate_handover_patterns(df)
    insights.update(handover_metrics)

    # Convert to DataFrame for easier handling
    insights_df = pd.DataFrame([insights])

    return insights_df


def _compute_ue_mobility(df: pd.DataFrame) -> Dict:
    """
    Compute UE mobility patterns.

    Calculates velocity and movement patterns for each UE.

    Args:
        df: DataFrame with UE location and tick data

    Returns:
        Dictionary with mobility metrics
    """
    mobility_metrics = {}

    # Group by UE and compute velocities
    ue_groups = df.groupby('ue_id')

    velocities = []
    for ue_id, ue_df in ue_groups:
        ue_df = ue_df.sort_values('tick')
        if len(ue_df) < 2:
            continue

        # Compute distance traveled between ticks
        if 'longitude' in ue_df.columns and 'latitude' in ue_df.columns:
            lons = ue_df['longitude'].values
            lats = ue_df['latitude'].values

            # Simple Euclidean distance (approximation)
            distances = np.sqrt(
                np.diff(lons)**2 + np.diff(lats)**2
            )

            # Velocity = distance / time (assuming 1 tick = 1 time unit)
            ue_velocities = distances  # distance per tick
            velocities.extend(ue_velocities)

    if velocities:
        velocities = np.array(velocities)
        mobility_metrics['velocity_mean'] = velocities.mean()
        mobility_metrics['velocity_std'] = velocities.std()
        mobility_metrics['velocity_max'] = velocities.max()

        # Classify mobility levels
        # High mobility if mean velocity > threshold
        mobility_metrics['high_mobility_ues'] = int(velocities.mean() > 0.01)  # Threshold can be tuned
    else:
        mobility_metrics['velocity_mean'] = 0.0
        mobility_metrics['velocity_std'] = 0.0
        mobility_metrics['velocity_max'] = 0.0
        mobility_metrics['high_mobility_ues'] = 0

    return mobility_metrics


def _compute_cell_statistics(df: pd.DataFrame, topology: pd.DataFrame) -> Dict:
    """
    Compute cell-level statistics.

    Analyzes cell dominance and load distribution.

    Args:
        df: DataFrame with cell assignments
        topology: Network topology

    Returns:
        Dictionary with cell statistics
    """
    cell_metrics = {}

    # Find strongest cell per UE per tick
    if 'cell_rxpower_dbm' in df.columns or 'sinr_db' in df.columns:
        # Group by (ue_id, tick) and find cell with highest signal
        power_col = 'sinr_db' if 'sinr_db' in df.columns else 'cell_rxpower_dbm'

        strongest_cells = df.loc[df.groupby(['ue_id', 'tick'])[power_col].idxmax()]

        # Count attachments per cell
        cell_counts = strongest_cells['cell_id'].value_counts()

        cell_metrics['dominant_cell'] = cell_counts.idxmax()
        cell_metrics['dominant_cell_percent'] = (cell_counts.max() / cell_counts.sum()) * 100

        # Check if load is balanced
        cell_metrics['cells_serving'] = len(cell_counts)
        cell_metrics['load_balance_score'] = (cell_counts.std() / cell_counts.mean()) if cell_counts.mean() > 0 else 0.0

    return cell_metrics


def _estimate_handover_patterns(df: pd.DataFrame) -> Dict:
    """
    Estimate potential handover patterns from signal data.

    Detects situations where UEs might experience handovers.

    Args:
        df: DataFrame with signal measurements

    Returns:
        Dictionary with handover metrics
    """
    handover_metrics = {}

    # Detect potential handovers by looking at signal fluctuations
    if 'sinr_db' in df.columns:
        # Group by UE and check signal variations
        ue_groups = df.groupby('ue_id')

        fluctuations = []
        for ue_id, ue_df in ue_groups:
            ue_df = ue_df.sort_values('tick')

            # Check SINR variations per cell
            for cell_id in ue_df['cell_id'].unique():
                cell_data = ue_df[ue_df['cell_id'] == cell_id]['sinr_db']
                if len(cell_data) > 1:
                    fluctuation = cell_data.std()
                    fluctuations.append(fluctuation)

        if fluctuations:
            handover_metrics['signal_fluctuation_mean'] = np.mean(fluctuations)
            handover_metrics['signal_fluctuation_max'] = np.max(fluctuations)

            # High fluctuation suggests potential handovers
            handover_metrics['handover_risk'] = 'HIGH' if np.mean(fluctuations) > 5.0 else 'LOW'
        else:
            handover_metrics['signal_fluctuation_mean'] = 0.0
            handover_metrics['signal_fluctuation_max'] = 0.0
            handover_metrics['handover_risk'] = 'UNKNOWN'

    return handover_metrics


def format_insights_for_llm(insights_df: pd.DataFrame) -> str:
    """
    Format insights DataFrame into a human-readable string for LLM prompt.

    Args:
        insights_df: DataFrame with aggregated insights

    Returns:
        Formatted string suitable for LLM consumption
    """
    if insights_df.empty:
        return "No insights available."

    insights = insights_df.iloc[0].to_dict()

    # Build formatted string
    lines = []
    lines.append("=== Network Overview ===")
    lines.append(f"Number of UEs: {insights.get('num_ues', 'N/A')}")
    lines.append(f"Number of Cells: {insights.get('num_cells', 'N/A')}")
    lines.append(f"Number of Ticks: {insights.get('num_ticks', 'N/A')}")
    lines.append(f"Total Measurements: {insights.get('total_measurements', 'N/A')}")

    lines.append("\n=== Signal Quality (SINR) ===")
    lines.append(f"Mean SINR: {insights.get('sinr_mean', 'N/A'):.2f} dB")
    lines.append(f"Std Dev: {insights.get('sinr_std', 'N/A'):.2f} dB")
    lines.append(f"Range: [{insights.get('sinr_min', 'N/A'):.2f}, {insights.get('sinr_max', 'N/A'):.2f}] dB")
    lines.append(f"Percentiles: P10={insights.get('sinr_p10', 'N/A'):.2f}, P50={insights.get('sinr_p50', 'N/A'):.2f}, P90={insights.get('sinr_p90', 'N/A'):.2f} dB")
    lines.append(f"Poor Signal Quality: {insights.get('poor_sinr_percent', 'N/A'):.1f}% (SINR < 0 dB)")

    lines.append("\n=== Mobility Patterns ===")
    lines.append(f"Mean Velocity: {insights.get('velocity_mean', 'N/A'):.4f} units/tick")
    lines.append(f"Max Velocity: {insights.get('velocity_max', 'N/A'):.4f} units/tick")
    lines.append(f"High Mobility UEs: {insights.get('high_mobility_ues', 'N/A')}")

    lines.append("\n=== Cell Statistics ===")
    lines.append(f"Dominant Cell: {insights.get('dominant_cell', 'N/A')}")
    lines.append(f"Dominant Cell Share: {insights.get('dominant_cell_percent', 'N/A'):.1f}%")
    lines.append(f"Cells Serving: {insights.get('cells_serving', 'N/A')}")
    lines.append(f"Load Balance Score: {insights.get('load_balance_score', 'N/A'):.2f} (lower = more balanced)")

    lines.append("\n=== Handover Risk Assessment ===")
    lines.append(f"Signal Fluctuation (Mean): {insights.get('signal_fluctuation_mean', 'N/A'):.2f} dB")
    lines.append(f"Signal Fluctuation (Max): {insights.get('signal_fluctuation_max', 'N/A'):.2f} dB")
    lines.append(f"Handover Risk Level: {insights.get('handover_risk', 'N/A')}")

    return "\n".join(lines)
