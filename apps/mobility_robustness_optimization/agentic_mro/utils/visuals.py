import os
from typing import List, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from radp.digital_twin.utils.constants import LATENT_BACKGROUND_NOISE_DB, RLF_THRESHOLD


def plot_sinr_db_by_ue(
    df: pd.DataFrame,
    df2: pd.DataFrame,
    ue_ids: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (1200, 700),
    rlf_threshold: float = RLF_THRESHOLD,
) -> go.Figure:
    """
    Interactive Plotly visualization of SINR (in dB) over ticks with dropdown to select UEs.

    - Solid bold line: Connected cell_id (from df), color-coded.
    - Dotted lines: All cell_id sinr_db values from df2 for context.
    - RLF events: Drop to bottom with bold markers.
    - RLF_THRESHOLD: Horizontal dashed line.

    Parameters:
        df (pd.DataFrame): Connected cell data: 'ue_id', 'tick', 'sinr_db', 'cell_id' (or 'RLF').
        df2 (pd.DataFrame): All candidate cell data: 'ue_id', 'tick', 'cell_id', 'sinr_db'.
        ue_ids (Optional[List[int]]): List of UE IDs to include. If None, includes all UEs.
        figsize (Tuple[int, int]): Figure size as (width, height) tuple in pixels.
        rlf_threshold (float): RLF threshold value in dB.

    Returns:
        go.Figure: Plotly Figure object with interactive UE dropdown selector.

    Example:
        >>> fig = plot_sinr_db_by_ue(df_connected, df_all_candidates, ue_ids=[0, 1, 2])
        >>> fig.show()  # Display in notebook
        >>> fig.write_html('sinr_plot.html')  # Save to file

    Data format:
    +--------+------+----------+----------+
    | ue_id  | tick | cell_id  | sinr_db  |
    +========+======+==========+==========+
    |   0    |  0   |    1     |  14.0    |
    |   0    |  0   |    2     |  12.5    |
    |   1    |  0   |    1     |  13.2    |
    |   1    |  0   |    2     |  10.8    |
    +--------+------+----------+----------+
    """
    # Filter UEs if specified
    if ue_ids is not None:
        df = df[df["ue_id"].isin(ue_ids)].copy()
        df2 = df2[df2["ue_id"].isin(ue_ids)].copy()

    if len(df) == 0 or len(df2) == 0:
        raise ValueError("No data to plot after filtering")

    # Get available UEs
    available_ues = sorted(df["ue_id"].unique())

    # Create color map for cells
    base_colors = {"1": "#EF4444", "2": "#10B981", "3": "#3B82F6", 1: "#EF4444", 2: "#10B981", 3: "#3B82F6"}
    all_cell_ids = pd.concat([df2["cell_id"], df[df["cell_id"] != "RLF"]["cell_id"]]).unique()
    plotly_colors = ["#F59E0B", "#8B5CF6", "#EC4899", "#14B8A6", "#F97316", "#06B6D4", "#84CC16", "#A855F7"]
    missing_ids = [cid for cid in all_cell_ids if cid not in base_colors and cid != "RLF"]
    dynamic_colors = {cid: plotly_colors[i % len(plotly_colors)] for i, cid in enumerate(missing_ids)}
    full_color_map = {**base_colors, **dynamic_colors}

    # Create figure
    fig = go.Figure()

    # Track traces per UE for visibility control
    traces_per_ue_list = []

    # Create traces for each UE
    for ue_id in available_ues:
        ue_df = df[df["ue_id"] == ue_id].sort_values("tick").reset_index(drop=True)
        ue_df2 = df2[df2["ue_id"] == ue_id].sort_values("tick")

        if ue_df.empty or ue_df2.empty:
            continue

        # Calculate drop value for RLF markers
        min_sinr = min(ue_df2["sinr_db"].min(), ue_df[ue_df["cell_id"] != "RLF"]["sinr_db"].min())
        drop_value = min_sinr - 5

        trace_count = 0
        is_first_ue = (ue_id == available_ues[0])

        # --- Plot all candidate cell SINRs (dotted lines) ---
        for cell_id, group in ue_df2.groupby("cell_id"):
            color = full_color_map.get(cell_id, "#6B7280")
            fig.add_trace(
                go.Scatter(
                    x=group["tick"],
                    y=group["sinr_db"],
                    mode="lines",
                    name=f"Cell {cell_id} (candidate)",
                    line=dict(width=2.5, dash="dot", color=color),
                    opacity=0.7,
                    legendgroup=f"ue{ue_id}",
                    showlegend=True,
                    hovertemplate=(
                        f"<b>UE {ue_id} - Cell {cell_id} (candidate)</b><br>"
                        + "Tick: %{x}<br>"
                        + "SINR: %{y:.2f} dB<extra></extra>"
                    ),
                    visible=is_first_ue,
                )
            )
            trace_count += 1

        # --- Plot connected UE SINR as continuous lines (color-coded per cell_id) ---
        cell_groups = {}
        for i in range(len(ue_df)):
            cell_id = ue_df.loc[i, "cell_id"]
            if cell_id != "RLF":
                if cell_id not in cell_groups:
                    cell_groups[cell_id] = {"ticks": [], "sinr": []}
                cell_groups[cell_id]["ticks"].append(ue_df.loc[i, "tick"])
                cell_groups[cell_id]["sinr"].append(ue_df.loc[i, "sinr_db"])

        for cell_id, data in cell_groups.items():
            color = full_color_map.get(cell_id, "#6B7280")
            fig.add_trace(
                go.Scatter(
                    x=data["ticks"],
                    y=data["sinr"],
                    mode="lines+markers",
                    name=f"Cell {cell_id} (connected)",
                    line=dict(width=3, color=color),
                    marker=dict(size=6, color=color),
                    legendgroup=f"ue{ue_id}",
                    showlegend=True,
                    hovertemplate=(
                        f"<b>UE {ue_id} - Cell {cell_id} (CONNECTED)</b><br>"
                        + "Tick: %{x}<br>"
                        + "SINR: %{y:.2f} dB<extra></extra>"
                    ),
                    visible=is_first_ue,
                )
            )
            trace_count += 1

        # --- Plot RLF events ---
        rlf_ticks = ue_df[ue_df["cell_id"] == "RLF"]["tick"].tolist()
        if rlf_ticks:
            fig.add_trace(
                go.Scatter(
                    x=rlf_ticks,
                    y=[drop_value] * len(rlf_ticks),
                    mode="markers",
                    name="RLF Event",
                    marker=dict(size=12, color="black", symbol="x"),
                    legendgroup=f"ue{ue_id}",
                    showlegend=True,
                    hovertemplate=(
                        f"<b>UE {ue_id} - RLF EVENT</b><br>"
                        + "Tick: %{x}<br>"
                        + "Radio Link Failure<extra></extra>"
                    ),
                    visible=is_first_ue,
                )
            )
            trace_count += 1

        # --- RLF Threshold line ---
        fig.add_trace(
            go.Scatter(
                x=[ue_df2["tick"].min(), ue_df2["tick"].max()],
                y=[rlf_threshold, rlf_threshold],
                mode="lines",
                name=f"RLF Threshold ({rlf_threshold} dB)",
                line=dict(width=2, dash="dash", color="black"),
                legendgroup=f"ue{ue_id}",
                showlegend=True,
                hovertemplate=f"RLF Threshold: {rlf_threshold} dB<extra></extra>",
                visible=is_first_ue,
            )
        )
        trace_count += 1

        traces_per_ue_list.append(trace_count)

    # Create dropdown menu to select UE
    buttons = []
    trace_offset = 0

    for i, ue_id in enumerate(available_ues):
        traces_count = traces_per_ue_list[i]

        # Create visibility list (True only for this UE's traces)
        visibility = [False] * sum(traces_per_ue_list)
        visibility[trace_offset : trace_offset + traces_count] = [True] * traces_count

        buttons.append(
            dict(
                label=f"UE {ue_id}",
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f"SINR over Time for UE {ue_id}"},
                ],
            )
        )

        trace_offset += traces_count

    # Update layout
    fig.update_layout(
        title=f"SINR over Time for UE {available_ues[0]}",
        xaxis_title="Tick",
        yaxis_title="SINR (dB)",
        width=figsize[0],
        height=figsize[1],
        hovermode="closest",
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                x=0.02,
                xanchor="left",
                y=0.98,
                yanchor="top",
            )
        ],
        showlegend=True,
        legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
    )

    return fig


def mro_plot_scatter(df: pd.DataFrame, topology: pd.DataFrame, save_path: str, rlf_threshold=RLF_THRESHOLD) -> None:
    """
    Plot a scatter plot of cell towers and UE (User Equipment) locations and saves to file.
    @param df: DataFrame containing UE data with columns 'loc_x', 'loc_y', 'cell_id', and 'sinr_db'.
    @param topology: DataFrame containing cell tower data with columns 'cell_lon', 'cell_lat', and 'cell_id'.
    @param save_path: File path where the plot will be saved (e.g., '/path/to/plot.png').
    @returns: None. Saves a scatter plot with cell towers and UE locations.

    df:
    +---------+--------+--------+----------+
    | cell_id | loc_x  | loc_y  | sinr_db  |
    +=========+========+========+==========+
    |    1    | 90.412 | 23.810 |   15.3   |
    |    2    | 90.413 | 23.811 |   12.1   |
    |    1    | 90.415 | 23.812 |   18.7   |
    |    2    | 90.416 | 23.813 |    5.5   |
    +---------+--------+--------+----------+

    topology:
    +---------+----------+----------+
    | cell_id | cell_lon | cell_lat |
    +=========+==========+==========+
    |    1    | 90.410   | 23.809   |
    |    2    | 90.414   | 23.810   |
    +---------+----------+----------+

    """

    # Create a figure and axis
    plt.figure(figsize=(10, 8))

    plt.scatter([], [], color="grey", label="RLF")

    # Define color mapping based on cell_id for both cells and UEs (dynamic)
    base_colors = {1: "red", 2: "green", 3: "blue"}
    all_cell_ids = pd.concat([topology["cell_id"], df["cell_id"]]).unique()
    missing_ids = [cid for cid in all_cell_ids if cid not in base_colors]
    extra_colors = cm.get_cmap("tab10", len(missing_ids))
    dynamic_colors = {cid: extra_colors(i) for i, cid in enumerate(missing_ids)}
    color_map = {**base_colors, **dynamic_colors}

    # Plot UEs from df without labels but with the same color coding
    # Plot UEs first with lower zorder so they appear underneath cell towers
    for _, row in df.iterrows():
        color = color_map.get(row["cell_id"], "black")  # Default to black if unknown cell_id
        if row["sinr_db"] < rlf_threshold:  # REMOVE COMMENT WHEN sinr_db IS FIXED
            color = "grey"  # Change to grey if sinr_db < 2

        plt.scatter(row["loc_x"], row["loc_y"], color=color, zorder=1)

    # Plot cell towers from the topology dataframe with triangle markers and corresponding colors
    # Plot cell towers with higher zorder so they appear on top
    for _, row in topology.iterrows():
        color = color_map.get(row["cell_id"], "black")  # Default to black if unknown cell_id
        plt.scatter(
            row["cell_lon"],
            row["cell_lat"],
            marker="^",
            color=color,
            s=200,
            label=f"Cell {row['cell_id']}",
            zorder=2,
            edgecolors="black",
            linewidths=1.5,
        )

    # Add labels and title
    plt.xlabel("Longitude (loc_x)")
    plt.ylabel("Latitude (loc_y)")
    plt.title("Cell Towers and UE Locations")

    # Create a legend for the cells only
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def add_sinr_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'sinr_db' column to the input DataFrame, computing the Signal-to-Interference-plus-Noise Ratio (SINR)
    for each UE–cell pair based on received signal power, background noise, and interference.

    Parameters:
        df (pd.DataFrame): DataFrame with 'ue_id', 'cell_rxpower_dbm', and 'cell_carrier_freq_mhz' per row.

    Returns:
        pd.DataFrame: Updated DataFrame with an additional 'sinr_db' column.

    +--------+---------+------------------+------------------------+
    | ue_id  | cell_id | cell_rxpower_dbm | cell_carrier_freq_mhz  |
    +========+=========+==================+========================+
    |   0    |    1    |   -100.311970    |         2100.0         |
    |   0    |    2    |    -99.841523    |         2100.0         |
    |   1    |    1    |   -100.294405    |         2100.0         |
    |   1    |    2    |   -100.132420    |         2100.0         |
    |   2    |    1    |   -100.650003    |         2100.0         |
    |   2    |    2    |   -100.456381    |         2100.0         |
    |   3    |    1    |   -100.987321    |         2100.0         |
    |   3    |    2    |   -100.864529    |         2100.0         |
    +--------+---------+------------------+------------------------+

    """
    df = df.copy()
    sinr_column = []

    # Group by location
    for (_, group) in df.groupby(["ue_id", "tick"]):
        # Group further by frequency layer within the same location
        freq_groups = group.groupby("cell_carrier_freq_mhz")

        # Create a temporary Series to store sinr values for current group
        group_sinr_values = pd.Series(index=group.index, dtype=float)

        for freq, freq_group in freq_groups:
            # List of all rx powers in this frequency group
            all_rxpowers = freq_group["cell_rxpower_dbm"].tolist()
            noise_db = LATENT_BACKGROUND_NOISE_DB

            for idx, row in freq_group.iterrows():
                serving_power = row["cell_rxpower_dbm"]
                # Remove this row's signal from interference
                interference_others = [p for p in all_rxpowers if p != serving_power or all_rxpowers.count(p) > 1]
                sinr_db = _compute_row_level_sinr(serving_power, interference_others, noise_db)
                group_sinr_values.at[idx] = sinr_db

        sinr_column.append(group_sinr_values)

    # Combine all the sinr values and add to DataFrame
    df["sinr_db"] = pd.concat(sinr_column).sort_index()

    return df


# Compute SINR for each row (UE–cell pair), given its group
def _compute_row_level_sinr(signal_dbm: float, interference_dbm_list: list, noise_db: float) -> float:
    """
        Computes the SINR for a single UE–cell pair by removing interference
        and noise from the received signal power.

    Parameters:
            row (pd.Series): Current row containing signal data.
            group (pd.DataFrame): Group of UE–cell rows sharing the same UE and frequency.

        +--------+---------+------------------+------------------------+
        | ue_id  | cell_id | cell_rxpower_dbm | cell_carrier_freq_mhz |
        +========+=========+==================+========================+
        |   0    |    1    |   -100.311970    |         2100.0         |
        |   0    |    2    |    -99.841523    |         2100.0         |
        |   1    |    1    |   -100.294405    |         2100.0         |
        |   1    |    2    |   -100.132420    |         2100.0         |
        |   2    |    1    |   -100.650003    |         2100.0         |
        |   2    |    2    |   -100.456381    |         2100.0         |
        |   3    |    1    |   -100.987321    |         2100.0         |
        |   3    |    2    |   -100.864529    |         2100.0         |
        +--------+---------+------------------+------------------------+


        Returns:
            float: The computed SINR value in decibels for the current UE–cell pair.
    """
    signal_linear = 10 ** (signal_dbm / 10)
    interference_linear = sum(10 ** (p / 10) for p in interference_dbm_list)
    noise_linear = 10 ** (noise_db / 10)

    sinr_linear = signal_linear / (interference_linear + noise_linear)
    return 10 * np.log10(sinr_linear)
