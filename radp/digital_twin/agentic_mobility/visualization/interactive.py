"""Interactive Plotly-based visualizations for UE mobility tracks."""
from typing import List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_ue_wise_interactive(
    df: pd.DataFrame,
    ue_ids: Optional[List[int]] = None,
    show_arrows: bool = True,
    figsize: Tuple[int, int] = (900, 700),
) -> go.Figure:
    """
    Interactive UE track visualization with dropdown to select specific UEs.


    Mode A: Select UE ID → show all ticks as a track with directional flow.

    Args:
        df: DataFrame with columns ['mock_ue_id', 'tick', 'lat', 'lon']
        ue_ids: Optional list of UE IDs to include. If None, includes all
        show_arrows: If True, add directional arrows along the path
        figsize: Figure size as (width, height) tuple in pixels

    Returns:
        Plotly Figure object

    Example:
        >>> fig = plot_ue_wise_interactive(df, ue_ids=[0, 1, 2, 3, 4])
        >>> fig.show()  # Display in notebook
        >>> fig.write_html('ue_tracks.html')  # Save to file
    """
    # Filter UEs if specified
    if ue_ids is not None:
        df = df[df["mock_ue_id"].isin(ue_ids)].copy()

    if len(df) == 0:
        raise ValueError("No data to plot after filtering")

    df = df.sort_values(["mock_ue_id", "tick"]).reset_index(drop=True)
    available_ues = sorted(df["mock_ue_id"].unique())

    # Create figure
    fig = go.Figure()

    # Create a trace for each UE (all will be added but visibility controlled by dropdown)
    for ue_id in available_ues:
        ue_data = df[df["mock_ue_id"] == ue_id].sort_values("tick")

        # Main track line
        fig.add_trace(
            go.Scatter(
                x=ue_data["lon"],
                y=ue_data["lat"],
                mode="lines+markers",
                name=f"UE {ue_id}",
                line=dict(width=2),
                marker=dict(size=6),
                customdata=ue_data[["tick", "mock_ue_id"]].values,
                hovertemplate=(
                    "<b>UE %{customdata[1]}</b><br>"
                    + "Tick: %{customdata[0]}<br>"
                    + "Lat: %{y:.6f}<br>"
                    + "Lon: %{x:.6f}<extra></extra>"
                ),
                visible=(ue_id == available_ues[0]),  # Only first UE visible initially
            )
        )

        # Start point marker
        fig.add_trace(
            go.Scatter(
                x=[ue_data["lon"].iloc[0]],
                y=[ue_data["lat"].iloc[0]],
                mode="markers",
                marker=dict(size=12, color="green", symbol="circle"),
                name=f"Start UE {ue_id}",
                showlegend=False,
                hovertemplate=(
                    f"<b>START - UE {ue_id}</b><br>"
                    + f'Tick: {ue_data["tick"].iloc[0]}<br>'
                    + "Lat: %{y:.6f}<br>"
                    + "Lon: %{x:.6f}<extra></extra>"
                ),
                visible=(ue_id == available_ues[0]),
            )
        )

        # End point marker
        fig.add_trace(
            go.Scatter(
                x=[ue_data["lon"].iloc[-1]],
                y=[ue_data["lat"].iloc[-1]],
                mode="markers",
                marker=dict(size=12, color="red", symbol="square"),
                name=f"End UE {ue_id}",
                showlegend=False,
                hovertemplate=(
                    f"<b>END - UE {ue_id}</b><br>"
                    + f'Tick: {ue_data["tick"].iloc[-1]}<br>'
                    + "Lat: %{y:.6f}<br>"
                    + "Lon: %{x:.6f}<extra></extra>"
                ),
                visible=(ue_id == available_ues[0]),
            )
        )

    # Create dropdown menu to select UE
    buttons = []
    traces_per_ue = 3  # line + start marker + end marker

    for i, ue_id in enumerate(available_ues):
        # Create visibility list (True only for this UE's traces)
        visibility = [False] * (len(available_ues) * traces_per_ue)
        visibility[i * traces_per_ue : (i + 1) * traces_per_ue] = [True] * traces_per_ue

        buttons.append(
            dict(
                label=f"UE {ue_id}",
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f'UE {ue_id} - Full Track ({len(df[df["mock_ue_id"]==ue_id])} positions)'},
                ],
            )
        )

    # Update layout
    fig.update_layout(
        title=f'UE {available_ues[0]} - Full Track ({len(df[df["mock_ue_id"]==available_ues[0]])} positions)',
        xaxis_title="Longitude",
        yaxis_title="Latitude",
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
    )

    return fig


def plot_tick_wise_interactive(
    df: pd.DataFrame,
    initial_tick: int = 0,
    color_by_ue: bool = True,
    show_trails: bool = False,
) -> go.Figure:
    """
    Interactive tick-by-tick visualization showing all UEs at each moment.

    Mode B: Select tick → show all UE positions at that snapshot in time.

    Args:
        df: DataFrame with columns ['mock_ue_id', 'tick', 'lat', 'lon']
        initial_tick: Starting tick for the visualization
        color_by_ue: If True, color points by UE ID. If False, all same color
        show_trails: If True, show faint trail of previous positions

    Returns:
        Plotly Figure object with animation controls

    Example:
        >>> fig = plot_tick_wise_interactive(df)
        >>> fig.show()  # Displays with play button and tick slider
    """
    df = df.sort_values(["tick", "mock_ue_id"]).reset_index(drop=True)

    if color_by_ue:
        # Use plotly express for easy animation with color
        fig = px.scatter(
            df,
            x="lon",
            y="lat",
            animation_frame="tick",
            color="mock_ue_id",
            hover_data=["mock_ue_id", "tick"],
            title="UE Positions Over Time (Tick-wise View)",
            labels={"lon": "Longitude", "lat": "Latitude", "mock_ue_id": "UE ID"},
        )

        # Customize hover template
        fig.update_traces(
            hovertemplate=(
                "<b>UE %{customdata[0]}</b><br>"
                + "Tick: %{customdata[1]}<br>"
                + "Lat: %{y:.6f}<br>"
                + "Lon: %{x:.6f}<extra></extra>"
            )
        )
    else:
        fig = px.scatter(
            df,
            x="lon",
            y="lat",
            animation_frame="tick",
            hover_data=["mock_ue_id", "tick"],
            title="UE Positions Over Time (Tick-wise View)",
            labels={"lon": "Longitude", "lat": "Latitude"},
        )

    # Update layout for better animation controls
    fig.update_layout(
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        hovermode="closest",
    )

    # Customize animation settings
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 300  # ms per frame
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 200

    return fig
