"""Enhanced UE track visualization functions."""
from typing import List, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd


def plot_ue_tracks(
    df: pd.DataFrame,
    legend: bool = False,
    ue_ids: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None,
    show_start_points: bool = True,
    show_end_points: bool = False,
    arrow_width: float = 0.002,
    max_legend_items: int = 20,
    zoom_to_ues: bool = True,
) -> None:
    """
    Plot UE movement tracks with enhanced control options.

    Args:
        df: DataFrame with columns ['mock_ue_id', 'tick', 'lat', 'lon']
        legend: If True, show legend. If False, hide legend (default: False)
        ue_ids: Optional list of specific UE IDs to plot. If None, plot all UEs
        figsize: Figure size as (width, height) tuple
        title: Custom title for the plot. If None, auto-generate title
        show_start_points: If True, mark starting points with circles
        show_end_points: If True, mark ending points with squares
        arrow_width: Width of the directional arrows
        max_legend_items: Maximum number of items to show in legend before truncating
        zoom_to_ues: If True, auto-scale axes to fit selected UEs. If False, use full dataset bounds

    Example:
        # Plot all UEs without legend
        plot_ue_tracks(df, legend=False)

        # Plot specific UEs with legend
        plot_ue_tracks(df, legend=True, ue_ids=[0, 1, 2, 3, 4])

        # Plot single UE with full dataset context
        plot_ue_tracks(df, legend=True, ue_ids=[5], zoom_to_ues=False)
    """
    # Store full dataset bounds before filtering (for zoom_to_ues=False)
    full_lon_min, full_lon_max = df["lon"].min(), df["lon"].max()
    full_lat_min, full_lat_max = df["lat"].min(), df["lat"].max()

    # Filter by specific UE IDs if provided
    if ue_ids is not None:
        df = df[df["mock_ue_id"].isin(ue_ids)].copy()
        if len(df) == 0:
            print(f"Warning: No data found for UE IDs: {ue_ids}")
            return

    # Reset index to avoid .loc issues with filtered data
    df = df.reset_index(drop=True)

    # Identify batches (if tick resets, it's a new batch)
    batch_indices = []
    for i in range(1, len(df)):
        if df.loc[i, "tick"] == 0 and df.loc[i - 1, "tick"] != 0:
            batch_indices.append(i)
    batch_indices.append(len(df))

    # Plot each batch
    start_idx = 0
    for batch_num, end_idx in enumerate(batch_indices):
        batch_data = df.iloc[start_idx:end_idx].copy()
        batch_data = batch_data.reset_index(drop=True)

        # Create figure
        _, ax = plt.subplots(figsize=figsize)

        unique_ue_ids = sorted(batch_data["mock_ue_id"].unique())
        num_ues = len(unique_ue_ids)

        # Choose appropriate colormap
        if num_ues <= 20:
            color_map = cm.get_cmap("tab20", num_ues)
        else:
            color_map = cm.get_cmap("hsv", num_ues)

        # Plot each UE's track
        for idx, ue_id in enumerate(unique_ue_ids):
            ue_data = batch_data[batch_data["mock_ue_id"] == ue_id].copy()
            ue_data = ue_data.reset_index(drop=True)

            if len(ue_data) < 2:
                # Single point, just plot it
                ax.scatter(
                    ue_data["lon"].iloc[0],
                    ue_data["lat"].iloc[0],
                    color=color_map(idx),
                    s=50,
                    label=f"UE {ue_id}" if legend else None,
                    zorder=5,
                )
                continue

            color = color_map(idx)

            # Plot directional arrows for movement
            for i in range(len(ue_data) - 1):
                x_start = ue_data.loc[i, "lon"]
                y_start = ue_data.loc[i, "lat"]
                x_end = ue_data.loc[i + 1, "lon"]
                y_end = ue_data.loc[i + 1, "lat"]

                dx = x_end - x_start
                dy = y_end - y_start

                ax.quiver(
                    x_start,
                    y_start,
                    dx,
                    dy,
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    color=color,
                    width=arrow_width,
                    headwidth=3,
                    headlength=5,
                    zorder=3,
                )

            # Mark start points
            if show_start_points:
                ax.scatter(
                    ue_data["lon"].iloc[0],
                    ue_data["lat"].iloc[0],
                    color=color,
                    s=100,
                    marker="o",
                    edgecolors="black",
                    linewidths=1.5,
                    label=f"UE {ue_id}" if legend else None,
                    zorder=5,
                )

            # Mark end points
            if show_end_points:
                ax.scatter(
                    ue_data["lon"].iloc[-1],
                    ue_data["lat"].iloc[-1],
                    color=color,
                    s=100,
                    marker="s",
                    edgecolors="black",
                    linewidths=1.5,
                    zorder=5,
                )

        # Set title
        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")
        else:
            batch_suffix = f" - Batch {batch_num + 1}" if len(batch_indices) > 1 else ""
            ue_count_str = f"{num_ues} UE{'s' if num_ues != 1 else ''}"
            ax.set_title(f"UE Tracks with Direction ({ue_count_str}){batch_suffix}", fontsize=14, fontweight="bold")

        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)
        ax.grid(True, alpha=0.3)

        # Handle legend
        if legend:
            if num_ues <= max_legend_items:
                ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
            else:
                # Too many UEs, show truncated legend with warning
                handles, labels = ax.get_legend_handles_labels()
                truncated_handles = handles[:max_legend_items]
                truncated_labels = labels[:max_legend_items] + [f"... and {num_ues - max_legend_items} more"]
                ax.legend(
                    truncated_handles,
                    truncated_labels[: len(truncated_handles)],
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1),
                    fontsize=9,
                )
                print(
                    f"Note: Legend truncated to {max_legend_items} items. "
                    f"Total UEs: {num_ues}. Use ue_ids parameter to filter."
                )

        # Apply axis limits based on zoom_to_ues parameter
        if not zoom_to_ues:
            ax.set_xlim(full_lon_min, full_lon_max)
            ax.set_ylim(full_lat_min, full_lat_max)
        # else: use matplotlib's autoscale (zooms to visible data)

        plt.tight_layout()
        plt.show()

        start_idx = end_idx


def plot_ue_tracks_comparison(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    legend: bool = True,
    ue_ids: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (20, 8),
    titles: Optional[Tuple[str, str]] = None,
    zoom_to_ues: bool = False,
) -> None:
    """
    Plot two UE track datasets side by side for comparison.

    Args:
        df1: First DataFrame with columns ['mock_ue_id', 'tick', 'lat', 'lon']
        df2: Second DataFrame with columns ['mock_ue_id', 'tick', 'lat', 'lon']
        legend: If True, show legend. If False, hide legend
        ue_ids: Optional list of specific UE IDs to plot in both datasets
        figsize: Figure size as (width, height) tuple
        titles: Optional tuple of (title1, title2) for the two plots
        zoom_to_ues: If True, auto-scale axes to fit selected UEs. If False, use full dataset bounds

    Example:
        plot_ue_tracks_comparison(df1, df2, legend=False)
        plot_ue_tracks_comparison(df1, df2, ue_ids=[0, 1, 2], titles=("Scenario A", "Scenario B"))
        plot_ue_tracks_comparison(df1, df2, ue_ids=[0, 1], zoom_to_ues=False)
    """
    # Store full dataset bounds before filtering
    full_lon_min_1, full_lon_max_1 = df1["lon"].min(), df1["lon"].max()
    full_lat_min_1, full_lat_max_1 = df1["lat"].min(), df1["lat"].max()
    full_lon_min_2, full_lon_max_2 = df2["lon"].min(), df2["lon"].max()
    full_lat_min_2, full_lat_max_2 = df2["lat"].min(), df2["lat"].max()

    # Filter by specific UE IDs if provided
    if ue_ids is not None:
        df1 = df1[df1["mock_ue_id"].isin(ue_ids)].copy()
        df2 = df2[df2["mock_ue_id"].isin(ue_ids)].copy()

    _, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot first dataset
    title1 = titles[0] if titles else "Dataset 1"
    _plot_on_axis(
        axes[0],
        df1,
        title=title1,
        legend=legend,
        full_lon_bounds=(full_lon_min_1, full_lon_max_1),
        full_lat_bounds=(full_lat_min_1, full_lat_max_1),
        zoom_to_ues=zoom_to_ues,
    )

    # Plot second dataset
    title2 = titles[1] if titles else "Dataset 2"
    _plot_on_axis(
        axes[1],
        df2,
        title=title2,
        legend=legend,
        full_lon_bounds=(full_lon_min_2, full_lon_max_2),
        full_lat_bounds=(full_lat_min_2, full_lat_max_2),
        zoom_to_ues=zoom_to_ues,
    )

    plt.tight_layout()
    plt.show()


def _plot_on_axis(
    ax,
    df: pd.DataFrame,
    title: str,
    legend: bool = True,
    arrow_width: float = 0.002,
    full_lon_bounds: Optional[Tuple[float, float]] = None,
    full_lat_bounds: Optional[Tuple[float, float]] = None,
    zoom_to_ues: bool = True,
) -> None:
    """
    Helper function to plot UE tracks on a given matplotlib axis.

    Args:
        ax: Matplotlib axis object
        df: DataFrame with columns ['mock_ue_id', 'tick', 'lat', 'lon']
        title: Title for the plot
        legend: If True, show legend
        arrow_width: Width of the directional arrows
        full_lon_bounds: Optional (min, max) longitude bounds for full dataset
        full_lat_bounds: Optional (min, max) latitude bounds for full dataset
        zoom_to_ues: If True, auto-scale. If False, use full bounds
    """
    df = df.reset_index(drop=True)
    unique_ue_ids = sorted(df["mock_ue_id"].unique())
    num_ues = len(unique_ue_ids)

    # Choose appropriate colormap
    if num_ues <= 20:
        color_map = cm.get_cmap("tab20", num_ues)
    else:
        color_map = cm.get_cmap("hsv", num_ues)

    for idx, ue_id in enumerate(unique_ue_ids):
        ue_data = df[df["mock_ue_id"] == ue_id].copy()
        ue_data = ue_data.reset_index(drop=True)

        if len(ue_data) < 2:
            continue

        color = color_map(idx)

        # Plot arrows
        for i in range(len(ue_data) - 1):
            x_start = ue_data.loc[i, "lon"]
            y_start = ue_data.loc[i, "lat"]
            x_end = ue_data.loc[i + 1, "lon"]
            y_end = ue_data.loc[i + 1, "lat"]

            dx = x_end - x_start
            dy = y_end - y_start

            ax.quiver(
                x_start,
                y_start,
                dx,
                dy,
                angles="xy",
                scale_units="xy",
                scale=1,
                color=color,
                width=arrow_width,
                headwidth=3,
                headlength=5,
            )

        # Mark start points
        ax.scatter(
            ue_data["lon"].iloc[0],
            ue_data["lat"].iloc[0],
            color=color,
            s=80,
            marker="o",
            edgecolors="black",
            linewidths=1,
            label=f"UE {ue_id}" if legend else None,
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)

    if legend and num_ues <= 15:
        ax.legend(fontsize=8)

    # Apply axis limits based on zoom_to_ues parameter
    if not zoom_to_ues and full_lon_bounds and full_lat_bounds:
        ax.set_xlim(full_lon_bounds[0], full_lon_bounds[1])
        ax.set_ylim(full_lat_bounds[0], full_lat_bounds[1])


def plot_ue_tracks_with_topology(
    df: pd.DataFrame,
    topology_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None,
    arrow_width: float = 0.002,
) -> None:
    """
    Plot UE movement tracks with cell tower topology overlay.

    This function handles batched data and plots UE tracks with directional
    arrows along with cell tower locations.

    Args:
        df: DataFrame with columns ['mock_ue_id', 'tick', 'lat', 'lon']
        topology_df: DataFrame with columns ['cell_lat', 'cell_lon']
        figsize: Figure size as (width, height) tuple
        title: Custom title for the plot. If None, auto-generate title
        arrow_width: Width of movement arrows (default: 0.002)

    Example:
        >>> df, metadata = AgenticMobilityIntegration.generate_from_natural_language(query)
        >>> topology_df = TopologyGenerator.generate_from_llm(...)
        >>> plot_ue_tracks_with_topology(df, topology_df)
        >>> plt.savefig("output.png")
    """
    # Detect batches (where tick resets to 0)
    batch_indices = []
    for i in range(1, len(df)):
        if df.loc[i, "tick"] == 0 and df.loc[i - 1, "tick"] != 0:
            batch_indices.append(i)
    batch_indices.append(len(df))

    start_idx = 0
    for batch_num, end_idx in enumerate(batch_indices):
        batch_data = df.iloc[start_idx:end_idx]
        plt.figure(figsize=figsize)

        # Plot UE tracks with arrows
        color_map = cm.get_cmap("tab20", len(batch_data["mock_ue_id"].unique()))
        for idx, ue_id in enumerate(batch_data["mock_ue_id"].unique()):
            ue_data = batch_data[batch_data["mock_ue_id"] == ue_id]
            color = color_map(idx)

            # Draw arrows for movement
            for i in range(len(ue_data) - 1):
                x_start = ue_data.iloc[i]["lon"]
                y_start = ue_data.iloc[i]["lat"]
                x_end = ue_data.iloc[i + 1]["lon"]
                y_end = ue_data.iloc[i + 1]["lat"]
                dx = x_end - x_start
                dy = y_end - y_start
                plt.quiver(
                    x_start,
                    y_start,
                    dx,
                    dy,
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    color=color,
                    width=arrow_width,
                    headwidth=3,
                    headlength=5,
                    alpha=0.6,
                )

            # Mark starting point
            plt.scatter(
                ue_data["lon"].iloc[0],
                ue_data["lat"].iloc[0],
                color=color,
                s=30,
                alpha=0.6,
            )

        # Plot cell towers as red triangles
        plt.scatter(
            topology_df["cell_lon"],
            topology_df["cell_lat"],
            color="red",
            marker="^",
            s=150,
            edgecolors="darkred",
            linewidths=1,
            zorder=10,
            label="Cell Towers",
        )

        # Set title
        if title:
            plot_title = title
        else:
            num_ues = len(batch_data["mock_ue_id"].unique())
            num_towers = len(topology_df.groupby(["cell_lat", "cell_lon"]))
            if len(batch_indices) > 1:
                plot_title = (
                    f"UE Tracks with Cell Towers (Batch {batch_num + 1})\n{num_ues} UEs, {num_towers} Cell Sites"
                )
            else:
                plot_title = f"UE Tracks with Cell Towers\n{num_ues} UEs, {num_towers} Cell Sites"

        plt.title(plot_title)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right")
        plt.tight_layout()

        start_idx = end_idx
