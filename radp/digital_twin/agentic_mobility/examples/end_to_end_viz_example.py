"""End-to-end example: Natural language → Mobility simulation + Visualization + Topology."""
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from radp.digital_twin.agentic_mobility.integration import AgenticMobilityIntegration
from radp.digital_twin.agentic_mobility.topology_generator import TopologyGenerator
from radp.digital_twin.mobility.param_regression import get_predicted_alpha, preprocess_ue_data

# ----------------------------------------------------------------------
# Visualization functions
# ----------------------------------------------------------------------
def plot_ue_tracks_with_topology(df: pd.DataFrame, topology_df: pd.DataFrame) -> None:
    """Plots the movement tracks of unique UE IDs with cell tower locations."""
    batch_indices = []
    for i in range(1, len(df)):
        if df.loc[i, "tick"] == 0 and df.loc[i - 1, "tick"] != 0:
            batch_indices.append(i)
    batch_indices.append(len(df))

    start_idx = 0
    for batch_num, end_idx in enumerate(batch_indices):
        batch_data = df.iloc[start_idx:end_idx]
        plt.figure(figsize=(12, 8))

        # Plot UE tracks
        color_map = cm.get_cmap("tab20", len(batch_data["mock_ue_id"].unique()))
        for idx, ue_id in enumerate(batch_data["mock_ue_id"].unique()):
            ue_data = batch_data[batch_data["mock_ue_id"] == ue_id]
            color = color_map(idx)
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
                    width=0.002,
                    headwidth=3,
                    headlength=5,
                    alpha=0.6,
                )
            plt.scatter(
                ue_data["lon"].iloc[0],
                ue_data["lat"].iloc[0],
                color=color,
                s=30,
                alpha=0.6,
            )

        # Plot cell towers - simple triangles only
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

        plt.title(f"UE Tracks with Cell Towers for Batch {batch_num + 1}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig("ue_tracks_with_topology.png", dpi=150)
        print(f"Saved visualization to: ue_tracks_with_topology.png")
        start_idx = end_idx


def plot_ue_tracks(df: pd.DataFrame) -> None:
    """Plots the movement tracks of unique UE IDs on a grid of subplots."""
    batch_indices = []
    for i in range(1, len(df)):
        if df.loc[i, "tick"] == 0 and df.loc[i - 1, "tick"] != 0:
            batch_indices.append(i)
    batch_indices.append(len(df))

    start_idx = 0
    for batch_num, end_idx in enumerate(batch_indices):
        batch_data = df.iloc[start_idx:end_idx]
        plt.figure(figsize=(10, 6))
        color_map = cm.get_cmap("tab20", len(batch_data["mock_ue_id"].unique()))
        for idx, ue_id in enumerate(batch_data["mock_ue_id"].unique()):
            ue_data = batch_data[batch_data["mock_ue_id"] == ue_id]
            color = color_map(idx)
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
                    width=0.002,
                    headwidth=3,
                    headlength=5,
                )
            plt.scatter(
                ue_data["lon"].iloc[0],
                ue_data["lat"].iloc[0],
                color=color,
            )
        plt.title(f"UE Tracks with Direction for Batch {batch_num + 1}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1))
        plt.savefig("ue_tracks.png")
        start_idx = end_idx


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
def main():
    print("=" * 80)
    print("Agentic Mobility Generation - End-to-End Example with Visualization & Topology")
    print("Natural Language → Parameters → Simulation → DataFrame → Topology → Plot")
    print("=" * 80)

    query1 = (
        "Give me for Tokyo. consider it as a suburban area with lots of pedestrians and cars. "
        "There are so many motorbikes, consider them as cyclists. Have two hundreds total devices."
    )

    print(f"Query: '{query1}'\nProcessing...")

    df1, metadata1 = AgenticMobilityIntegration.generate_from_natural_language(query1)
    print(f"\n✓ Generated {len(df1)} mobility points for {metadata1['query_intent']['num_ues']} UEs.")

    # Optional: preprocess and alpha prediction
    # ue_data = preprocess_ue_data(df1)
    # alpha_val = get_predicted_alpha(ue_data)
    # print(f"Predicted alpha: {alpha_val:.3f}")

    # Generate topology using LLM-based parameter generation
    print("\n📡 Generating cell topology using LLM...")
    location_data = metadata1.get("location_data", {})
    query_intent = metadata1.get("query_intent", {})

    # Save parameters to JSON for debugging
    params_info = {
        "query": query1,
        "location_data": location_data,
        "query_intent": query_intent,
        "mobility_metadata": {
            "retry_count": metadata1.get("retry_count"),
            "validation_warnings": metadata1.get("validation_warnings"),
        },
    }

    with open("generation_params.json", "w") as f:
        json.dump(params_info, f, indent=2)
    print(f"💾 Saved generation parameters to: generation_params.json")

    topology_df = TopologyGenerator.generate_from_llm(
        area_type=location_data.get("area_type", "suburban"),
        num_ues=query_intent.get("num_ues", 200),
        min_lat=location_data.get("min_lat", df1["lat"].min()),
        max_lat=location_data.get("max_lat", df1["lat"].max()),
        min_lon=location_data.get("min_lon", df1["lon"].min()),
        max_lon=location_data.get("max_lon", df1["lon"].max()),
        raw_query=query1,
    )

    print(f"✓ Generated {len(topology_df)} cell sectors")

    # Count unique locations
    unique_locations = topology_df.groupby(['cell_lat', 'cell_lon']).size()
    print(f"✓ Generated {len(unique_locations)} unique cell sites")
    print(f"\nTopology preview:")
    print(topology_df.to_string())

    # Save topology to CSV
    topology_csv_path = "cell_topology.csv"
    topology_df.to_csv(topology_csv_path, index=False)
    print(f"\n💾 Saved topology to: {topology_csv_path}")

    # Print boundary info
    print(f"\nBoundary Check:")
    print(f"  Lat boundaries: {location_data.get('min_lat', df1['lat'].min()):.6f} to {location_data.get('max_lat', df1['lat'].max()):.6f}")
    print(f"  Lon boundaries: {location_data.get('min_lon', df1['lon'].min()):.6f} to {location_data.get('max_lon', df1['lon'].max()):.6f}")
    print(f"  Cell lat range: {topology_df['cell_lat'].min():.6f} to {topology_df['cell_lat'].max():.6f}")
    print(f"  Cell lon range: {topology_df['cell_lon'].min():.6f} to {topology_df['cell_lon'].max():.6f}")

    # Visualize with topology
    print("\n📈 Visualizing UE mobility tracks with cell towers...")
    plot_ue_tracks_with_topology(df1, topology_df)

    print("\nExample complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
