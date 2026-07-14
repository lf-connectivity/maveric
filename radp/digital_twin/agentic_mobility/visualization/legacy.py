from os import name

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd


def plot_ue_tracks(df: pd.DataFrame) -> None:
    """
    Plots the movement tracks of unique UE IDs on a grid of subplots.

    +-------------+------+-----------+------------+
    | mock_ue_id  | tick |   lat     |    lon     |
    +=============+======+===========+============+
    |     1       |  0   | 23.8103   | 90.4125    |
    |     1       |  1   | 23.8109   | 90.4130    |
    |     1       |  2   | 23.8115   | 90.4135    |
    |     2       |  0   | 23.8120   | 90.4140    |
    |     2       |  1   | 23.8125   | 90.4145    |
    |     2       |  2   | 23.8130   | 90.4150    |
    +-------------+------+-----------+------------+

    """

    # Initialize an empty list to store batch indices
    batch_indices = []

    # Identify where tick resets and mark the indices
    for i in range(1, len(df)):
        if df.loc[i, "tick"] == 0 and df.loc[i - 1, "tick"] != 0:
            batch_indices.append(i)

    # Add the final index to close the last batch
    batch_indices.append(len(df))

    # Now, iterate over the identified batches
    start_idx = 0
    for batch_num, end_idx in enumerate(batch_indices):
        batch_data = df.iloc[start_idx:end_idx]

        # Create a new figure
        plt.figure(figsize=(10, 6))

        # Generate a color map with different colors for each ue_id
        color_map = cm.get_cmap("tab20", len(batch_data["mock_ue_id"].unique()))

        # Plot each ue_id's movement over ticks in this batch
        for idx, ue_id in enumerate(batch_data["mock_ue_id"].unique()):
            ue_data = batch_data[batch_data["mock_ue_id"] == ue_id]
            color = color_map(idx)  # Get a unique color for each ue_id

            # Plot the path with arrows
            for i in range(len(ue_data) - 1):
                x_start = ue_data.iloc[i]["lon"]
                y_start = ue_data.iloc[i]["lat"]
                x_end = ue_data.iloc[i + 1]["lon"]
                y_end = ue_data.iloc[i + 1]["lat"]

                # Calculate the direction vector
                dx = x_end - x_start
                dy = y_end - y_start

                # Plot the line with an arrow with reduced width and unique color
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

            # Plot starting points as circles with the same color
            plt.scatter(
                ue_data["lon"].iloc[0],
                ue_data["lat"].iloc[0],
                color=color,
                label=f"Start UE {ue_id}",
            )

        # Set plot title and labels
        plt.title(f"UE Tracks with Direction for Batch {batch_num + 1}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1))

        # Display the plot
        plt.show()

        # Update start_idx for the next batch
        start_idx = end_idx


def plot_ue_tracks_side_by_side(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    """
    Plots the movement tracks of unique UE IDs from two DataFrames side by side.

    df1:
    +-------------+-----------+------------+
    | mock_ue_id  |   lat     |    lon     |
    +=============+===========+============+
    |     1       | 23.8101   | 90.4100    |
    |     2       | 23.8105   | 90.4110    |
    |     3       | 23.8110   | 90.4120    |
    |     4       | 23.8115   | 90.4130    |
    +-------------+-----------+------------+

    df2:
    +-------------+-----------+------------+
    | mock_ue_id  |   lat     |    lon     |
    +=============+===========+============+
    |     1       | 23.8120   | 90.4140    |
    |     2       | 23.8125   | 90.4150    |
    |     3       | 23.8130   | 90.4160    |
    |     4       | 23.8135   | 90.4170    |
    +-------------+-----------+------------+

    """
    # Set up subplots with 2 columns for side by side plots
    fig, axes = plt.subplots(1, 2, figsize=(25, 10))  # 2 rows, 2 columns (side by side)

    # Plot the first DataFrame
    plot_ue_tracks_on_axis(df1, axes[0], title="DataFrame 1")

    # Plot the second DataFrame
    plot_ue_tracks_on_axis(df2, axes[1], title="DataFrame 2")

    # Adjust layout and show
    plt.tight_layout()
    plt.show()


def plot_ue_tracks_on_axis(df: pd.DataFrame, ax, title: str) -> None:
    """
    Helper function to plot UE tracks on a given axis.

    +-------------+-----------+------------+
    | mock_ue_id  |   lat     |    lon     |
    +=============+===========+============+
    |     1       | 23.8103   | 90.4125    |
    |     1       | 23.8109   | 90.4130    |
    |     1       | 23.8115   | 90.4135    |
    |     2       | 23.8120   | 90.4140    |
    |     2       | 23.8125   | 90.4145    |
    |     2       | 23.8130   | 90.4150    |
    +-------------+-----------+------------+


    """
    data = df
    unique_ids = data["mock_ue_id"].unique()
    num_plots = len(unique_ids)

    color_map = cm.get_cmap("tab20", num_plots)

    for idx, ue_id in enumerate(unique_ids):
        ue_data = data[data["mock_ue_id"] == ue_id]

        for i in range(len(ue_data) - 1):
            x_start = ue_data.iloc[i]["lon"]
            y_start = ue_data.iloc[i]["lat"]
            x_end = ue_data.iloc[i + 1]["lon"]
            y_end = ue_data.iloc[i + 1]["lat"]

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
                color=color_map(idx),
            )

        ax.scatter(ue_data["lon"], ue_data["lat"], color=color_map(idx), label=f"UE {ue_id}")

    ax.set_title(title)
    ax.legend()


if name == "__main__":
    ue_df = pd.read_csv(
        "radp/digital_twin/agentic_mobility/visualization/visualization_outputs/agentic_mobility_20UE_15ticks.csv"
    )
    plot_ue_tracks(ue_df)
