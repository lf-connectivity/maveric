# Location - radp/digital_twin/mro/mro.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple

from radp.digital_twin.utils import constants
from radp.digital_twin.utils.gis_tools import GISTools
from radp.digital_twin.rf.bayesian.bayesian_engine import (
    BayesianDigitalTwin,
    NormMethod,
)
from notebooks.radp_library import get_percell_data
from radp.digital_twin.utils.cell_selection import perform_attachment


class MobilityRobustnessOptimization:
    """
    A class to perform Mobility Robustness Optimization (MRO) using Bayesian Digital Twins. This class integrates
    user equipment (UE) data with cell topology to predict the received power at various UE locations and
    determines the optimal cell attachment based on these predictions. The class uses Bayesian modeling to
    accurately forecast signal strength, accounting for factors such as distance, frequency, and antenna characteristics.
    """

    def __init__(
        self,
        ue_data: pd.DataFrame,
        topology: pd.DataFrame,
        prediction_data: pd.DataFrame,
        tx_power_dbm: int = 23,
    ):
        self.ue_data = ue_data
        self.topology = topology
        self.tx_power_dbm = tx_power_dbm
        self.full_data = self._preprocess_ue_topology_data()
        self.prediction_data = prediction_data
        self.bayesian_digital_twins = {}

    def _connect_ue_to_all_cells(self, pred_data: bool = False) -> pd.DataFrame:
        """
        Connects each user equipment (UE) entry to all cells in the topology for each tick,
        effectively creating a Cartesian product of UEs and cells, which includes data from both sources.
        """
        # Create copies to avoid modifying class attributes directly
        if pred_data:
            ue_data_tmp = self.prediction_data.copy()
        ue_data_tmp = self.ue_data.copy()
        topology_tmp = self.topology.copy()

        # Remove the 'cell_' prefix and convert cell_id to integer if needed
        if self.topology["cell_id"].dtype == object:
            self.topology["cell_id"] = (
                self.topology["cell_id"].str.replace("cell_", "").astype(int)
            )
        ue_data_tmp["key"] = 1
        topology_tmp["key"] = 1
        combined_df = pd.merge(ue_data_tmp, topology_tmp, on="key").drop("key", axis=1)

        return combined_df

    def _calculate_received_power(
        self, distance_km: float, frequency_mhz: int
    ) -> float:
        """
        Calculate received power using the Free-Space Path Loss (FSPL) model.
        """
        # Convert distance from kilometers to meters
        distance_m = distance_km * 1000

        # Calculate Free-Space Path Loss (FSPL) in dB
        fspl_db = 20 * np.log10(distance_m) + 20 * np.log10(frequency_mhz) - 27.55

        # Calculate and return the received power in dBm
        received_power_dbm = self.tx_power_dbm - fspl_db
        return received_power_dbm

    def _preprocess_ue_topology_data(self) -> pd.DataFrame:
        full_data = self._connect_ue_to_all_cells()
        full_data["log_distance"] = full_data.apply(
            lambda row: GISTools.get_log_distance(
                row["latitude"], row["longitude"], row["cell_lat"], row["cell_lon"]
            ),
            axis=1,
        )

        full_data["cell_rxpwr_dbm"] = full_data.apply(
            lambda row: self._calculate_received_power(
                row["log_distance"], row["cell_carrier_freq_mhz"]
            ),
            axis=1,
        )

        return full_data

    def _preprocess_ue_training_data(self) -> pd.DataFrame:
        data = self.full_data.copy()
        train_per_cell_df = [x for _, x in data.groupby("cell_id")]
        n_cell = len(self.topology.index)

        metadata_df = pd.DataFrame(
            {
                "cell_id": [cell_id for cell_id in self.topology.cell_id],
                "idx": [i + 1 for i in range(n_cell)],
            }
        )
        idx_cell_id_mapping = dict(zip(metadata_df.idx, metadata_df.cell_id))
        desired_idxs = [1 + r for r in range(n_cell)]

        n_samples_train = []
        for df in train_per_cell_df:
            n_samples_train.append(df.shape[0])

        train_per_cell_df_processed = []
        for i in range(n_cell):
            train_per_cell_df_processed.append(
                get_percell_data(
                    data_in=train_per_cell_df[i],
                    choose_strongest_samples_percell=False,
                    n_samples=n_samples_train[i],
                )[0][0]
            )

        training_data = {}

        for i, df in enumerate(train_per_cell_df_processed):
            train_cell_id = idx_cell_id_mapping[i + 1]
            training_data[train_cell_id] = df

        for train_cell_id, training_data_idx in training_data.items():
            training_data_idx["cell_id"] = train_cell_id
            training_data_idx["cell_lat"] = self.topology[
                self.topology["cell_id"] == train_cell_id
            ]["cell_lat"].values[0]
            training_data_idx["cell_lon"] = self.topology[
                self.topology["cell_id"] == train_cell_id
            ]["cell_lon"].values[0]
            training_data_idx["cell_az_deg"] = self.topology[
                self.topology["cell_id"] == train_cell_id
            ]["cell_az_deg"].values[0]
            training_data_idx["cell_carrier_freq_mhz"] = self.topology[
                self.topology["cell_id"] == train_cell_id
            ]["cell_carrier_freq_mhz"].values[0]
            training_data_idx["relative_bearing"] = [
                GISTools.get_relative_bearing(
                    training_data_idx["cell_az_deg"].values[0],
                    training_data_idx["cell_lat"].values[0],
                    training_data_idx["cell_lon"].values[0],
                    lat,
                    lon,
                )
                for lat, lon in zip(
                    training_data_idx["latitude"], training_data_idx["longitude"]
                )
            ]

        return training_data

    def _preprocess_prediction_data(self) -> pd.DataFrame:
        data = self._connect_ue_to_all_cells(pred_data=True)

        data["log_distance"] = data.apply(
            lambda row: GISTools.get_log_distance(
                row["latitude"], row["longitude"], row["cell_lat"], row["cell_lon"]
            ),
            axis=1,
        )
        data["cell_rxpwr_dbm"] = data.apply(
            lambda row: self._calculate_received_power(
                row["log_distance"], row["cell_carrier_freq_mhz"]
            ),
            axis=1,
        )

        data["relative_bearing"] = data.apply(
            lambda row: GISTools.get_relative_bearing(
                row["cell_az_deg"],
                row["cell_lat"],
                row["cell_lon"],
                row["latitude"],
                row["longitude"],
            ),
            axis=1,
        )
        return data

    def training(self, maxiter: int) -> List[float]:
        """
        Trains the Bayesian Digital Twins for each cell in the topology using the UE locations and features
        like log distance, relative bearing, and cell received power (Rx power).
        """
        training_data = self._preprocess_ue_training_data()
        bayesian_digital_twins = {}
        loss_vs_iters = []
        for train_cell_id, training_data_idx in training_data.items():
            bayesian_digital_twins[train_cell_id] = BayesianDigitalTwin(
                data_in=[training_data_idx],
                x_columns=["log_distance", "relative_bearing"],
                y_columns=["cell_rxpwr_dbm"],
                norm_method=NormMethod.MINMAX,
            )
            self.bayesian_digital_twins[train_cell_id] = bayesian_digital_twins[
                train_cell_id
            ]
            loss_vs_iters.append(
                bayesian_digital_twins[train_cell_id].train_distributed_gpmodel(
                    maxiter=maxiter,
                )
            )
        return loss_vs_iters

    def predictions(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Predicts the received power for each User Equipment (UE) at different locations and ticks using Bayesian Digital Twins.
        It then determines the best cell for each UE to attach based on the predicted power values.
        """
        prediction_data = self._preprocess_prediction_data()
        full_prediction_df = pd.DataFrame()

        # Loop over each 'tick'
        for tick, tick_df in prediction_data.groupby("tick"):
            # Loop over each 'cell_id' within the current 'tick'
            for cell_id, cell_df in tick_df.groupby("cell_id"):
                # Perform the Bayesian prediction
                pred_means_percell, _ = self.bayesian_digital_twins[
                    cell_id
                ].predict_distributed_gpmodel(prediction_dfs=[cell_df])

                # Assuming 'pred_means_percell' returns a list of predictions corresponding to the DataFrame index
                cell_df["pred_means"] = pred_means_percell[0]

                # Include additional necessary columns for the final DataFrame
                cell_df["tick"] = tick
                cell_df["cell_id"] = cell_id

                # Append the predictions to the full DataFrame
                full_prediction_df = pd.concat(
                    [full_prediction_df, cell_df], ignore_index=True
                )
        full_prediction_df = full_prediction_df.rename(
            columns={"latitude": "loc_y", "longitude": "loc_x"}
        )
        predicted = perform_attachment(full_prediction_df, self.topology)

        return predicted, full_prediction_df


# Scatter plot of the Cell towers and UE Locations


def mro_plot_scatter(df, topology):
    # Create a figure and axis
    plt.figure(figsize=(10, 8))

    plt.scatter([], [], color="grey", label="RLF")

    # Define color mapping based on cell_id for both cells and UEs
    color_map = {1: "red", 2: "green", 3: "blue"}

    # Plot cell towers from the topology dataframe with 'X' markers and corresponding colors
    for _, row in topology.iterrows():
        color = color_map.get(
            row["cell_id"], "black"
        )  # Default to black if unknown cell_id
        plt.scatter(
            row["cell_lon"],
            row["cell_lat"],
            marker="x",
            color=color,
            s=200,
            label=f"Cell {row['cell_id']}",
        )

    # Plot UEs from df without labels but with the same color coding
    for _, row in df.iterrows():
        color = color_map.get(
            row["cell_id"], "black"
        )  # Default to black if unknown cell_id
        if row["sinr_db"] < -2.9:  # REMOVE COMMENT WHEN sinr_db IS FIXED
            color = "grey"  # Change to grey if sinr_db < 2

        plt.scatter(row["loc_x"], row["loc_y"], color=color)

    # Add labels and title
    plt.xlabel("Longitude (loc_x)")
    plt.ylabel("Latitude (loc_y)")
    plt.title("Cell Towers and UE Locations")

    # Create a legend for the cells only
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Show the plot
    plt.show()
