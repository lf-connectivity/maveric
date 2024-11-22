import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from typing import Any, Dict, List, Tuple, Optional

from radp.digital_twin.utils import constants
from radp.digital_twin.utils.gis_tools import GISTools
from radp.digital_twin.rf.bayesian.bayesian_engine import (
    BayesianDigitalTwin,
    NormMethod,
)
from notebooks.radp_library import get_percell_data
from radp.digital_twin.utils.cell_selection import perform_attachment
from notebooks.radp_library import get_ue_data


class MobilityRobustnessOptimization:
    """
    A class to perform Mobility Robustness Optimization (MRO) using Bayesian Digital Twins. This class integrates
    user equipment (UE) data with cell topology to predict the received power at various UE locations and
    determines the optimal cell attachment based on these predictions. The class uses Bayesian modeling to
    accurately forecast signal strength, accounting for factors such as distance, frequency, and antenna characteristics.
    """

    def __init__(
        self,
        mobility_params: Dict[str, Dict],
        topology: pd.DataFrame,
        bdt: Optional[Dict[str, BayesianDigitalTwin]] = None,
    ):
        self.topology = topology
        self.tx_power_dbm = 23
        self.bayesian_digital_twins = bdt if bdt is not None else {}
        self.mobility_params = mobility_params
        self.training_data = None
        self.prediction_data = None
        self.update_data = None
        self.simulation_data = None

    def update(self, new_data: pd.DataFrame):
        """
        Updates or trains Bayesian Digital Twins for each cell with new data.
        """
        try:
            if not isinstance(new_data, pd.DataFrame):
                raise TypeError("The input 'new_data' must be a pandas DataFrame.")

            expected_columns = {"mock_ue_id", "longitude", "latitude", "tick"}
            if not expected_columns.issubset(new_data.columns):
                raise ValueError(
                    f"The input DataFrame must contain the following columns: {expected_columns}"
                )

            if self.bayesian_digital_twins:
                self.update_data = new_data
                updated_data = self._preprocess_ue_update_data()
                updated_data_list = list(updated_data.values())

                for data_idx, update_data_df in enumerate(updated_data_list):
                    update_cell_id = data_idx + 1
                    if update_cell_id in self.bayesian_digital_twins:
                        self.bayesian_digital_twins[
                            update_cell_id
                        ].update_trained_gpmodel([update_data_df])
            else:
                print(
                    "No Bayesian Digital Twins available for update. Training from scratch."
                )
                self.training(maxiter=100, train_data=new_data)
        except TypeError as te:
            print(f"TypeError: {te}")
        except ValueError as ve:
            print(f"ValueError: {ve}")
        except KeyError as ke:
            print(f"KeyError: {ke}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def save(bayesian_digital_twins, file_loc):
        """
        Saves the Bayesian Digital Twins to a pickle file.
        """
        filename = f"{file_loc}/digital_twins.pkl"
        try:
            if not isinstance(bayesian_digital_twins, dict):
                raise TypeError(
                    "The input 'bayesian_digital_twins' must be a dictionary."
                )

            with open(filename, "wb") as fp:
                pickle.dump(bayesian_digital_twins, fp)

            print("Twins Saved Successfully as Pickle.")
        except TypeError as te:
            print(f"TypeError: {te}")
        except OSError as oe:
            print(f"OSError: {oe}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def solve(self):
        """
         Processes the generation and simulation of new user equipment (UE) data, and performs
         multiple operations to optimize network efficiency and ensure robust mobile connectivity.

        This method carries out the following operations sequentially:
            1. Generates and simulates new UE data to model real-world mobile connectivity scenarios.
            2. Applies the 'perform_attachment' method to determine the optimal cell for each UE based on the simulated data.
            3. Computes the Mobility Robustness Optimization (MRO) metrics to evaluate the effectiveness of the current network configuration.
            4. Applies the 'perform_attachment' method again, this time incorporating hysteresis and time-to-trigger adjustments,
               to refine the decision process for UE cell attachment.
        """
        self.simulation_data = get_ue_data(self.mobility_params)
        ue_data = self._preprocess_ue_simulation_data()

        ue_data, topology = self._format_ue_data_and_topology(ue_data,topology)
    
        history = perform_attachment(ue_data,topology)
        
        simulation_data = self.simulation_data.copy()
        simulation_data = simulation_data.rename(columns={"lat": "loc_y", "lon": "loc_x"})

        # Reattach the columns of the data found from the history of the simulation by using performing attachment data and the actual simulated data
        reattached_data = reattach_columns(history, simulation_data)
        # Count the number of handovers
        ns_handovers, nf_handovers, no_change = count_handovers(reattached_data)

        # Calculate the MRO Metric
        mro_metric = calculate_mro_metric(
            ns_handovers, nf_handovers, self.simulation_data
        )

        return mro_metric
    
    def _format_ue_data_and_topology(self, ue_data, topology):
        ue_data = ue_data.rename(columns={"lat": "loc_y", "lon": "loc_x"})
        topology = self.topology
        topology["cell_id"] = topology["cell_id"].str.extract("(\d+)").astype(int)
        topology = topology.rename(columns={"cell_lat" :"loc_y", "cell_lon": "loc_x"})
        return ue_data, topology

    def _training(self, maxiter: int, train_data: pd.DataFrame) -> List[float]:
        """
        Trains the Bayesian Digital Twins for each cell in the topology using the UE locations and features
        like log distance, relative bearing, and cell received power (Rx power).
        """
        self.training_data = train_data
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

    def _predictions(self, pred_data) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Predicts the received power for each User Equipment (UE) at different locations and ticks using Bayesian Digital Twins.
        It then determines the best cell for each UE to attach based on the predicted power values.
        """
        self.prediction_data = pred_data
        prediction_data = self._preprocess_prediction_data()
        full_prediction_df = pd.DataFrame()

        # Loop over each 'tick'
        for tick, tick_df in prediction_data.groupby("tick"):
            # Loop over each 'cell_id' within the current 'tick'
            for cell_id, cell_df in tick_df.groupby("cell_id"):
                cell_id = f"cell_{cell_id}"  # FIXME: should look better
                # Check if the Bayesian model for this cell_id exists
                if cell_id in self.bayesian_digital_twins:
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
                else:
                    # Handle missing models, e.g., log a warning or initialize a default model
                    print(
                        f"No model available for cell_id {cell_id}, skipping prediction."
                    )

        full_prediction_df = full_prediction_df.rename(
            columns={"latitude": "loc_y", "longitude": "loc_x"}
        )
        predicted = perform_attachment(full_prediction_df, self.topology)

        return predicted, full_prediction_df

    def _connect_ue_to_all_cells(
        self, prediction: bool = False, simulation: bool = False, update: bool = False
    ) -> pd.DataFrame:
        """
        Connects each user equipment (UE) entry to all cells in the topology for each tick,
        effectively creating a Cartesian product of UEs and cells, which includes data from both sources.
        """

        if prediction:
            ue_data_tmp = self.prediction_data.copy()
        elif simulation:
            ue_data_tmp = self.simulation_data.copy()
        elif update:
            ue_data_tmp = self.update_data.copy()
        else:
            ue_data_tmp = self.training_data.copy()
        topology_tmp = self.topology.copy()
        # Remove the 'cell_' prefix and convert cell_id to integer if needed
        if topology_tmp["cell_id"].dtype == object:
            topology_tmp["cell_id"] = (
                topology_tmp["cell_id"].str.replace("cell_", "").astype(int)
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
    
    def _preprocess_ue_simulation_data(self) -> pd.DataFrame:
        data = self._connect_ue_to_all_cells(simulation=True)
        data["log_distance"] = data.apply(
            lambda row: GISTools.get_log_distance(
                row["lat"], row["lon"], row["cell_lat"], row["cell_lon"]
            ),
            axis=1,
        )

        data["rxpower_dbm"] = data.apply(
            lambda row: self._calculate_received_power(
                row["log_distance"], row["cell_carrier_freq_mhz"]
            ),
            axis=1,
        )

        return data

    def _preprocess_ue_training_data(self) -> pd.DataFrame:
        data = self._preprocess_ue_topology_data()
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

    def _preprocess_ue_update_data(self) -> pd.DataFrame:
        data = self._connect_ue_to_all_cells(update=True)
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

        update_per_cell_df = [x for _, x in data.groupby("cell_id")]
        n_cell = len(self.topology.index)

        metadata_df = pd.DataFrame(
            {
                "cell_id": [cell_id for cell_id in self.topology.cell_id],
                "idx": [i + 1 for i in range(n_cell)],
            }
        )
        idx_cell_id_mapping = dict(zip(metadata_df.idx, metadata_df.cell_id))

        n_samples_update = []
        for df in update_per_cell_df:
            n_samples_update.append(df.shape[0])

        update_per_cell_df_processed = []
        for i in range(n_cell):
            update_per_cell_df_processed.append(
                get_percell_data(
                    data_in=update_per_cell_df[i],
                    choose_strongest_samples_percell=False,
                    n_samples=n_samples_update[i],
                )[0][0]
            )

        update_data = {}

        for i, df in enumerate(update_per_cell_df_processed):
            update_cell_id = idx_cell_id_mapping[i + 1]
            update_data[update_cell_id] = df

        for update_cell_id, update_data_idx in update_data.items():
            update_data_idx["cell_id"] = update_cell_id
            update_data_idx["cell_lat"] = self.topology[
                self.topology["cell_id"] == update_cell_id
            ]["cell_lat"].values[0]
            update_data_idx["cell_lon"] = self.topology[
                self.topology["cell_id"] == update_cell_id
            ]["cell_lon"].values[0]
            update_data_idx["cell_az_deg"] = self.topology[
                self.topology["cell_id"] == update_cell_id
            ]["cell_az_deg"].values[0]
            update_data_idx["cell_carrier_freq_mhz"] = self.topology[
                self.topology["cell_id"] == update_cell_id
            ]["cell_carrier_freq_mhz"].values[0]
            update_data_idx["relative_bearing"] = [
                GISTools.get_relative_bearing(
                    update_data_idx["cell_az_deg"].values[0],
                    update_data_idx["cell_lat"].values[0],
                    update_data_idx["cell_lon"].values[0],
                    lat,
                    lon,
                )
                for lat, lon in zip(
                    update_data_idx["latitude"], update_data_idx["longitude"]
                )
            ]
        return update_data

    def _preprocess_prediction_data(self) -> pd.DataFrame:
        data = self._connect_ue_to_all_cells(prediction=True)

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


# Functions for MRO metrics and Handover events
def count_handovers(df):
    # Initialize counters for NS (Successful Handovers), NF (Radio Link Failures), and no change
    ns_handover_count = 0
    nf_handover_count = 0
    no_change = 0

    # Threshold for considering a Radio Link Failure (RLF)
    rlf_threshold = -2.9

    # Track the previous state for each UE
    ue_previous_state = {}

    # Loop through the dataframe row by row
    for _, row in df.iterrows():
        ue_id = row["ue_id"]
        current_cell_id = row["cell_id"]
        current_sinr_db = row["sinr_db"]

        # Check if we have previous state for this UE
        if ue_id in ue_previous_state:
            previous_cell_id, previous_sinr_db = ue_previous_state[ue_id]

            # Check for cell ID change
            if previous_cell_id != current_cell_id:
                # Check if the SINR is above the threshold after a cell change
                if current_sinr_db >= rlf_threshold:
                    ns_handover_count += 1  # Successful handover
                else:
                    nf_handover_count += 1  # Failed handover due to RLF after change
            elif previous_sinr_db < rlf_threshold and current_sinr_db >= rlf_threshold:
                ns_handover_count += (
                    1  # Successful recovery from RLF without cell change
                )
            elif current_sinr_db < rlf_threshold:
                nf_handover_count += 1  # Ongoing or new RLF
            else:
                no_change += 1  # No significant event

        else:
            # If first occurrence of UE has SINR below the RLF threshold, consider it as RLF
            if current_sinr_db < rlf_threshold:
                nf_handover_count += 1
            else:
                no_change += 1  # No significant event when UE first appears and SINR is above threshold

        # Update the state for this UE
        ue_previous_state[ue_id] = (current_cell_id, current_sinr_db)

    return ns_handover_count, nf_handover_count, no_change


def reattach_columns(predicted_df, full_prediction_df):
    # Filter full_prediction_df for the needed columns and drop duplicates based on loc_x and loc_y
    filtered_full_df = full_prediction_df[
        ["mock_ue_id", "tick", "loc_x", "loc_y"]
    ].drop_duplicates(subset=["loc_x", "loc_y"])

    # Merge with predicted_df based on loc_x and loc_y, ensuring size matches predicted_df
    merged_df = pd.merge(
        predicted_df, filtered_full_df, on=["loc_x", "loc_y"], how="left"
    )

    # Rename mock_ue_id to ue_id
    merged_df.rename(columns={"mock_ue_id": "ue_id"}, inplace=True)

    return merged_df


def calculate_mro_metric(ns_handover_count, nf_handover_count, prediction_ue_data):
    # Constants for interruption times
    ts = 50 / 1000  # Convert ms to seconds
    t_nas = 1000 / 1000  # Convert ms to seconds

    # Calculate total time (T) based on ticks; assuming each tick represents a uniform time slice
    # This could be adjusted if ticks represent variable time slices
    # Rather than passing the UE Data as whole we can send just an integar for tick
    ticks = len(prediction_ue_data["tick"].unique())
    # Assuming each tick represents 50ms (this value may need to be adjusted based on actual data characteristics)
    tick_duration_seconds = 1  # 1 second per tick
    T = ticks * tick_duration_seconds

    # Calculate D
    D = T - (ns_handover_count * ts + nf_handover_count * t_nas)

    return D
