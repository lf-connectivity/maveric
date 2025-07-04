import os
import pickle
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.settings import cholesky_jitter
from gpytorch.utils.warnings import NumericalWarning

from notebooks.radp_library import (
    add_cell_info,
    calc_log_distance,
    calc_relative_bearing,
    check_cartesian_format,
    get_percell_data,
    normalize_cell_ids,
    preprocess_ue_data,
)
from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin, NormMethod
from radp.digital_twin.utils import constants
from radp.digital_twin.utils.cell_selection import perform_attachment

# Suppress the specific NumericalWarning from gpytorch
warnings.filterwarnings("ignore", category=NumericalWarning)


class MobilityRobustnessOptimization(ABC):
    """
    A class that contains a prototypical proof-of-concept of an `Mobility Robustness Optimization (MRO)` RIC xApp.
    """

    def __init__(
        self,
        mobility_model_params: Dict[str, Dict],
        topology: pd.DataFrame,
        bdt: Optional[Dict[str, BayesianDigitalTwin]] = None,
    ):
        self.topology = topology
        self.bayesian_digital_twins = bdt if bdt is not None else {}
        self.mobility_model_params = mobility_model_params
        self.simulation_data = None

    def train_or_update_rf_twins(self, new_data: pd.DataFrame) -> None:
        """
        Updates the Bayesian Digital Twins with new observations if they exist.
        If not, it trains new twins from scratch.

        Parameters:
            new_data (pd.DataFrame): UE data with rx power data in cartesian (UEs x Cells) format.
            The DataFrame should contain ['longitude', 'latitude', 'cell_id', 'cell_rxpwr_dbm'] columns.

            +-----------+-----------+-----------+-----------------+
            | longitude | latitude  | cell_id   | cell_rxpwr_dbm  |
            +===========+===========+===========+=================+
            | 1.0       | 2.0       | cell_1    |  ...            |
            | 1.0       | 2.0       | cell_2    |  ...            |
            | 1.0       | 2.0       | cell_3    |  ...            |
            | 3.0       | 4.0       | cell_1    |  ...            |
            | 3.0       | 4.0       | cell_2    |  ...            |
            | 3.0       | 4.0       | cell_3    |  ...            |
            +-----------+-----------+-----------+-----------------+

        """
        try:
            if not isinstance(new_data, pd.DataFrame):
                raise TypeError("The input 'new_data' must be a pandas DataFrame.")

            expected_columns = {"longitude", "latitude", "cell_id", "cell_rxpwr_dbm"}
            if not expected_columns.issubset(new_data.columns):
                raise ValueError(f"The input DataFrame must contain the following columns: {expected_columns}")

            # normalize cell_id format - regardless of dtype
            self.topology = normalize_cell_ids(self.topology)
            new_data = normalize_cell_ids(new_data)

            # Check if the new data is in the expected cartesian format
            check_cartesian_format(new_data, self.topology)

            # Prepare the new data for training or updating
            prepared_data = self._prepare_train_or_update_data(new_data)

            # update if bayesian digital twins exist already
            if self.bayesian_digital_twins:
                print("Updating existing Bayesian Digital Twins with new data.")

                for cell_id, df in prepared_data.items():
                    self._update(cell_id, df)
                print("Bayesian Digital Twins updated successfully.")

            # If no Bayesian Digital Twins exist, train from scratch
            else:
                print("No Bayesian Digital Twins available for update. Training from scratch.")
                self._training(maxiter=100, train_data=prepared_data)
                print("\nBayesian Digital Twins trained successfully.")

        except TypeError as te:
            print(f"TypeError: {te}")
        except ValueError as ve:
            print(f"ValueError: {ve}")
        except KeyError as ke:
            print(f"KeyError: {ke}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def save_bdt(self, file_relative_path="data/mro_data") -> bool:
        """
        Saves the Bayesian Digital Twins to a pickle file. Returns `True` if saving succeeds.
        """
        cwd = Path().absolute()
        filename = cwd / Path(file_relative_path) / Path("digital_twins.pkl")

        try:
            if not isinstance(self.bayesian_digital_twins, dict):
                raise TypeError("The attribute 'bayesian_digital_twins' must be a dictionary.")

            # Ensure the directory exists
            os.makedirs(file_relative_path, exist_ok=True)

            with open(filename, "wb") as fp:
                pickle.dump(self.bayesian_digital_twins, fp)

            print(f"Twins Saved Successfully as Pickle at: {filename}")

            return True  # Indicate successful save

        except TypeError as te:
            print(f"TypeError: {te}")
            return False

        except OSError as oe:
            print(f"OSError: {oe}")
            return False

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return False

    def load_bdt(self, file_relative_path="data/mro_data/digital_twins.pkl") -> bool:
        """
        Loads the Bayesian Digital Twins from a pickle file. Returns `True` if loading succeeds.
        """
        cwd = Path().absolute()
        filename = cwd / Path(file_relative_path)

        try:
            with open(filename, "rb") as fp:
                self.bayesian_digital_twins = pickle.load(fp)
            print(f"Twins Loaded Successfully from Pickle at: {filename}")

            return True  # Indicate successful load

        except FileNotFoundError as fnf:
            print(f"FileNotFoundError: {fnf}")
            return False

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return False

    @abstractmethod
    def solve(self) -> None:
        """
        Solve the mobility robustness optimization problem.

        This method is an abstract method that must be implemented by its subclasses.
        """
        pass

    def _training(self, maxiter: int, train_data: Dict[str, pd.DataFrame]) -> List[float]:
        """
        Trains the Bayesian Digital Twins for each cell in the topology using the UE locations and features
        like log distance, relative bearing, and cell received power (Rx power).
        
        +---------+----------+-----------+----------------+--------------+-------------------+
        | cell_id | latitude | longitude | cell_rxpwr_dbm | log_distance | relative_bearing  |
        +=========+==========+===========+================+==============+===================+
        |    1    | 90.412   | 23.810    |      -85       |   -2.546     |       25.0        |
        |    1    | 90.413   | 23.811    |      -90       |   -2.850     |       45.0        |
        |    2    | 90.415   | 23.812    |      -80       |   -2.268     |       60.0        |
        |    2    | 90.416   | 23.813    |      -88       |   -2.547     |       90.0        |
        |    3    | 90.417   | 23.814    |      -78       |   -2.120     |       30.0        |
        |    3    | 90.418   | 23.815    |      -92       |   -2.760     |       75.0        |
        +---------+----------+-----------+----------------+--------------+-------------------+

        """
        bayesian_digital_twins = {}
        loss_vs_iters = []

        for train_cell_id, training_data_idx in train_data.items():
            bayesian_digital_twins[train_cell_id] = BayesianDigitalTwin(
                data_in=[training_data_idx],
                x_columns=["log_distance", "relative_bearing"],
                y_columns=["cell_rxpwr_dbm"],
                norm_method=NormMethod.MINMAX,
            )

            self.bayesian_digital_twins[train_cell_id] = bayesian_digital_twins[train_cell_id]

            loss_vs_iters.append(
                bayesian_digital_twins[train_cell_id].train_distributed_gpmodel(
                    maxiter=maxiter,
                )
            )

        return loss_vs_iters

    def _update(self, cell_id: str, df: pd.DataFrame) -> None:
        """
        Updates the Bayesian Digital Twin (BDT) model for a specific cell.

        Updates by deduplicating samples using 'log_distance' and 'relative_bearing', subsampling up to 500
        strongest signals, reconfiguring the Gaussian Process with a Scale and RBF kernel, increasing observation
        noise via GaussianLikelihood, and using higher jitter to stabilize Cholesky decomposition before training
        on the processed data.
        
        +---------+----------+-----------+----------------+--------------+-------------------+
        | cell_id | latitude | longitude | cell_rxpwr_dbm | log_distance | relative_bearing  |
        +=========+==========+===========+================+==============+===================+
        |    1    | 90.412   | 23.810    |      -85       |   -2.546     |       25.0        |
        |    1    | 90.413   | 23.811    |      -90       |   -2.850     |       45.0        |
        |    2    | 90.415   | 23.812    |      -80       |   -2.268     |       60.0        |
        |    2    | 90.416   | 23.813    |      -88       |   -2.547     |       90.0        |
        |    3    | 90.417   | 23.814    |      -78       |   -2.120     |       30.0        |
        |    3    | 90.418   | 23.815    |      -92       |   -2.760     |       75.0        |
        +---------+----------+-----------+----------------+--------------+-------------------+

        """
        # Remove near-duplicates in feature space
        df = df.drop_duplicates(subset=["log_distance", "relative_bearing"])

        # Subsample to at most 500 strongest samples per cell
        if df.shape[0] > 500:
            df = get_percell_data(data_in=df, choose_strongest_samples_percell=True, n_samples=500,)[
                0
            ][0]

        twin = self.bayesian_digital_twins[cell_id]

        # Reconfigure the kernel to include scale + RBF
        twin.model.covar_module = ScaleKernel(RBFKernel())

        # Increase observation noise via GaussianLikelihood
        if not hasattr(twin, "likelihood"):
            twin.likelihood = GaussianLikelihood()  # type: ignore
        twin.likelihood.noise = 1e-2  # type: ignore

        # Use an increased jitter context
        with cholesky_jitter(1e-1):
            twin.update_trained_gpmodel([df])

    def _prepare_train_or_update_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Returnd key value pairs of cell_id and processed DataFrame for each cell_id.
            
        +--------+----------+-----------+------+---------+----------+----------+--------------+------------------------+
        | ue_id  | latitude | longitude | tick | cell_id | cell_lon | cell_lat | cell_az_deg  | cell_carrier_freq_mhz  |
        +========+==========+===========+======+=========+==========+==========+==============+========================+
        |   0    | 90.412   | 23.810    |  0   |    1    | 90.410   | 23.809   |     120       |        1800           |
        |   1    | 90.413   | 23.811    |  0   |    1    | 90.414   | 23.810   |     120       |        1800           |
        |   0    | 90.415   | 23.812    |  1   |    2    | 90.410   | 23.809   |     240       |        2100           |
        |   1    | 90.416   | 23.813    |  1   |    2    | 90.414   | 23.810   |     240       |        2100           |
        +--------+----------+-----------+------+---------+----------+----------+--------------+------------------------+

        
        """
        required_columns = {"cell_lat", "cell_lon", "cell_az_deg"}
        if not required_columns.issubset(df.columns):
            df = add_cell_info(df, self.topology)

        self.update_data = calc_log_distance(df)
        self.update_data = calc_relative_bearing(self.update_data)

        self.update_data = self.update_data.loc[:, ["cell_id", "log_distance", "relative_bearing", "cell_rxpwr_dbm"]]

        # anything refering as training indicates training or update data
        train_per_cell_df = [x for _, x in self.update_data.groupby("cell_id")]
        n_cell = len(self.topology.index)

        metadata_df = pd.DataFrame(
            {
                "cell_id": [cell_id for cell_id in self.topology.cell_id],
                "idx": [i + 1 for i in range(n_cell)],
            }
        )

        idx_cell_id_mapping = dict(zip(metadata_df.idx, metadata_df.cell_id))
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

        return training_data

    def _predictions(self, pred_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Predicts the received power for each User Equipment (UE) at different locations
        and ticks using Bayesian Digital Twins.
          
        +---------+-----------+------------+----------+
        |  ue_id  | latitude  | longitude  |   tick   |
        +=========+===========+============+==========+
        |    1    | 90.412    | 23.810     |     0    |
        |    2    | 90.413    | 23.811     |     0    |
        |    1    | 90.415    | 23.812     |     1    |
        |    2    | 90.416    | 23.813     |     1    |
        +---------+-----------+------------+----------+
        
        It then determines the best cell for each UE to attach based on the predicted power values.
        """
        # self.prediction_data = pred_data
        prediction_data = preprocess_ue_data(pred_data, self.topology)
        prediction_data = calc_relative_bearing(prediction_data)
        full_prediction_df = pd.DataFrame()

        # Loop over each 'tick'
        for tick, tick_df in prediction_data.groupby("tick"):
            # Loop over each 'cell_id' within the current 'tick'
            for cell_id, cell_df in tick_df.groupby("cell_id"):
                cell_id = f"cell_{cell_id}"  # FIXME: should look better
                # Check if the Bayesian model for this cell_id exists
                if cell_id in self.bayesian_digital_twins:
                    # Perform the Bayesian prediction
                    pred_means_percell, _ = self.bayesian_digital_twins[cell_id].predict_distributed_gpmodel(
                        prediction_dfs=[cell_df]
                    )

                    # Assuming 'pred_means_percell' returns a list of predictions corresponding to the DataFrame index
                    cell_df["pred_means"] = pred_means_percell[0]

                    # Include additional necessary columns for the final DataFrame
                    cell_df["tick"] = tick
                    cell_df["cell_id"] = cell_id

                    # Append the predictions to the full DataFrame
                    full_prediction_df = pd.concat([full_prediction_df, cell_df], ignore_index=True)

                else:
                    # Handle missing models, e.g., log a warning or initialize a default model
                    print(f"No model available for cell_id {cell_id}, skipping prediction.")

        full_prediction_df = full_prediction_df.rename(columns={"latitude": "loc_y", "longitude": "loc_x"})
        if full_prediction_df["cell_id"].dtype == object:
            full_prediction_df["cell_id"] = full_prediction_df["cell_id"].str.extract(r"(\d+)").astype(int)
        predicted = perform_attachment(full_prediction_df, self.topology)
        if full_prediction_df["cell_id"].dtype == int:
            full_prediction_df["cell_id"] = full_prediction_df["cell_id"].apply(lambda x: f"cell_{x}")
        return predicted, full_prediction_df

    def _preprocess_simulation_data(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        +------------+-------------+-------------+-------------+------------+------------+-------------------------+
        | mock_ue_id | cell_id     | rxpower_dbm |  rxpower_stddev_dbm  |  log_distance | pred_means |     tick    |
        +============+=============+=============+=====================+==============+=============+==============+
        |     0      | "cell_1"    |   -85.0     |        1.2           |     0.305     |   -86.3     |     0      |
        |     1      | "cell_2"    |   -88.5     |        1.5           |     0.422     |   -87.1     |     0      |
        |     0      | "cell_2"    |   -82.1     |        1.1           |     0.207     |   -84.2     |     1      |
        |     1      | "cell_3"    |   -90.4     |        1.3           |     0.499     |   -89.0     |     1      |
        +------------+-------------+-------------+----------------------+--------------+-------------+-------------+

        '''
        df.drop(
            columns=["rxpower_stddev_dbm", "rxpower_dbm", "cell_rxpwr_dbm"],
            inplace=True,
        )
        df.rename(
            columns={
                "mock_ue_id": "ue_id",
                "log_distance": "distance_km",
                "pred_means": "cell_rxpower_dbm",
            },
            inplace=True,
        )
        if self.topology["cell_id"].dtype == object:
            self.topology["cell_id"] = self.topology["cell_id"].str.replace("cell_", "").astype(int)
        if df["cell_id"].dtype == object:
            df["cell_id"] = df["cell_id"].str.extract(r"(\d+)").astype(int)
        df = self._add_sinr_column(df)
        return df
    # TODO: Use Utils version of this function
    def _add_sinr_column(self, df: pd.DataFrame) -> pd.DataFrame:
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
                noise_db = constants.LATENT_BACKGROUND_NOISE_DB

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


# Functions for MRO metrics and Handover events
def _count_handovers(df: pd.DataFrame) -> int:
    """
    Count the number of seemless cell handovers (cell to cell switches) for user equipment (UE) based
    on cell_id changes between consecutive ticks, excluding switches to 'RLF'.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'ue_id', 'cell_id', and 'tick' columns.

    +--------+---------+------+
    | ue_id  | cell_id | tick |
    +========+=========+======+
    |   0    |    4    |  1   |
    |   1    |    2    |  2   |
    |   2    |    5    |  3   |
    |   3    |    3    |  4   |
    +--------+---------+------+

    Returns:
        int: Total number of valid cell switches across all UEs.
    """
    count = 0
    df = df.sort_values(by=["ue_id", "tick"])  # Ensure correct order
    prev_cells = {}
    prev_ticks = {}

    for _, row in df.iterrows():
        ue_id, cell_id, tick = row["ue_id"], row["cell_id"], row["tick"]

        if ue_id in prev_cells and prev_cells[ue_id] != cell_id and prev_cells[ue_id] is not None:
            if tick == prev_ticks[ue_id] + 1 and cell_id != "RLF":
                count += 1

        prev_cells[ue_id] = cell_id
        prev_ticks[ue_id] = tick
    return count


def reattach_columns(predicted_df, full_prediction_df):
    # Filter full_prediction_df for the needed columns and drop duplicates based on loc_x and loc_y
    filtered_full_df = full_prediction_df[["mock_ue_id", "tick", "loc_x", "loc_y"]].drop_duplicates(
        subset=["loc_x", "loc_y"]
    )

    # Merge with predicted_df based on loc_x and loc_y, ensuring size matches predicted_df
    merged_df = pd.merge(predicted_df, filtered_full_df, on=["loc_x", "loc_y"], how="left")

    # Rename mock_ue_id to ue_id
    merged_df.rename(columns={"mock_ue_id": "ue_id"}, inplace=True)

    return merged_df


def calculate_mro_metric(data: pd.DataFrame) -> float:
    """
    Calculated total operational cellular time remaining after loss due to cell handovers (including RLF)

    Parameters:
        data (pd.DataFrame): DataFrame containing UE data with a 'tick' column.

    +--------+---------+------+
    | ue_id  | cell_id | tick |
    +========+=========+======+
    |   0    |    4    |  1   |
    |   1    |    2    |  2   |
    |   2    |    5    |  3   |
    |   3    |    3    |  4   |
    +--------+---------+------+

    Returns:
        float: Effective operational score symbolizing time effectively after subtracting handover and RLF delays.
    """
    # Constants for interruption times
    ts = 50 / 1000  # Convert ms to seconds
    t_nas = 1000 / 1000  # Convert ms to seconds

    # Calculate total time (T) based on ticks; assuming each tick represents a uniform time slice
    # This could be adjusted if ticks represent variable time slices
    # Rather than passing the UE Data as whole we can send just an integar for tick
    ticks = len(data["tick"].unique())
    # Assuming each tick represents 50ms (this value may need to be adjusted based on actual data characteristics)
    tick_duration_seconds = 1  # 1 second per tick
    T = ticks * tick_duration_seconds

    ns_handover_count = _count_handovers(data)  # Count of handovers to different cells (excluding RLF)
    nf_handover_count = _count_rlf(data)  # Count of handovers to RLF

    # Calculate D
    D = T - (ns_handover_count * ts + nf_handover_count * t_nas)

    return D


def _count_rlf(df: pd.DataFrame) -> int:
    """
    Counts the number of Radio Link Failures (RLF) by analyzing cell handovers onto RLF for UE

    Parameters:
        df (pd.DataFrame): DataFrame containing 'ue_id', 'cell_id', and 'tick' columns.

    +--------+---------+------+
    | ue_id  | cell_id | tick |
    +========+=========+======+
    |   0    |    4    |  1   |
    |   1    |    2    |  2   |
    |   2    |    5    |  3   |
    |   3    |    3    |  4   |
    +--------+---------+------+

    Returns:
        int: Total number of UE transitions to RLF cells.
    """
    return (df["cell_id"] == "RLF").sum()
