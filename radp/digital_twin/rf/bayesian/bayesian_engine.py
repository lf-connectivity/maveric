#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#  TODO: refactor this module to add/remove documentation, helper methods
#  etc to improve its readability and remove unnecessary complexity

from __future__ import annotations

import logging
import os
import pickle
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import gpytorch
import numpy as np
import pandas as pd
import torch

from radp.common.helpers.file_system_safety import atomic_write
from radp.digital_twin.utils import constants
from radp.digital_twin.utils.gis_tools import GISTools

logger = logging.getLogger(__name__)


class NormMethod(Enum):
    MINMAX = "minmax"  # {value - min}/{max - min}
    ZSCORE = "zscore"  # {value - mean}/{std}


class ExactGPModel(gpytorch.models.ExactGP):
    # We will use the simplest form of GP model, exact inference
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([1]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([1])),
            batch_shape=torch.Size([1]),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BayesianDigitalTwin:
    def __init__(
        self,
        data_in: List[pd.DataFrame],
        x_columns: List[str],
        y_columns: List[str],
        norm_method: NormMethod = NormMethod.MINMAX,
        x_max: Optional[Dict[str, float]] = None,
        x_min: Optional[Dict[str, float]] = None,
    ):
        """
        `data_in` is a list of Pandas dataframes, where each one corresponds to
        training data for one cell. `stats` is a list of Pandas dataframes, of the
        same length as `data_in`, where each contains statistics of the corresponding
        cell, to be used for pre-training normalization and post-prediction de-normalization.

        `x_columns` specifies the columns to train on, and `y_columns` specifies the columns
        to predict. These must be present in `data_in`.

        `x_max` and `x_min` contains optional user-specified max and min values for columns
        in `data_in` -- if these are provided, they are used instead of observed ranges
        during input pre-training normalization.

        `data_in` and `stats` may be constructed using `get_percell_data`.
        """
        self.is_cuda = False
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            self.is_cuda = True

        self.cell_ids = [data_in_cell.cell_id.unique()[0] for data_in_cell in data_in]
        self.num_cells = len(self.cell_ids)
        self.x_columns = x_columns
        self.y_columns = y_columns
        self.num_features = len(self.x_columns)

        self.cell_stats = [cell_data.describe() for cell_data in data_in]
        self.xmeans = []
        self.xstds = []
        self.xmax = []
        self.xmin = []
        self.ymeans = []
        self.ystds = []

        self.norm_method = norm_method

        # Create Gaussian Process Regression (GPR) model; independent outputs.
        # produce normalization ranges
        for m in range(self.num_cells):
            self.xmax.append(self.cell_stats[m].loc["max", self.x_columns])
            self.xmin.append(self.cell_stats[m].loc["min", self.x_columns])
            # if explicitly provided, replace and use instead of empirical ranges
            if x_max:
                for k, v in x_max.items():
                    if k in self.x_columns:
                        self.xmax[m][k] = v
            if x_min:
                for k, v in x_min.items():
                    if k in self.x_columns:
                        self.xmin[m][k] = v
            self.xmeans.append(self.cell_stats[m].loc["mean", self.x_columns])
            self.xstds.append(self.cell_stats[m].loc["std", self.x_columns])
            self.ymeans.append(self.cell_stats[m].loc["mean", self.y_columns])
            self.ystds.append(self.cell_stats[m].loc["std", self.y_columns])

        # create training tensors
        train_X, train_Y = self._create_training_tensors(data_in)

        # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([self.num_cells]))

        self.model = ExactGPModel(train_X, train_Y, likelihood)

    def _create_training_tensors(
        self,
        data_in: List[pd.DataFrame],
    ):
        n_train = data_in[0].shape[0]

        # Get train_X and train_Y, create training tensors
        train_X = torch.zeros([self.num_cells, n_train, self.num_features], dtype=torch.float32)

        train_Y = torch.zeros([self.num_cells, n_train], dtype=torch.float32)

        for m in range(self.num_cells):
            if self.norm_method == NormMethod.MINMAX:
                train_x_cell = (data_in[m][self.x_columns] - self.xmin[m]) / (self.xmax[m] - self.xmin[m])
            elif self.norm_method == NormMethod.ZSCORE:
                train_x_cell = (data_in[m][self.x_columns] - self.xmeans[m]) / self.xstds[m]

            train_X_cell = torch.tensor(train_x_cell.iloc[:, :].values, dtype=torch.float32)

            train_y_cell = (data_in[m][self.y_columns] - self.ymeans[m]) / self.ystds[m]
            train_Y_cell = torch.tensor(train_y_cell.iloc[:, :].values, dtype=torch.float32)

            train_X[m] = train_X_cell.reshape(shape=(1, -1, self.num_features))
            train_Y[m] = torch.transpose(train_Y_cell, 0, 1)

        return train_X, train_Y

    @staticmethod
    def preprocess_ue_training_data(ue_training_data_df: pd.DataFrame, topology_df: pd.DataFrame) -> Dict:
        """Preprocess UE data before training

        ue_training_data_df -   dataframe containing location data as well as config
        topology_df -           dataframe containing the static cell topology
        """

        topology_df.reset_index(drop=True, inplace=True)

        # feature engineering -- add relative bearing and distance
        for i in ue_training_data_df.index:
            cell_topology = topology_df[topology_df.cell_id == ue_training_data_df.at[i, "cell_id"]]

            # change lon/lat to loc_x/loc_y in ue data
            ue_training_data_df.at[i, "loc_x"] = ue_training_data_df.at[i, "lon"]
            ue_training_data_df.at[i, "loc_y"] = ue_training_data_df.at[i, "lat"]

            # add the topology columns to training data
            ue_training_data_df.at[i, "cell_lat"] = cell_topology.cell_lat.values[0]
            ue_training_data_df.at[i, "cell_lon"] = cell_topology.cell_lon.values[0]
            ue_training_data_df.at[i, "cell_az_deg"] = cell_topology.cell_az_deg.values[0]
            ue_training_data_df.at[i, "cell_carrier_freq_mhz"] = cell_topology.cell_carrier_freq_mhz.values[0]

            # engineer and add the log distance and relative bearing features
            ue_training_data_df.at[i, "log_distance"] = np.log(
                1
                + 1000
                * GISTools.dist(
                    (
                        ue_training_data_df.at[i, "cell_lat"],
                        ue_training_data_df.at[i, "cell_lon"],
                    ),
                    (
                        ue_training_data_df.at[i, "lat"],
                        ue_training_data_df.at[i, "lon"],
                    ),
                )
            )
            ue_training_data_df.at[i, "relative_bearing"] = GISTools.rel_bearing(
                ue_training_data_df.at[i, "cell_az_deg"],
                GISTools.convert_bearing_0_to_360(
                    GISTools.get_bearing(
                        (
                            ue_training_data_df.at[i, "lat"],
                            ue_training_data_df.at[i, "lon"],
                        ),
                        (
                            ue_training_data_df.at[i, "cell_lat"],
                            ue_training_data_df.at[i, "cell_lon"],
                        ),
                    )
                ),
            )

        ue_training_data_df.cell_id = ue_training_data_df.cell_id
        return {k: v for k, v in ue_training_data_df.groupby("cell_id")}

    @staticmethod
    def preprocess_ue_prediction_data(
        ue_data_df: pd.DataFrame,
        config_df: pd.DataFrame,
        topology_df: pd.DataFrame,
    ) -> Dict:
        """Preprocess UE data before prediction

        ue_data_df -    dataframe containing the cell_id, location data and optionally
                        mock_ue_id/tick
        config_df -     dataframe containing cell config
        topology_df -   dataframe containing dataframe containing the static cell topology
        """

        config_df.reset_index(drop=True, inplace=True)

        # feature engineering -- add relative bearing and distance
        for i in ue_data_df.index:
            # pull the config and topology for this cell
            cell_config = config_df[config_df.cell_id == ue_data_df.at[i, "cell_id"]]
            cell_topology = topology_df[topology_df.cell_id == ue_data_df.at[i, "cell_id"]]

            # change lon/lat to loc_x/loc_y in ue data
            ue_data_df.at[i, "loc_x"] = ue_data_df.at[i, "lon"]
            ue_data_df.at[i, "loc_y"] = ue_data_df.at[i, "lat"]

            # add the topology columns to ue data
            ue_data_df.at[i, "cell_lat"] = cell_topology.cell_lat.values[0]
            ue_data_df.at[i, "cell_lon"] = cell_topology.cell_lon.values[0]
            ue_data_df.at[i, "cell_az_deg"] = cell_topology.cell_az_deg.values[0]
            ue_data_df.at[i, "cell_carrier_freq_mhz"] = cell_topology.cell_carrier_freq_mhz.values[0]

            # add the config columns to ue data
            ue_data_df.at[i, "cell_el_deg"] = cell_config.cell_el_deg.values[0]

            # engineer and add the log distance and relative bearing features
            ue_data_df.at[i, "log_distance"] = np.log(
                1
                + 1000
                * GISTools.dist(
                    (
                        ue_data_df.at[i, "cell_lat"],
                        ue_data_df.at[i, "cell_lon"],
                    ),
                    (ue_data_df.at[i, "lat"], ue_data_df.at[i, "lon"]),
                )
            )
            ue_data_df.at[i, "relative_bearing"] = GISTools.rel_bearing(
                ue_data_df.at[i, "cell_az_deg"],
                GISTools.convert_bearing_0_to_360(
                    GISTools.get_bearing(
                        (
                            ue_data_df.at[i, "lat"],
                            ue_data_df.at[i, "lon"],
                        ),
                        (
                            ue_data_df.at[i, "cell_lat"],
                            ue_data_df.at[i, "cell_lon"],
                        ),
                    )
                ),
            )

        ue_data_df.cell_id = ue_data_df.cell_id
        return {k: v for k, v in ue_data_df.groupby("cell_id")}

    @staticmethod
    def load_model_map_from_pickle(
        model_file_path: str,
    ) -> Dict[str, BayesianDigitalTwin]:
        # load trained digital twin model map
        try:
            with open(model_file_path, "rb") as pickle_file:
                model_map: Dict[str, BayesianDigitalTwin] = pickle.load(pickle_file)
                return model_map
        except Exception as e:
            logger.exception(f"Exception occurred while loading digital twin model from file: {model_file_path}")
            raise e

    @staticmethod
    def save_model_map_to_pickle(
        model_file_path: str,
        model_map: Dict[str, BayesianDigitalTwin],
    ) -> None:
        """Saving mapping of models to file"""
        try:
            with atomic_write(model_file_path, "wb") as pickle_file:
                pickle.dump(model_map, pickle_file)
            logger.info(f"Successfully saved model to file: {model_file_path}")
        except Exception as e:
            logger.exception(f"Exception occurred writing digital twin model to file: {model_file_path}")
            raise e

    @staticmethod
    def split_training_and_test_data(
        data_in: pd.DataFrame,
        n_sim: int,
        alpha: float,
    ) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], pd.DataFrame, Any]:
        """Split the simulation data groups into test and training sets.

        data_in: aggregated simulation data in the form of a dataframe (test+train)
        n_sim: number of simulations aggregated in data_in
        alpha: percent of simulation runs used for training

        Each row of `data_in` corresponds to one pixel, and the columns
        are assumed to contain :
            - settings corresponding to one or more cells, with column names
            `settng_name_<n>` for different settings of interest and where
            n refers to the index of a cell in the modeled cluster.
            - `rx_loc1` and `rx_loc2`, the geo-coordinates for the pixel
            - `rxpower_dbm_<n>`, the received powers for the cell with index n
            - `rsrp_dbm` is the max power and `cell_id` is the cell index of that cell
            - `sim_idx` is the simulation index

        Example:
        ['cell_azimuth_deg_1', 'cell_azimuth_deg_2', 'cell_azimuth_deg_3',
        'cell_elec_tilt_deg_1', 'cell_elec_tilt_deg_2', 'cell_elec_tilt_deg_3',
        'cell_mech_tilt_deg_1', 'cell_mech_tilt_deg_2', 'cell_mech_tilt_deg_3',
        'cell_txpower_dbm_1', 'cell_txpower_dbm_2', 'cell_txpower_dbm_3',
        'rxpower_dbm_1', 'rxpower_dbm_2', 'rxpower_dbm_3', 'rsrp_dbm',
        'sinr_db', 'cell_id', 'rx_loc1', 'rx_loc2', 'sim_idx']

        """
        n_training_group = np.max([int(alpha * 0.01 * n_sim), 1])
        n_test_group = n_sim - n_training_group
        logger.info(f"Splitting data into {n_training_group} training and {n_test_group} test groups...")
        training_data = data_in[data_in[constants.SIM_IDX] > n_test_group].reset_index(drop=True)
        test_data = data_in[data_in[constants.SIM_IDX] <= n_test_group].reset_index(drop=True)
        stats = data_in.describe(include="all")
        return training_data, test_data, stats, n_training_group

    @staticmethod
    def create_prediction_frames(
        site_config_df: pd.DataFrame,
        prediction_frame_template: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """
        `site_config_df` : 1 unique cell per row, contains at least the columns
            [cell_lat, cell_lon, cell_el_deg, cell_az_deg, cell_id]
            Assumption : `bayesian_digital_twin` was trained with respect tp `site_config_df`
        `prediction_frame_template` : 1 prediction point per row, contains columsn [loc_x, loc_y]
            e.g. loc_x is longitude, and loc_y is latitude
        """

        prediction_dfs: Dict[str, pd.DataFrame] = {}

        for c in site_config_df.itertuples():
            prediction_df = prediction_frame_template.copy()

            prediction_df[constants.CELL_LAT] = c.cell_lat
            prediction_df[constants.CELL_LON] = c.cell_lon
            prediction_df[constants.CELL_EL_DEG] = c.cell_el_deg
            prediction_df[constants.CELL_ID] = c.cell_id
            prediction_df[constants.CELL_CARRIER_FREQ_MHZ] = c.cell_carrier_freq_mhz
            prediction_df[constants.HTX] = c.hTx
            prediction_df[constants.HRX] = c.hRx

            prediction_df[constants.LOG_DISTANCE] = [
                GISTools.get_log_distance(
                    c.cell_lat,
                    c.cell_lon,
                    lat,
                    lon,
                )
                for lat, lon in zip(prediction_frame_template.loc_y, prediction_frame_template.loc_x)
            ]

            prediction_df[constants.RELATIVE_BEARING] = [
                GISTools.get_relative_bearing(
                    c.cell_az_deg,
                    c.cell_lat,
                    c.cell_lon,
                    lat,
                    lon,
                )
                for lat, lon in zip(prediction_frame_template.loc_y, prediction_frame_template.loc_x)
            ]

            prediction_df[constants.ANTENNA_GAIN] = GISTools.get_antenna_gain(
                c.hTx, c.hRx, prediction_df[constants.LOG_DISTANCE], c.cell_el_deg
            )

            prediction_dfs[c.cell_id] = prediction_df
        return prediction_dfs

    def train_distributed_gpmodel(
        self,
        maxiter: int = 100,
        lr: float = 0.05,
        stopping_threshold: float = 1e-4,
        load_model: bool = False,
        save_model: bool = False,
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        # This method actually calls train on the model.
        # Or can also be overloaded to alternatively pre-load an already trained model.
        # It can conditionally be told to save the trained model to filesystem,
        # then finally returns the loss array.
        """
        loss_vs_iter = np.zeros(maxiter)

        # Train model
        if load_model:
            # Check that model path and name are both provided
            if not model_path or not model_name:
                raise RuntimeError("Exception loading model: model_path and model_name must be provided")
            logger.info("Now loading GP model (this should be quick...)")
            state_dict = torch.load(model_path + model_name)
            self.model.load_state_dict(state_dict)
            if self.is_cuda:
                logger.info("Cuda enabled for model.")
                self.model = self.model.cuda()
        else:
            self.model.train()
            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

            train_X, train_Y = (
                mll.model.train_inputs,
                mll.model.train_targets,
            )

            if self.is_cuda:
                logger.info("Cuda enabled for training data.")
                train_X = [train_X[0].cuda()]
                train_Y = train_Y.cuda()
                self.model = self.model.cuda()
                mll = mll.cuda()

            last_loss = float("-inf")
            for i in range(maxiter):
                optimizer.zero_grad()
                output = self.model(*train_X)
                loss = -mll(output, train_Y).sum()
                loss.backward()
                optimizer.step()
                this_loss = loss.item()
                loss_vs_iter[i] = this_loss
                delta = this_loss - last_loss
                last_loss = this_loss
                logger.info("Iter %d/%d - Loss: %.3f (delta=%.6f)" % (i + 1, maxiter, this_loss, delta))
                if abs(delta) < stopping_threshold:
                    logger.info("Stopping criteria met...exiting.")
                    break

        # Save model
        if save_model:
            # Check that model path and name are both provided
            if not model_path or not model_name:
                raise RuntimeError("Exception saving model: model_path and model_name must be provided")
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(self.model.state_dict(), model_path + model_name)
            logger.info(f"Saved trained model to: {model_path + model_name}")

        return loss_vs_iter

    def update_trained_gpmodel(
        self,
        data_in: List[pd.DataFrame],
    ):
        """
        data_in: new training data across all cells and for one or more simulations

        Pre-condition : This method assumes that the model is already trained and
        prediction is run on the model at least once. New data observations can only
        be added after making predictions with a model so that all test independent
        caches exist.

        If you have a trained model, you can run predict on the model.
        This way, you can use the model that has been trained and has prediction run
        on it and can be used to update the model using the new observations.

        """

        # Create Training Tensors
        train_X, train_Y = self._create_training_tensors(data_in)

        self.model = self.model.get_fantasy_model(inputs=train_X, targets=train_Y)

    def predict_distributed_gpmodel(
        self,
        prediction_dfs: List[pd.DataFrame],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts Rx power, RSRP and SINR.

        `prediction_dfs` : one prediction dataframe, per cell.

        It is assumed that columns loc_x and loc_y in each dataframe inside `prediction_dfs`
        are the same, and that they appear in the same order.

        Returns the prediction mean and standard deviation for Rx power
        as numpy ndarrays, for all locations in the dataframe,
        in the same order given, with one such array per cell.

        Mutates `prediction_dfs` and adds columns for predicted Rx mean and stddev.

        Returns, in order :
            prediction mean for Rx power (one numpy ndarray per cell)
            prediction std dev for Rx power (one numpy ndarray per cell)
            combined RF dataframe with RSRP and SINR
        """
        self.model.eval()

        num_locations = prediction_dfs[0].shape[0]
        pred_means = torch.zeros([num_locations, self.num_cells], dtype=torch.float32)
        pred_stds = torch.zeros([num_locations, self.num_cells], dtype=torch.float32)
        predict_X = torch.zeros([self.num_cells, num_locations, self.num_features], dtype=torch.float32)

        for m in range(self.num_cells):
            if self.norm_method == NormMethod.MINMAX:
                predict_x_cell = (prediction_dfs[m][self.x_columns] - self.xmin[m]) / (self.xmax[m] - self.xmin[m])
            elif self.norm_method == NormMethod.ZSCORE:
                predict_x_cell = (prediction_dfs[m][self.x_columns] - self.xmeans[m]) / self.xstds[m]

            predict_X_cell = torch.tensor(predict_x_cell.iloc[:, :].values, dtype=torch.float32)
            predict_X[m] = predict_X_cell.reshape(shape=(1, -1, self.num_features))

        if self.is_cuda:
            logger.info("Cuda enabled for test data.")
            predict_X = predict_X.cuda()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.model.likelihood(self.model(predict_X))
            mean = observed_pred.mean
            var = observed_pred.variance

        pred_means = mean.detach().cpu().numpy() * self.ystds + self.ymeans
        pred_stds = np.sqrt(var.detach().cpu().numpy()) * self.ystds

        # add pred_means and pred_std to prediction_dfs
        for idx in range(len(prediction_dfs)):
            prediction_dfs[idx][constants.RXPOWER_DBM] = pred_means[idx]
            prediction_dfs[idx][constants.RXPOWER_STDDEV_DBM] = pred_stds[idx]

        return pred_means, pred_stds
