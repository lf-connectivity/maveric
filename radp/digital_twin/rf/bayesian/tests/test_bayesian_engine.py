# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This test demonstrates how the BayesianDigitalTwin (a RADP RF Digital Twin model)
is trained and inferred on, using small sample data.
"""

import logging
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from radp.utility.simulation_utils import seed_everything
from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin
from radp.digital_twin.utils import constants
from radp.digital_twin.utils.gis_tools import GISTools


# Hard-coded sample site config data and UE data
def get_sample_site_config_and_ue_data():
    site_configs_df = pd.DataFrame(
        {
            "cell_id": ["Cell1", "Cell2", "Cell3"],
            "cell_az_deg": [0, 120, 240],
            "cell_el_deg": [0, 0, 0],
            "cell_lat": [35.690555, 35.690555, 35.690555],
            "cell_lon": [139.69194, 139.69194, 139.69194],
            "cell_carrier_freq_mhz": [2100, 2100, 2100],
        }
    )

    # synthetic sample UE data : 5 data points

    ue_data_df = pd.DataFrame(
        {
            "cell_id": [
                "Cell1",
                "Cell1",
                "Cell1",
                "Cell1",
                "Cell1",
                "Cell2",
                "Cell2",
                "Cell2",
                "Cell2",
                "Cell2",
                "Cell3",
                "Cell3",
                "Cell3",
                "Cell3",
                "Cell3",
            ],
            "avg_rsrp": [
                -80,
                -70,
                -75,
                -72,
                -71,
                -100,
                -101,
                -102,
                -99,
                -98,
                -77,
                -78,
                -79,
                -80,
                -79,
            ],
            "lon": [
                139.699058,
                139.707889,
                139.700023,
                139.702645,
                139.702645,
                139.707067,
                139.700519,
                139.701644,
                139.701644,
                139.701644,
                139.702793,
                139.703664,
                139.704312,
                139.704312,
                139.70469,
            ],
            "lat": [
                35.644327,
                35.64781,
                35.643857,
                35.645913,
                35.64591,
                35.647007,
                35.644816,
                35.645196,
                35.645196,
                35.645198,
                35.645571,
                35.645876,
                35.646208,
                35.646209,
                35.645790,
            ],
        }
    )
    return site_configs_df, ue_data_df


def get_x_max_and_x_min():
    """
    This provides reasonable ranges for typical parameters used in RF Digital Twin.
    This is useful for normalizing values before training.
    """

    x_max = {
        "cell_el_deg": 50,
        "cell_lat": 90,
        "cell_lon": 180,
        "distance": 5,  # log(1 + max distance of 50000 meters)
        "relative_bearing": 360,
        "relative_tilt": 180,
        "relative_tilt_squared": 32400,
    }
    x_min = {
        "cell_el_deg": -10,
        "cell_lat": -90,
        "cell_lon": -180,
        "distance": 0,
        "relative_bearing": 0,
        "relative_tilt": -90,
        "relative_tilt_squared": 0,
    }

    return x_max, x_min


def augment_ue_data(ue_data_df: pd.DataFrame, site_configs_df: pd.DataFrame):
    """
    This method takes in UE data and site config and adds to the UE data
    as well as engineers new training features in (e.g. log distance and
    relative bearing)
    """

    for i in ue_data_df.index:
        site_config = site_configs_df[site_configs_df.cell_id == ue_data_df.at[i, "cell_id"]]
        ue_data_df.at[i, "cell_id"] = site_config.cell_id.values[0]
        ue_data_df.at[i, "loc_x"] = ue_data_df.at[i, "lon"]
        ue_data_df.at[i, "loc_y"] = ue_data_df.at[i, "lat"]
        ue_data_df.at[i, "cell_lat"] = site_config.cell_lat.values[0]
        ue_data_df.at[i, "cell_lon"] = site_config.cell_lon.values[0]
        ue_data_df.at[i, "cell_az_deg"] = site_config.cell_az_deg.values[0]
        ue_data_df.at[i, "cell_el_deg"] = site_config.cell_el_deg.values[0]
        ue_data_df.at[i, "cell_carrier_freq_mhz"] = site_config.cell_carrier_freq_mhz.values[0]

        ue_data_df.at[i, "log_distance"] = GISTools.get_log_distance(
            ue_data_df.at[i, "cell_lat"],
            ue_data_df.at[i, "cell_lon"],
            ue_data_df.at[i, "lat"],
            ue_data_df.at[i, "lon"],
        )
        ue_data_df.at[i, "relative_bearing"] = GISTools.get_relative_bearing(
            ue_data_df.at[i, "cell_az_deg"],
            ue_data_df.at[i, "cell_lat"],
            ue_data_df.at[i, "cell_lon"],
            ue_data_df.at[i, "lat"],
            ue_data_df.at[i, "lon"],
        )


def split_training_and_test_data(ue_data_df: pd.DataFrame, test_size: float):
    """
    This function splits the UE data into training and test data according to the defined test size
    It returns the training and test data mapped to cell_id
    """
    cell_id_ue_data_map = {k: v for k, v in ue_data_df.groupby("cell_id")}

    cell_id_training_data_map = {}
    cell_id_test_data_map = {}

    all_coords = pd.DataFrame(columns=["lat", "lon"])

    for cell_id, ue_data in cell_id_ue_data_map.items():
        train, test = train_test_split(ue_data, test_size=test_size)
        cell_id_training_data_map[cell_id] = train
        cell_id_test_data_map[cell_id] = test
        coords = test[["lat", "lon"]]
        all_coords = pd.merge(all_coords, coords, how="outer", on=["lat", "lon"])

    return cell_id_training_data_map, cell_id_test_data_map


class TestBayesianDigitalTwin(unittest.TestCase):

    def test_rf_bayesian_digital_twin(self):
        seed_everything(1)
        # get sample data
        site_configs_df, ue_data_df = get_sample_site_config_and_ue_data()

        site_configs_df.reset_index(drop=True, inplace=True)

        # feature engineering -- add relative bearing and distance
        augment_ue_data(ue_data_df, site_configs_df)

        logging.info(ue_data_df)

        # split into training/test
        cell_id_training_data_map, cell_id_test_data_map = split_training_and_test_data(ue_data_df, 0.2)

        # train
        bayesian_digital_twin_map = {}

        x_max, x_min = get_x_max_and_x_min()
        # Test data expected_loss_vs_iter is generated with seed_everything(1)
        expected_loss_vs_iter = {
            "Cell1": [
                1.3360857963562012,
                1.3250646591186523,
                1.3147950172424316,
                1.305358648300171,
                1.296657919883728,
                1.288337230682373,
                1.2800240516662598,
                1.2716268301010132,
                1.2632856369018555,
                1.2552096843719482,
                1.2475690841674805,
                1.2404227256774902,
                1.2336828708648682,
                1.2271721363067627,
                1.2207590341567993,
                1.2144416570663452,
                1.208317518234253,
                1.202500820159912,
                1.197044849395752,
                1.1918997764587402,
            ],
            "Cell2": [
                1.4958915710449219,
                1.4850997924804688,
                1.476243019104004,
                1.46876859664917,
                1.461909294128418,
                1.455451250076294,
                1.4494956731796265,
                1.4441474676132202,
                1.4394338130950928,
                1.4352850914001465,
                1.4315435886383057,
                1.4280192852020264,
                1.4245692491531372,
                1.4211323261260986,
                1.4177082777023315,
                1.4143176078796387,
                1.41096830368042,
                1.4076392650604248,
                1.404287576675415,
                1.400879144668579,
            ],
            "Cell3": [
                1.4421265125274658,
                1.4319207668304443,
                1.4234235286712646,
                1.4165308475494385,
                1.4108211994171143,
                1.4056122303009033,
                1.4002456665039062,
                1.3943250179290771,
                1.3877692222595215,
                1.3807103633880615,
                1.3733720779418945,
                1.3659957647323608,
                1.3587980270385742,
                1.3519399166107178,
                1.3454982042312622,
                1.3394557237625122,
                1.3337373733520508,
                1.3282841444015503,
                1.323117971420288,
                1.318347454071045,
            ],
        }
        for cell_id, training_data in cell_id_training_data_map.items():
            bayesian_digital_twin_map[cell_id] = BayesianDigitalTwin(
                data_in=[training_data],
                x_columns=[
                    constants.CELL_EL_DEG,
                    constants.LOG_DISTANCE,
                    constants.RELATIVE_BEARING,
                ],
                y_columns=["avg_rsrp"],
                x_max=x_max,
                x_min=x_min,
            )
            loss_vs_iter = bayesian_digital_twin_map[cell_id].train_distributed_gpmodel(
                maxiter=20,
                stopping_threshold=1e-5,
            )
            logging.info(
                f"\nTrained {len(loss_vs_iter)} epochs of Bayesian Digital Twin (Gaussian Process Regression) "
                f"on {len(training_data)} data points"
                f" for cell {cell_id}, with min learning loss {min(loss_vs_iter):0.5f}, "
                f"avg learning loss {np.mean(loss_vs_iter):0.5f} and final learning loss {loss_vs_iter[-1]:0.5f}"
            )
            self.assertEqual(loss_vs_iter.tolist(), expected_loss_vs_iter[cell_id])

        # predict/test
        # Test data EXPECTED_MAE and EXPECTED_MAPE is generated with seed_everything(1)
        EXPECTED_MAE = {"Cell1": 1.7881388210634555, "Cell2": 0.07000514313754991, "Cell3": 0.07576626164573952}
        EXPECTED_MAPE = {"Cell1": 2.384185094751274, "Cell2": 0.07000514313754991, "Cell3": 0.09590666031106268}
        for cell_id, testing_data in cell_id_test_data_map.items():
            (pred_means, _) = bayesian_digital_twin_map[cell_id].predict_distributed_gpmodel(
                prediction_dfs=[testing_data]
            )
            MAE = abs(testing_data.avg_rsrp - pred_means[0]).mean()
            # mean absolute percentage error
            MAPE = 100 * abs((testing_data.avg_rsrp - pred_means[0]) / testing_data.avg_rsrp).mean()
            logging.info(
                f"cell_id = {cell_id}, MAE = {MAE:0.5f} dB, MAPE = {MAPE:0.5f} %,"
                "# test points = {len(testing_data.avg_rsrp)}"
            )
            self.assertEqual(MAE, EXPECTED_MAE[cell_id])
            self.assertEqual(MAPE, EXPECTED_MAPE[cell_id])
