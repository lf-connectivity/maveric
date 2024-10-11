# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from radp.common import constants
from radp.common.helpers.file_system_helper import RADPFileSystemHelper
from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin
from training.enums import ModelStatus

logger = logging.getLogger(__name__)


class TrainingDriver:
    """Class which drives training process"""

    def handle_training_job(self, job_data: Dict):
        """Handle a training job, start to finish

        job_data: the job event object pulled from jobs topic

        event object format:
            {
                job_id: str
                model_id: str
                model_update: bool
                model_file_path: str
                training_params: dict
                topology_file_path: str
                ue_training_data_file_path: str
            }
        """
        logger.info(f"Handling training job: {job_data}")

        model_id = job_data[constants.MODEL_ID]
        model_file_path = job_data[constants.MODEL_FILE_PATH]
        model_update = job_data[constants.MODEL_UPDATE]

        job_id = job_data[constants.JOB_ID]
        training_params = job_data[constants.TRAINING_PARAMS]

        topology_file_path = job_data[constants.TOPOLOGY_FILE_PATH_KEY]
        ue_training_data_file_path = job_data[constants.UE_TRAINING_DATA_FILE_PATH_KEY]

        # load training data
        try:
            with open(ue_training_data_file_path, "r") as ue_training_data_file, open(
                topology_file_path
            ) as topology_file:
                ue_training_data_df = pd.read_csv(ue_training_data_file)
                topology_df = pd.read_csv(topology_file)
        except Exception as e:
            logger.exception(
                f"Exception occurred while loading digital twin training data: {e}"
            )
            raise Exception

        # Preprocess training data
        logger.info("Preprocessing training data...")
        cell_id_ue_data_map = BayesianDigitalTwin.preprocess_ue_training_data(
            ue_training_data_df, topology_df
        )
        logger.info("Finished preprocessing training data...")
        logger.info("Starting model training...")

        """
        models are created and stored (mapped) per cell_id
        by default, the incoming csv files would overwrite this map completely
        but in "update" mode, RADP will have to:
            load the existing model map
            if it does not exist, all models are created fresh
            else, for each incoming cell_id
                if there is an existing model for it, run update
                else create model fresh
        save all cell_id to model maps
        """

        if model_update:
            model_map = BayesianDigitalTwin.load_model_map_from_pickle(model_file_path)
        else:
            model_map: Dict[str, BayesianDigitalTwin] = {}

        for cell_id, training_data in cell_id_ue_data_map.items():
            if not model_update or cell_id not in model_map:
                # train the model per cell
                model_map[cell_id] = self._train_model(
                    [training_data],
                    training_params,
                )
                logger.info("Finished training model")
            else:
                model_map[cell_id] = self._update_trained_model(
                    [training_data],
                    model_map[cell_id],
                )
                logger.info("Finished updating trained model")

            # prime the model cache by calling a mock prediction on it
            # using the first set of training data
            # so that it is ready for further updates or operations
            model_map[cell_id].predict_distributed_gpmodel([training_data.head(1)])

        # save the serialized model map object to file
        BayesianDigitalTwin.save_model_map_to_pickle(
            model_file_path=model_file_path,
            model_map=model_map,
        )

        # update the model's training status in metadata
        model_metadata = RADPFileSystemHelper.load_model_metadata(model_id)

        model_metadata[constants.STATUS] = ModelStatus.TRAINED.value
        model_metadata[constants.JOB_ID] = job_id
        model_metadata[constants.JOB_FINISHED_DATETIME] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()

        RADPFileSystemHelper.save_model_metadata(
            model_id=model_id,
            model_metadata=model_metadata,
        )

        logger.info("Saved model to file. Inference can now be run using this.")

        # TODO: implement/call cleanup method to clear training data from
        # file system and any other wrap required

    def _update_trained_model(
        self,
        training_data: List[pd.DataFrame],
        model: BayesianDigitalTwin,
    ) -> BayesianDigitalTwin:
        """Update a trained model per each cell of data"""

        model.update_trained_gpmodel(
            data_in=training_data,
        )

        return model

    def _train_model(
        self,
        training_data: List[pd.DataFrame],
        training_params: Dict,
    ) -> BayesianDigitalTwin:
        """Train a model per each cell of data"""

        # this class init method fully prepares the model to be trained
        #   but stops just short of actually calling train() on it
        model = BayesianDigitalTwin(
            data_in=training_data,
            x_columns=["cell_el_deg", "log_distance", "relative_bearing"],
            y_columns=["avg_rsrp"],
            x_min={
                "cell_el_deg": -10,
                "cell_lat": -90,
                "cell_lon": -180,
                "log_distance": 0,
                "relative_bearing": 0,
                "relative_tilt": -90,
                "relative_tilt_squared": 0,
            },
            x_max={
                "cell_el_deg": 50,
                "cell_lat": 90,
                "cell_lon": 180,
                "log_distance": 5,  # log(1 + max distance of 50000 meters)
                "relative_bearing": 360,
                "relative_tilt": 180,
                "relative_tilt_squared": 32400,
            },
        )

        loss_vs_iter = model.train_distributed_gpmodel(**training_params)
        logger.info(
            f"\nTrained {len(loss_vs_iter)} epochs of Bayesian Digital Twin (Gaussian Process Regression) "
            f"on {len(training_data)} data points "
            f"with min learning loss {min(loss_vs_iter):0.5f}, "
            f"avg learning loss {np.mean(loss_vs_iter):0.5f} and final learning loss {loss_vs_iter[-1]:0.5f}"
        )

        return model
