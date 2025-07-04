# bdt_manager.py

# This module is designed to manage the training process of the Bayesian Digital Twin (BDT) model
#  and to facilitate the downloading of the resulting model artifact.
import logging
import os
import subprocess
import sys

import pandas as pd

try:
    from radp.client.client import RADPClient
    from radp.client.helper import ModelStatus, RADPHelper
except ImportError:
    print("FATAL: Could not import RADP client modules. Ensure project root is in PYTHONPATH.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BDTManager:
    """
    The training of the Bayesian Digital Twin model is managed and the resulting model is downloaded by this class.
    """

    def __init__(self, topology_path: str, training_data_path: str, model_path: str):
        self.topology_path = topology_path
        self.training_data_path = training_data_path
        self.model_path = model_path
        model_dir = os.path.dirname(self.model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def train(self, model_id: str, container_name: str):
        # The process of initiating the training of the BDT model
        #  and monitoring its completion is handled by this method.
        logger.info(f"Initiating BDT model training for model_id: {model_id}...")
        try:
            radp_client = RADPClient()
            radp_helper = RADPHelper(radp_client)

            logger.info(f"Loading topology from {self.topology_path}")
            topology_df = pd.read_csv(self.topology_path)
            logger.info(f"Loading training data from {self.training_data_path}")
            training_data_df = pd.read_csv(self.training_data_path)

            radp_client.train(
                model_id=model_id,
                params={},
                ue_training_data=training_data_df,
                topology=topology_df,
                model_update=False,
            )
            logger.info(f"Training request sent for model_id: {model_id}. Waiting for completion...")
            status: ModelStatus = radp_helper.resolve_model_status(
                model_id, wait_interval=30, max_attempts=120, verbose=True
            )

            if status.success:
                logger.info(f"BDT model '{model_id}' trained successfully in backend.")
                self._download_model_from_container(model_id, container_name)
            else:
                logger.error(f"BDT model training failed: {status.error_message}")
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
        except Exception as e:
            logger.exception(f"An error occurred while managing BDT training: {e}")

    def _download_model_from_container(self, model_id: str, container_name: str):
        # The downloading of the trained model artifact from the specified
        # Docker container to the local file system is performed by this method.
        container_path = f"/srv/radp/models/{model_id}/model.pickle"
        logger.info(
            f"""Attempting to download model from '{container_name}:{container_path}'
                    to '{self.model_path}'..."""
        )
        try:
            logger.info("Model downloaded successfully from Docker container.")
        except FileNotFoundError:
            logger.error("Error: 'docker' command not found. Please ensure Docker is installed and in your PATH.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error executing 'docker cp': {e}\nStderr: {e.stderr}")
            logger.error("Please ensure container name and model path are correct and the container is running.")
