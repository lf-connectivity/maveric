# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional

import pandas as pd

from radp.common import constants
from radp.common.enums import ModelStatus, ModelType, SimulationStage
from radp.common.helpers.file_system_safety import atomic_write
from radp.utility.pandas_utils import read_feather_df, write_feather_df

logger = logging.getLogger(__name__)


class RADPFileSystemHelper:
    """Helper class to support operations on radp file system"""

    @staticmethod
    def gen_simulation_directory(simulation_id: str) -> str:
        """Generate a specific simulation directory using an ID"""
        return os.path.join(constants.SIMULATION_DATA_FOLDER, simulation_id)

    @staticmethod
    def gen_simulation_metadata_file_path(simulation_id: str) -> str:
        """Helper method to generated simulation metadata file path"""
        sim_metadata_file_name = f"{constants.SIM_METADATA_FILE_NAME}.{constants.SIM_METADATA_FILE_EXTENSION}"
        return os.path.join(
            constants.SIMULATION_DATA_FOLDER,
            simulation_id,
            sim_metadata_file_name,
        )

    @staticmethod
    def gen_simulation_ue_data_file_path(simulation_id: str) -> str:
        """Helper method to generated simulation ue data file path"""
        simulation_directory = RADPFileSystemHelper.gen_simulation_directory(
            simulation_id
        )
        return os.path.join(
            simulation_directory,
            f"{constants.UE_DATA_FILE_NAME}.{constants.DF_FILE_EXTENSION}",
        )

    @staticmethod
    def gen_simulation_cell_config_file_path(simulation_id: str) -> str:
        """Helper method to generated simulation config file path"""
        simulation_directory = RADPFileSystemHelper.gen_simulation_directory(
            simulation_id
        )
        return os.path.join(
            simulation_directory,
            f"{constants.CONFIG_FILE_NAME}.{constants.DF_FILE_EXTENSION}",
        )

    @staticmethod
    def load_simulation_metadata(simulation_id: str) -> Dict:
        """Helper method to load simulation metadata to an object"""
        metadata_file_path = RADPFileSystemHelper.gen_simulation_metadata_file_path(
            simulation_id
        )
        try:
            with open(metadata_file_path, "r") as json_file:
                return json.load(json_file)
        except Exception as e:
            logger.exception(
                f"Exception occurred while loading metadata for simulation: {simulation_id}: {e}"
            )
            raise e

    @staticmethod
    def save_simulation_metadata(sim_metadata: Dict, simulation_id: str):
        """Helper method to save simulation metadata to file"""
        metadata_file_path = RADPFileSystemHelper.gen_simulation_metadata_file_path(
            simulation_id
        )

        try:
            with atomic_write(metadata_file_path, "w") as json_file:
                json.dump(sim_metadata, json_file)
        except Exception as e:
            logger.exception(
                f"Exception occurred while saving metadata for simulation: {simulation_id}: {e}"
            )
            raise e

    @staticmethod
    def save_simulation_ue_data(simulation_id: str, ue_data_file_path: str):
        """Helper method to save simulation ue data to simulation directory

        ue_data_file_path - file path of ue data csv file passed in by user
        """
        sim_ue_data_file_path = RADPFileSystemHelper.gen_simulation_ue_data_file_path(
            simulation_id
        )
        try:
            # load UE data df and save to feather format
            with open(ue_data_file_path, "r") as csv_file:
                ue_data_df = pd.read_csv(csv_file)
        except Exception as e:
            logger.exception(f"Exception occurred while reading UE data csv file: {e}")
            raise Exception(e)
        write_feather_df(file_path=sim_ue_data_file_path, df=ue_data_df)
        logger.info(f"Saved UE data to {sim_ue_data_file_path}")

    @staticmethod
    def save_simulation_cell_config(simulation_id: str, config_file_path: str):
        """Helper method to save simulation cell config to simulation directory

        config_file_path - file path of config csv file passed in by user
        """
        sim_config_file_path = (
            RADPFileSystemHelper.gen_simulation_cell_config_file_path(simulation_id)
        )

        try:
            # load config df and save to feather format
            with open(config_file_path, "r") as csv_file:
                config_df = pd.read_csv(csv_file)
        except Exception as e:
            logger.exception(f"Exception occurred while reading config csv file: {e}")
            raise Exception(e)
        write_feather_df(file_path=sim_config_file_path, df=config_df)
        logger.info(f"Saved cell config to {sim_config_file_path}")

    # TODO: make this method more robust
    @staticmethod
    def hash_val_found_in_output_folder(stage: SimulationStage, hash_val: str) -> bool:
        """Check whether a hash_val exists in an output data layer"""
        stage_output_folder = os.path.join(
            constants.SIMULATION_DATA_FOLDER,
            constants.SIMULATION_OUTPUTS_FOLDER,
            stage.value,
        )
        for object in os.listdir(stage_output_folder):
            if hash_val in object:
                return True
        return False

    @staticmethod
    def gen_stage_output_file_path(
        stage: SimulationStage, hash_val: str, batch: int
    ) -> str:
        """Helper method to generate a file path for a specific stage output"""
        stage_output_folder = os.path.join(
            constants.SIMULATION_DATA_FOLDER,
            constants.SIMULATION_OUTPUTS_FOLDER,
            stage.value,
        )
        file_name = f"{stage.value}-{hash_val}-{batch}.{constants.DF_FILE_EXTENSION}"
        return os.path.join(stage_output_folder, file_name)

    @staticmethod
    def gen_sim_output_zip_file_path(simulation_id: str, include_ext=True):
        """Generate the zip file path for a given simulation"""
        zip_file_name = f"{simulation_id}-{constants.SIM_OUTPUT_FILE_SUFFIX}"
        zip_file_name = zip_file_name + (
            f".{constants.SIM_OUTPUT_FILE_EXTENSION}" if include_ext else ""
        )
        return os.path.join(
            constants.SIMULATION_DATA_FOLDER, simulation_id, zip_file_name
        )

    @staticmethod
    def gen_sim_output_directory(simulation_id: str):
        """Generate the zip file path for a given simulation"""
        return os.path.join(
            constants.SIMULATION_DATA_FOLDER,
            simulation_id,
            constants.SIM_OUTPUT_DIRECTORY,
        )

    # TODO clean up this method
    @staticmethod
    def zip_output_files_to_simulation_folder_as_csvs(
        simulation_id: str,
        stage: SimulationStage,
        hash_val: str,
        num_batches: int,
    ):
        """Helper method to generate simulation folder path from id"""
        # get stage output file paths and output path
        stage_output_file_paths = RADPFileSystemHelper.get_stage_output_file_paths(
            stage=stage, hash_val=hash_val, num_batches=num_batches
        )
        simulation_output_directory = RADPFileSystemHelper.gen_sim_output_directory(
            simulation_id
        )

        # create output directory if it does not already exist
        if not os.path.exists(simulation_output_directory):
            os.makedirs(simulation_output_directory)

        # convert each feather file to csv
        for fp in stage_output_file_paths:
            # create new file name and path
            new_file_name = os.path.basename(fp).replace(
                f".{constants.DF_FILE_EXTENSION}",
                f".{constants.USER_FACING_DF_EXTENSION}",
            )
            new_file_path = os.path.join(simulation_output_directory, new_file_name)

            # write feather format to csv for output
            try:
                output_df = read_feather_df(fp)
                output_df.to_csv(new_file_path, index=False)
            except Exception as e:
                logger.exception(
                    "Exception occurred while writing csv's to output folder"
                )
                raise e

        # get the zip file path
        zip_file_path = RADPFileSystemHelper.gen_sim_output_zip_file_path(
            simulation_id, include_ext=False
        )

        try:
            # chdir to simulation output directory to only zip files
            # see https://docs.python.org/3/library/shutil.html#shutil.make_archive
            os.chdir(simulation_output_directory)
            zip_file_path = shutil.make_archive(
                base_name=zip_file_path,
                format="zip",
            )
            logger.info(f"Zipped output files to {zip_file_path}")
        except Exception as e:
            logger.exception(
                f"Exception occurred zipping files in simulation: {simulation_id}: {e}"
            )
            raise e

    @staticmethod
    def get_stage_output_file_paths(
        stage: SimulationStage, hash_val: str, num_batches: int
    ) -> List[str]:
        """Helper method to get list of output files for a stage"""
        # get output folder
        stage_output_folder = os.path.join(
            constants.SIMULATION_DATA_FOLDER,
            constants.SIMULATION_OUTPUTS_FOLDER,
            stage.value,
        )

        # generate list of files using hash and batch iteration
        file_name = f"{stage.value}-{hash_val}"
        file_path_without_batch = os.path.join(stage_output_folder, file_name)
        return [
            f"{file_path_without_batch}-{batch}.{constants.DF_FILE_EXTENSION}"
            for batch in range(1, num_batches + 1)
        ]

    @staticmethod
    def clear_output_data_from_stage(
        stage: SimulationStage, save_hash_val: Optional[str]
    ):
        """
        Clear the output from a stage unless it contains the save hash value

        If "save_hash_val" is None then just clear the stage output entirely.
        """
        # generate the stage output folder
        stage_output_folder = os.path.join(
            constants.SIMULATION_DATA_FOLDER,
            constants.SIMULATION_OUTPUTS_FOLDER,
            stage.value,
        )

        try:
            # gather all outputs in stage folder
            output_file_names = os.listdir(stage_output_folder)
            delete_count = 0
            # if no save hash val then delete everything
            if save_hash_val is None:
                for file_name in output_file_names:
                    file_path = os.path.join(stage_output_folder, file_name)
                    os.remove(file_path)
                    delete_count += 1
            else:
                # save hash val provided, save files which contain it
                for file_name in output_file_names:
                    if save_hash_val in file_name:
                        continue
                    else:
                        file_path = os.path.join(stage_output_folder, file_name)
                        os.remove(file_path)
                        delete_count += 1
            logger.info(
                f"Cleared {delete_count} unused outputs from stage: {stage.value}"
            )
        except Exception as e:
            logger.exception(
                f"Exception occurred while deleting outputs in stage: {stage.value}"
            )
            raise e

    @staticmethod
    def gen_model_metadata_frame(
        model_id: str,
        model_type: ModelType,
        status: ModelStatus,
        model_specific_params: Dict,
    ) -> Dict:
        """Generate model metadata object given parameters"""
        # build model frame
        model_metadata: Dict[Any, Any] = {
            constants.MODEL_ID: model_id,
            constants.MODEL_TYPE: model_type.value,
            constants.STATUS: status.value,
        }

        # add the model-specific parameters under model type key
        model_metadata[model_type.value] = model_specific_params

        return model_metadata

    @staticmethod
    def gen_model_folder_path(model_id: str) -> str:
        """Generate a model folder path from ID"""
        return os.path.join(constants.MODELS_FOLDER, model_id)

    @staticmethod
    def gen_model_file_path(model_id: str) -> str:
        """Generate model file path from ID"""
        model_folder_path = RADPFileSystemHelper.gen_model_folder_path(model_id)
        return os.path.join(
            model_folder_path,
            f"{constants.MODEL_FILE_NAME}.{constants.MODEL_FILE_EXTENSION}",
        )

    @staticmethod
    def gen_model_metadata_file_path(model_id: str) -> str:
        """Generate model mdetadata file path from ID"""
        model_folder_path = RADPFileSystemHelper.gen_model_folder_path(model_id)
        return os.path.join(
            model_folder_path,
            f"{constants.MODEL_METADATA_FILE_NAME}.{constants.MODEL_METADATA_FILE_EXTENSION}",
        )

    @staticmethod
    def gen_model_topology_file_path(model_id: str) -> str:
        """Generate model topology file path from model ID"""
        model_folder_path = RADPFileSystemHelper.gen_model_folder_path(model_id)
        return os.path.join(
            model_folder_path,
            f"{constants.TOPOLOGY_FILE_NAME}.{constants.DF_FILE_EXTENSION}",
        )

    @staticmethod
    def load_model_metadata(model_id: str) -> Dict:
        """Get a model's metadata object"""
        metadata_file_path = RADPFileSystemHelper.gen_model_metadata_file_path(model_id)

        try:
            with open(metadata_file_path, "r") as json_file:
                metadata = json.load(json_file)
                logger.debug(f"Loaded metadata for model: {model_id}")
                return metadata
        except Exception as e:
            logger.exception(
                f"Exception occurred loading metadata for model: {model_id}"
            )
            raise e

    @staticmethod
    def save_model_metadata(model_id: str, model_metadata: Dict):
        """Save a model's metadata to file"""
        model_folder_path = RADPFileSystemHelper.gen_model_folder_path(model_id)

        # TODO: remove this check once directory is created in handler
        # create the models folder if it does not exist
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        # get the metadata file path
        metadata_file_path = RADPFileSystemHelper.gen_model_metadata_file_path(model_id)

        try:
            with atomic_write(metadata_file_path, "w") as json_file:
                json.dump(model_metadata, json_file)
            logger.debug(f"Saved metadata for model: {model_id}")
        except Exception as e:
            logger.exception(
                f"Exception occurred loading metadata for model: {model_id}"
            )
            raise e

    @staticmethod
    def check_model_exists(model_id: str) -> bool:
        """Check if a given model exists"""
        model_file_path = RADPFileSystemHelper.gen_model_file_path(model_id)
        return os.path.exists(model_file_path)

    @staticmethod
    def get_model_status(model_id: str) -> ModelStatus:
        """Get model status from ID"""
        model_metadata = RADPFileSystemHelper.load_model_metadata(model_id)
        return ModelStatus[model_metadata[constants.STATUS]]

    @staticmethod
    def get_model_type(model_id: str) -> ModelType:
        """Get model type from ID"""
        model_metadata = RADPFileSystemHelper.load_model_metadata(model_id)
        return ModelType[model_metadata[constants.MODEL_TYPE]]
