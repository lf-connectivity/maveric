# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from datetime import datetime, timezone
from typing import Tuple

from werkzeug.datastructures import FileStorage

from radp.common import constants
from radp.common.enums import InputFileType

logger = logging.getLogger(__name__)


def get_utc_timestamp() -> str:
    """Append the current UTC time to a string"""
    now_utc = datetime.now(timezone.utc)
    return now_utc.strftime("%Y_%m_%d-%I_%M_%S_%p")


def save_file_from_flask(file_storage: FileStorage, upload_folder: str, file_name: str) -> str:
    """Helper method to save a werkzeug FileStorage file to disk"""
    try:
        # create folder if it does not exist
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
            logger.info(f"Created new directory '{upload_folder}'")

        file_path = os.path.join(upload_folder, file_name)
        file_storage.save(file_path)
        return file_path
    except Exception as e:
        logger.exception(f"Exception occurred while saving file '{file_name}': {e}")
        raise e


def save_input_file(input_file_type: InputFileType, file_storage: FileStorage):
    """Save user inputs to the inputs folder"""
    input_files_folder = constants.INPUT_FILES_FOLDER
    utc_timestamp = get_utc_timestamp()
    file_name = f"{input_file_type.value}-{utc_timestamp}.csv"

    return save_file_from_flask(
        file_storage=file_storage,
        upload_folder=input_files_folder,
        file_name=file_name,
    )


def bootstrap_radp_filesystem():
    """Create the directories required for RADP system to run"""
    SYSTEM_DIRECTORIES = [
        "/srv/radp/input_files",
        "/srv/radp/models",
        "/srv/radp/simulation_data/outputs/ue_tracks_generation",
        "/srv/radp/simulation_data/outputs/rf_prediction",
        "/srv/radp/simulation_data/outputs/protocol_emulation",
    ]

    for directory in SYSTEM_DIRECTORIES:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except Exception as e:
                logger.exception(f"Exception occurred creating directory '{directory}': {e}")
                raise e

    directories_output_string = "\n".join(SYSTEM_DIRECTORIES)
    logger.info(
        "Successfully created the following directories:\n{directories}".format(directories=directories_output_string)
    )


def get_directory_and_file_name(file_path: str) -> Tuple[str, str]:
    """Get a directory and file name from a path"""
    if not file_path:
        raise ValueError("Cannot extract directory/file name from empty string")

    # return directory and base name
    return os.path.dirname(file_path), os.path.basename(file_path)
