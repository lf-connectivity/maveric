# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
The Consume Simulation Output API handler.

This API allows a client consume the final output of a simulation
event. Once consumed, the data is deleted
"""

import logging
import os

from api_manager.exceptions.invalid_parameter_exception import InvalidParameterException
from api_manager.exceptions.simulation_output_not_found_exception import SimulationOutputNotFoundException

from radp.common.helpers.file_system_helper import RADPFileSystemHelper

logger = logging.getLogger(__name__)


class ConsumeSimulationOutputHandler:
    """The API handler for consuming a simulation's output"""

    def handle_consume_simulation_output_request(self, simulation_id: str) -> str:
        """Handle a consume simulation ouput request"""

        # validate input
        self._validate_request(simulation_id)

        # get simulation output zipfile
        output_zip_file_path = RADPFileSystemHelper.gen_sim_output_zip_file_path(simulation_id)

        if not os.path.exists(output_zip_file_path):
            logger.warning(f"Unable to find simulation output for simulation: {simulation_id}")
            raise SimulationOutputNotFoundException(simulation_id)

        logger.info(f"Found output zip file at '{output_zip_file_path}'")
        # return the path for application layer to send file
        return output_zip_file_path

    def _validate_request(self, simulation_id):
        """Validate simulation_id is valid string"""
        if not simulation_id:
            logger.exception("Empty or missing string provided for simulation_id")
            raise InvalidParameterException(
                "Invalid request sent to consume simulation output API:"
                "empty or missing string provided for simulation_id"
            )
