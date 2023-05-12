# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
The Describe Simulation API handler.

This API allows a client to describe a simulation event, providing details
on the event's status and other metadata.
"""

import logging
from typing import Dict

from api_manager.exceptions.invalid_parameter_exception import InvalidParameterException
from api_manager.exceptions.simulation_not_found_exception import SimulationNotFoundException

from radp.common.helpers.file_system_helper import RADPFileSystemHelper

logger = logging.getLogger(__name__)


class DescribeSimulationHandler:
    """The API handler for describing a simulation event"""

    def handle_describe_simulation_request(self, simulation_id: str) -> Dict:
        """Handle describe simulation request"""

        # validate input
        self._validate_request(simulation_id)

        # load metadata
        try:
            sim_metadata = RADPFileSystemHelper.load_simulation_metadata(simulation_id)
        except FileNotFoundError:
            logger.exception(f"Exception describing simulation: simulation '{simulation_id}' not found")
            raise SimulationNotFoundException(simulation_id)
        # TODO: implement a response DTO to validate response content
        return sim_metadata

    def _validate_request(self, simulation_id):
        """Validate simulation_id is valid string"""
        if not simulation_id:
            logger.exception("Empty or missing string provided for simulation_id")
            raise InvalidParameterException(
                "Invalid request sent to describe simulation API: empty or missing string provided for simulation_id"
            )
