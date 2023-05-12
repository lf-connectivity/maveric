# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from api_manager.exceptions.base_api_exception import APIException


class SimulationOutputNotFoundException(APIException):
    """Exception raised when user attempts to consume non-existent simulation output"""

    def __init__(self, simulation_id: str):
        self.code = 404
        self.message = (
            f"Simulation outputs not found for simulation '{simulation_id}'."
            f"This may be because simulation '{simulation_id}' has not finished execution yet."
        )
