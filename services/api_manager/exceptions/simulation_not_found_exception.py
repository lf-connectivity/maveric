# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from api_manager.exceptions.base_api_exception import APIException


class SimulationNotFoundException(APIException):
    """Exception raised when user describes non-existent simulation"""

    def __init__(self, simulation_id: str):
        self.code = 404
        self.message = f"Simulation '{simulation_id}' not found"
