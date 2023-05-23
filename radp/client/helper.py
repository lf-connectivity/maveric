# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from dataclasses import dataclass
from typing import Any

from radp.client import constants
from radp.client.client import RADPClient

DEFAULT_WAIT_INTERVAL = 60
DEFAULT_MAX_ATTEMPTS = 60
DEFAULT_VERBOSE = False


@dataclass
class ModelStatus:
    success: bool
    error_message: str = ""


@dataclass
class SimulationStatus:
    success: bool
    error_message: str = ""


def print_if_verbose(msg: Any, verbose: bool) -> None:
    """Helper method to print only if user sets verbose"""
    if verbose:
        print(msg)


class RADPHelper:
    """Class to provide RADPClient-calling helper methods"""

    def __init__(self, radp_client: RADPClient) -> None:
        self.radp_client = radp_client

    def resolve_model_status(
        self,
        model_id: str,
        wait_interval: int = DEFAULT_WAIT_INTERVAL,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        verbose: bool = DEFAULT_VERBOSE,
        job_id: str = None,
    ) -> ModelStatus:
        """Resolve the status of a model within RADP system"""
        attempt = 0
        while attempt < max_attempts:
            describe_model_response = self.radp_client.describe_model(model_id)
            if not describe_model_response[constants.MODEL_EXISTS]:
                print_if_verbose("Model not yet created", verbose)
            elif describe_model_response[constants.MODEL_STATUS] != constants.MODEL_TRAINED:
                print_if_verbose("Model not yet trained", verbose)
            elif job_id and describe_model_response[constants.JOB_ID] != job_id:
                print_if_verbose("Model training job not yet complete", verbose)
            else:
                print_if_verbose("Model training job complete!", verbose)
                return ModelStatus(success=True)
            print_if_verbose(f"Waiting {wait_interval} seconds", verbose)
            time.sleep(wait_interval)
            attempt += 1
        return ModelStatus(
            success=False,
            error_message="Timed out waiting for model to exist, check RADP system logs for more details",
        )

    def resolve_simulation_status(
        self,
        simulation_id: str,
        wait_interval: int = DEFAULT_WAIT_INTERVAL,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        verbose: bool = DEFAULT_VERBOSE,
        job_id: str = None,
    ) -> SimulationStatus:
        """Resolve the status of a RADP simulation"""
        attempt = 0
        while attempt < max_attempts:
            describe_simulation_response = self.radp_client.describe_simulation(simulation_id)
            if not describe_simulation_response[constants.SIMULATION_EXISTS]:
                print_if_verbose("Simulation not yet created", verbose)
            elif describe_simulation_response[constants.SIMULATION_STATUS] != constants.SIMULATION_FINISHED:
                print_if_verbose("Simulation not yet finished", verbose)
            elif job_id and describe_simulation_response[constants.JOB_ID] != job_id:
                print_if_verbose("Simulation job not yet complete", verbose)
            else:
                print_if_verbose("Simulation job complete!", verbose)
                return SimulationStatus(success=True)
            print_if_verbose(f"Waiting {wait_interval} seconds", verbose)
            time.sleep(wait_interval)
            attempt += 1
        return SimulationStatus(
            success=False,
            error_message="Timed out waiting for simulation to finish, check RADP system logs for more details",
        )
