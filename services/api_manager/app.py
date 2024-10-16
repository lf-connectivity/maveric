# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging

from flask import Flask, jsonify, request, send_file

from api_manager.exceptions.base_api_exception import APIException
from api_manager.exceptions.invalid_parameter_exception import InvalidParameterException
from api_manager.handlers.consume_simulation_output_handler import (
    ConsumeSimulationOutputHandler,
)
from api_manager.handlers.describe_model_handler import DescribeModelHandler
from api_manager.handlers.describe_simulation_handler import DescribeSimulationHandler
from api_manager.handlers.simulation_handler import SimulationHandler
from api_manager.handlers.train_handler import TrainHandler
from api_manager.utils.file_io import bootstrap_radp_filesystem, save_input_file

from radp.common import constants
from radp.common.enums import InputFileType

logger = logging.getLogger(__name__)

app = Flask(__name__)

# TODO: find out if there is a way to bootstrap system directories from docker compose file
# BOOTSTRAP radp filesystem
bootstrap_radp_filesystem()


# add custom exception handler to return formatted json to client
@app.errorhandler(APIException)
def handle_exception(e):
    """Return custom JSON when an APIException (sub)class"""
    response = {"status_code": e.code, "error": e.message}
    return jsonify(response), e.code


# add 500 exception handler to return formatted json to client
@app.errorhandler(500)
def handle_uncaught_exception(e):
    """Return JSON for any other internal server error"""
    logger.exception(f"Server encountered 500 error during execution: {e}")
    response = {"error": "Internal server error encountered"}
    return jsonify(response), 500


@app.route("/model/<model_id>", methods=["GET"])
def describe_model(model_id: str):
    logger.info(f"Received API request to describe model: {model_id}")
    return jsonify(DescribeModelHandler().handle_describe_model_request(model_id))


@app.route("/train", methods=["POST"])
def train():
    logger.info("Received API request to train model")

    # verify request contains payload and csv files
    if constants.REQUEST_PAYLOAD_FILE_KEY not in request.files:
        raise InvalidParameterException(
            f"Invalid request, missing file input '{constants.REQUEST_PAYLOAD_FILE_KEY}'"
        )
    if constants.REQUEST_UE_TRAINING_DATA_FILE_KEY not in request.files:
        raise InvalidParameterException(
            f"Invalid request, missing file input '{constants.REQUEST_UE_TRAINING_DATA_FILE_KEY}'"
        )
    if constants.REQUEST_TOPOLOGY_FILE_KEY not in request.files:
        raise InvalidParameterException(
            f"Invalid request, missing file input '{constants.REQUEST_TOPOLOGY_FILE_KEY}'"
        )

    payload = json.load(request.files[constants.REQUEST_PAYLOAD_FILE_KEY])

    # save training files to filesystem
    ue_training_data_file_path = save_input_file(
        input_file_type=InputFileType.UE_TRAINING_DATA,
        file_storage=request.files[constants.REQUEST_UE_TRAINING_DATA_FILE_KEY],
    )
    topology_file_path = save_input_file(
        input_file_type=InputFileType.TOPOLOGY,
        file_storage=request.files[constants.REQUEST_TOPOLOGY_FILE_KEY],
    )

    # create files object to pass to API handler
    files = {
        constants.UE_TRAINING_DATA_FILE_PATH_KEY: ue_training_data_file_path,
        constants.TOPOLOGY_FILE_PATH_KEY: topology_file_path,
    }

    # return jsonified response from handler
    return jsonify(TrainHandler().handle_train_request(payload, files=files))


# TODO implement this API route to accept and parse POST data
@app.route("/simulation", methods=["POST"])
def simulation():
    logger.info("Received API request to run simulation")

    # verify request contains json payload
    if constants.REQUEST_PAYLOAD_FILE_KEY not in request.files:
        raise InvalidParameterException(
            f"Invalid request, missing file input '{constants.REQUEST_PAYLOAD_FILE_KEY}'"
        )
    payload = json.load(request.files[constants.REQUEST_PAYLOAD_FILE_KEY])

    # store and pass whatever files are provided
    files = {}

    # check if UE data or config files provided
    if constants.REQUEST_UE_DATA_FILE_KEY in request.files:
        files[constants.UE_DATA_FILE_PATH_KEY] = save_input_file(
            input_file_type=InputFileType.UE_DATA,
            file_storage=request.files[constants.REQUEST_UE_DATA_FILE_KEY],
        )
    if constants.REQUEST_CONFIG_FILE_KEY in request.files:
        files[constants.CONFIG_FILE_PATH_KEY] = save_input_file(
            input_file_type=InputFileType.CONFIG,
            file_storage=request.files[constants.REQUEST_CONFIG_FILE_KEY],
        )
    return jsonify(SimulationHandler().handle_simulation_request(payload, files))


@app.route("/simulation/<simulation_id>", methods=["GET"])
def describe_simulation(simulation_id: str):
    logger.info(f"Received API request to describe simulation: {simulation_id}")
    return jsonify(
        DescribeSimulationHandler().handle_describe_simulation_request(simulation_id)
    )


@app.route("/simulation/<simulation_id>/download", methods=["GET"])
def consume_simulation_output(simulation_id: str):
    logger.info(f"Received API request to consume simulation output: {simulation_id}")
    output_zip_file_path = (
        ConsumeSimulationOutputHandler().handle_consume_simulation_output_request(
            simulation_id
        )
    )
    return send_file(output_zip_file_path)
