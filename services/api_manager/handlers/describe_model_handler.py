# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
The Describe Model API handler.

This API allows a client to obtain the status of a Digital Twin model.
The API runs validation ensuring that the model exists in filesystem.
"""

import logging
from typing import Dict

from api_manager.exceptions.invalid_parameter_exception import InvalidParameterException
from api_manager.exceptions.model_not_found_exception import ModelNotFoundException

from radp.common.helpers.file_system_helper import RADPFileSystemHelper

logger = logging.getLogger(__name__)


class DescribeModelHandler:
    """The API handler for describing a saved model"""

    def handle_describe_model_request(self, model_id: str) -> Dict:
        """Handle describe model request"""

        # validate input
        self._validate_request(model_id)

        # load metadata
        try:
            model_metadata = RADPFileSystemHelper.load_model_metadata(model_id=model_id)
        except FileNotFoundError:
            logger.exception(f"Exception describing model: model '{model_id}' not found")
            raise ModelNotFoundException(model_id)
        # TODO: implement a response DTO to validate response content
        return model_metadata

    def _validate_request(self, model_id):
        """Validate model_id is valid string"""
        if not model_id:
            logger.exception("Empty or missing string provided for model_id")
            raise InvalidParameterException(
                "Invalid request sent to describe model API: empty or missing string provided for model_id"
            )
