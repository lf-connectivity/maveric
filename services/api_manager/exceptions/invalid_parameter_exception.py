# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from api_manager.exceptions.base_api_exception import APIException


class InvalidParameterException(APIException):
    """Exception raised when API called with invalid/unexpected inputs"""

    def __init__(self, message: str = ""):
        self.code = 400
        self.message = f"Invalid parameter exception{f': {message}' if message else ''}"
