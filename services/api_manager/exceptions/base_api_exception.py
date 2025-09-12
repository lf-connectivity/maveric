# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class APIException(Exception):
    """All custom API Exceptions"""
    
    def __init__(self, message: str = ""):
        super().__init__(message)
        self.message = message
        self.code = 500  # Default error code
