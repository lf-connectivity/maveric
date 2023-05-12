# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

# Create the Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create the Handler
logger_handler = logging.StreamHandler()
logger_handler.setLevel(logging.INFO)

# Create a Formatter for formatting the log messages
formatter = logging.Formatter("[%(asctime)s] %(levelname)s:  %(message)s")
# Add the Formatter to the Handler
logger_handler.setFormatter(formatter)

# Add the Handler to the Logger
logger.addHandler(logger_handler)

# prevent logger propogating to root logger
logger.propagate = False
