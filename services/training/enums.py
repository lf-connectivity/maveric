# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


class ModelType(Enum):
    RF_DIGITAL_TWIN = "rf_digital_twin"
    # TODO: add mobility model here once we allow for training mobility models
    # MOBILITY = 'mobility'


class ModelStatus(Enum):
    TRAINED = "trained"
    IN_TRAINING = "in_training"
