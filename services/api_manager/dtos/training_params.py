# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from api_manager.dtos.base_dto import BaseDTO


@dataclass
class DigitalTwinTrainingParams(BaseDTO):
    maxiter: int = 100  # max # iterations to run training
    lr: float = 0.05  # learning rate
    stopping_threshold: float = 1e-4
