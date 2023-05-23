# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from api_manager.dtos.base_dto import BaseDTO
from api_manager.dtos.training_params import DigitalTwinTrainingParams


@dataclass
class TrainRequest(BaseDTO):
    model_id: str
    model_update: bool = False
    params: DigitalTwinTrainingParams = field(default_factory=dict)

    # TODO: create base class with generic form of this method in place
    @classmethod
    def from_dict(cls, obj: Dict) -> TrainRequest:
        obj["params"] = DigitalTwinTrainingParams(**obj["params"])
        return cls(**obj)
