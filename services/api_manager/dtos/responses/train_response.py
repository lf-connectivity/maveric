# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass

from api_manager.dtos.base_dto import BaseDTO


@dataclass
class TrainResponse(BaseDTO):
    job_id: str
    model_id: str
