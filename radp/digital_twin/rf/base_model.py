# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum


class DTModelType(Enum):
    SVGP = "svgp"
    BAYESIAN = "bayesian"


class DTModel(ABC):
    @property
    @abstractmethod
    def model_type(self) -> str:
        pass

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        pass
