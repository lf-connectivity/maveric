# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass
class BaseDTO:
    def to_dict(self) -> Dict:
        return asdict(self)
