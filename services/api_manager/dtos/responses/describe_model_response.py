# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict, dataclass
from typing import Dict, Optional

from api_manager.dtos.base_dto import BaseDTO


@dataclass
class DescribeModelResponse(BaseDTO):
    model_id: str
    type: str
    status: str

    # TODO: implement RFDigitalTwin and Mobility dataclass objects to store info
    # returned in a describe response for these models
    rf_digital_twin: Optional[Dict]
    mobility: Optional[Dict]

    def to_dict(self) -> Dict:
        dict_repr = asdict(self)
        if not dict_repr["rf_digital_twin"]:
            dict_repr.pop("rf_digital_twin")
        if not dict_repr["mobility"]:
            dict_repr.pop("mobility")
        return dict_repr
