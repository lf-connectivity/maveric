# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


class DataSource(Enum):
    USER_INPUT = "user_inputted"
    CACHE = "cache"


class InputFileType(Enum):
    UE_TRAINING_DATA = "ue_training_data"
    UE_DATA = "ue_data"
    CONFIG = "config"
    TOPOLOGY = "topology"


class ModelStatus(Enum):
    TRAINED = "trained"
    IN_TRAINING = "in_training"


class ModelType(Enum):
    RF_DIGITAL_TWIN = "rf_digital_twin"
    # TODO: add mobility model here once we allow for training mobility models
    # MOBILITY = 'mobility'


class OutputStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"


class SimulationStage(Enum):
    START = "start"
    UE_TRACKS_GENERATION = "ue_tracks_generation"
    RF_PREDICTION = "rf_prediction"
    PROTOCOL_EMULATION = "protocol_emulation"
    FINISH = "finish"


class WorkflowStatus(Enum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    FINISHED = "finished"
    FAILED = "failed"
