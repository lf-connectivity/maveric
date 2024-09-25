# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# HTTP
REQUEST_PAYLOAD_FILE_KEY = "payload"
REQUEST_UE_DATA_FILE_KEY = "ue_data.csv"
REQUEST_UE_TRAINING_DATA_FILE_KEY = "ue_training_data.csv"
REQUEST_CONFIG_FILE_KEY = "config.csv"
REQUEST_TOPOLOGY_FILE_KEY = "topology.csv"

# Model Related
MODEL_ID = "model_id"
MODEL_FILE_NAME = "model"
MODEL_METADATA_FILE_NAME = "metadata"
MODEL_FILE_EXTENSION = "pickle"
MODEL_FILE_PATH = "model_file_path"
MODEL_METADATA_FILE_EXTENSION = "json"
MODELS_FOLDER = "/srv/radp/models"
TOPOLOGY_FILE_NAME = "topology"
MODEL_TYPE = "model_type"
MODEL_UPDATE = "model_update"

# Simulation Related
SIMULATION_DATA_FOLDER = "/srv/radp/simulation_data"
SIMULATION_OUTPUTS_FOLDER = "/srv/radp/simulation_data/outputs"
DF_FILE_EXTENSION = "fea"
SIM_OUTPUT_FILE_SUFFIX = "output"
SIM_OUTPUT_FILE_EXTENSION = "zip"
SIM_OUTPUT_DIRECTORY = "output"
USER_FACING_DF_EXTENSION = "csv"
SIM_METADATA_FILE_NAME = "metadata"
SIM_METADATA_FILE_EXTENSION = "json"
STATUS_PLANNED = "planned"
UE_TRACKS = "ue_tracks"
PREDECESSOR_STAGE_HASH_VAL = "predecessor_stage_hash_val"
UE_TRACKS_GENERATION_HASH_VAL = "ue_tracks_generation_hash_val"
RF_PREDICTION_HASH_VAL = "rf_prediction_hash_val"
PROTOCOL_EMULATION_HASH_VAL = "protocol_emulation_hash_val"
UE_DATA_FILE_NAME = "ue_data"
CONFIG_FILE_NAME = "config"

# Kafka related
KAFKA_JOBS_TOPIC_NAME = "jobs"
KAFKA_OUTPUTS_TOPIC_NAME = "outputs"
KAFKA_JOB_TYPE = "job_type"
JOB_ID = "job_id"
JOB_FINISHED_DATETIME = "job_finished_datetime"
JOB_TYPE_ORCHESTRATION = "orchestration"
JOB_TYPE_RF_PREDICTION = "rf_prediction"
JOB_TYPE_UE_TRACKS_GENERATION = "ue_tracks_generation"
KAFKA_CONSUMER_POLL_INTERVAL = 10

# Simulation metadata related
PARAMS = "params"
SIMULATION_DURATION = "simulation_duration_seconds"
SIMULATION_TIME_INTERVAL = "simulation_time_interval_seconds"
NUM_TICKS = "num_ticks"
NUM_BATCHES = "num_batches"
OUTPUT_FILE_PATH = "output_file_path"
MODEL_FILE = "model_file"
HASH_VAL = "hash_val"
JOB_TYPE = "job_type"
BATCH = "batch"
STATE = "state"
BATCHES_OUTPUTTED = "batches_outputted"
LATEST_BATCH_WITHOUT_FAILURE = "latest_batch_without_failure"
LATEST_BATCH_TO_SUCCEED = "latest_batch_to_succeed"
BATCHES_RETRYING = "batches_retrying"
STATUS = "status"
SERVICE = "service"
OUTPUT_FILE_PREFIX = "output_file_prefix"
SIMULATION_ID = "simulation_id"
SIMULATION_STATUS = "simulation_status"

# RF Digital Twin related
UE_DATA_FILE_PATH_KEY = "ue_data_file_path"
UE_TRAINING_DATA_FILE_PATH_KEY = "ue_training_data_file_path"
CONFIG_FILE_PATH_KEY = "config_file_path"
TOPOLOGY_FILE_PATH_KEY = "topology_file_path"
NUM_CELLS = "num_cells"

# Training related
JOB_TYPE_TRAINING = "training"
TRAINING_PARAMS = "training_params"

# Flask API related
INPUT_FILES_FOLDER = "/srv/radp/input_files"

# RF Prediction related
CELL_ID = "cell_id"
RXPOWER_DBM = "rxpower_dbm"
MOCK_UE_ID = "mock_ue_id"
LONGITUDE = "lon"
LATITUDE = "lat"
TICK = "tick"
RF_PREDICTION = "rf_prediction"
OUTPUTS = "outputs"

# UE Tracks Generation
UE_TRACKS_GENERATION = "ue_tracks_generation"
UE_CLASS_DISTRIBUTION = "ue_class_distribution"
COUNT = "count"
STATIONARY = "stationary"
PEDESTRIAN = "pedestrian"
CYCLIST = "cyclist"
CAR = "car"
VELOCITY = "velocity"
VELOCITY_VARIANCE = "velocity_variance"
LON_LAT_BOUNDARIES = "lat_lon_boundaries"
MIN_LAT = "min_lat"
MIN_LON = "min_lon"
MAX_LAT = "max_lat"
MAX_LON = "max_lon"
ALPHA = "alpha"
VARIANCE = "variance"
RNG_SEED = "rng_seed"
LON_X_DIMS = "lon_x_dims"
LON_Y_DIMS = "lon_y_dims"
UE_TRACK_GENERATION_OUTPUTS_FOLDER = (
    "/srv/radp/simulation_data/outputs/ue_tracks_generation"
)
GAUSS_MARKOV_PARAMS = "gauss_markov_params"

# Protocol Emulation related
PROTOCOL_EMULATION = "protocol_emulation"
