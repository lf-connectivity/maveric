# RIC Algorithm Development Platform (RADP)

The Maveric project is the source of the RIC Algorithm Development Platform (RADP). It enables the development and evaluation of dynamic control policies that optimize cellular network operation, before their deployment on the network. It is intended to serve the use-case of developing solutions that leverage the Radio Intelligent Controller (RIC) paradigm. It is a developer platform that leverages AI/ML approaches to provide realistic cellular network representations (known as digital twins), as well as examples of simple xApp/rApp algorithms, that demonstrate its use.

## Table of Contents

- [RIC Algorithm Development Platform (RADP)](#ric-algorithm-development-platform-radp)
  - [Table of Contents](#table-of-contents)
  - [Development Workflow](#development-workflow)
  - [Getting Started](#getting-started)
    - [Downloading the repo](#downloading-the-repo)
    - [Installing and running Docker](#installing-and-running-docker)
    - [Booting up RADP](#booting-up-radp)
    - [Installing Python client dependencies](#installing-python-client-dependencies)
    - [Running an example simulation](#running-an-example-simulation)
    - [Stopping the RADP System](#stopping-the-radp-system)
  - [Train API](#train-api)
    - [Training Params](#training-params)
    - [UE Training Data File](#ue-training-data-file)
    - [Topology File](#topology-file)
  - [Simulation API](#simulation-api)
    - [Simulation Event Object](#simulation-event-object)
    - [Cell Config File](#cell-config-file)
    - [Simulation API Output](#simulation-api-output)
  - [Describe Model API](#describe-model-api)
  - [Describe Simulation API](#describe-simulation-api)
  - [Consume Simulation Output API](#consume-simulation-output-api)
  - [UE Tracks](#ue-tracks)
    - [Passing in UE Tracks via UE Data File](#passing-in-ue-tracks-via-ue-data-file)
    - [Generating UE Tracks](#generating-ue-tracks)
    - [UEClassDistribution](#ueclassdistribution)
    - [UEClassParams](#ueclassparams)
    - [LatLonBoundaries](#latlonboundaries)
    - [GaussianMarkovParams](#gaussianmarkovparams)
  - [RF Prediction](#rf-prediction)
    - [Input to RF Prediction](#input-to-rf-prediction)
    - [Output from RSRP](#output-from-rsrp)
  - [RADP Service Limitations](#radp-service-limitations)
    - [Job Lengths](#job-lengths)
  - [Something wrong?](#something-wrong)
  - [Running Notebooks](#running-notebooks)

## Development Workflow

Please follow [the development workflow](README-DEV.md) here when making code changes.

## Getting Started

### Downloading the repo

If you haven't already, download or clone this repo on the machine where you'd like to run RADP.

### Installing and running Docker

To run the RADP service you will need to install [Docker](https://docs.docker.com/get-docker/).

Once Docker is installed make sure to start the docker daemon.

### Booting up RADP

Once Docker is running you just need to cd to the "radp" directory (if you're not already there) and run the following command

```console
docker build -t radp radp
```

[Alternative] Using GPUs from host

Docker has native support for this.

Host must have Nvidia driver, Nvidia Container toolkit installed.

RADP Docker Compose files are already set to attach Nvidia GPUs if available.

You will need to specify a different base image for Docker Compose:

```console
docker build -f radp/Dockerfile-cuda -t radp radp
```

Then you can start RADP in production mode:

```console
docker compose -f dc.yml -f dc-prod.yml up -d --build
```

Or with Nvidia GPU support too:

```console
docker compose -f dc.yml -f dc-prod.yml -f dc-cuda.yml up -d --build
```

This will do the following:

1. Build the docker images using their local Docker files
2. Calls docker run to build containers for each service
3. Creates services, volumes, kafka topics, and other components of RADP architecture

The first time you run this command it may take a while since Docker will need to download container images. You'll see a lot of text appear on your terminal.

Once you can see the RADP API service expose a web address you'll know you're ready to call the system's APIs:

```text
radp-init               | Successfully created the following topics:
radp-init               | jobs
radp-init               | outputs
radp-init exited with code 0
```

### Installing Python client dependencies

RADP currently requires Python >= 3.8.x and < 3.11.x

The last remaining holdout to support Python 3.11 appears to be the Torch library, which should be resolved [soon](https://github.com/pytorch/pytorch/issues/86566).

CD to the top level of the repo. Run the following command to install Python client dependencies:

```console
python3 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install -r radp/client/requirements.txt
```

### Running an example simulation

Install example requirements:

```console
pip3 install -r apps/requirements.txt
```

You can copy the sample Environment files and make changes to them if desired:

```console
copy .env-prod .env
```

This package comes with an example script which you can call to see the RADP service in action:

```console
python3 apps/example/example_app.py
```

If the simulation runs end-to-end successfully, you should see an outputted dataframe containing the example simulation output.

Likewise, a basic Coverage and Capacity Optimization (CCO) example can be run as follows:

```console
python3 apps/coverage_capacity_optimization/cco_example_app.py
```

The app takes about 2 minutes to run and produce final results on the console.

### Stopping the RADP System

To stop the services run:

```console
docker compose -f dc.yml -f dc-prod.yml down
```

in a terminal tab.

If you encounter an error here or hit `ctrl + c` you may kill your containers, possibly leaving them or their volumes in an unhealthy state. To clean your Docker containers and volumes, just run the following commands.

1. `docker container prune` to remove stopped containers
2. `docker volume prune` to remove unused volumes

__IMPORTANT__: Running volume prune will completely remove your RADP system data so only run during development

---

## Train API

The Train API allows a user to train a new RF Digital Twin model.

```python
    radp_client.train(
        model_id=model_id,
        params={},
        ue_training_data="ue_training_data.csv",
        topology="topology.csv",
    )
```

- __model\_id (str, required)__ - the model_id which will be used to reference this model in subsequent calls to the Simulation API.
- __params (dict, required)__ - the training parameters to pass in for training, see [Training Params](#training-params)
- __ue\_training\_data (str or pandas dataframe object, required)__ - the file path of the [UE Training Data File](#ue-training-data-file) (or a training data pandas dataframe)
- __topology (str or pandas dataframe object, required)__ - the file path of the cell [Topology File](#topology-file) (or a cell topology pandas dataframe), required to enable the model to preprocess the UE training data and create per-cell engineered features for training

See the example_app.py script for an example where pandas dataframes are provided directly to the client.

### Training Params

The following object shows all training params that can be specified.

```python
{
    "maxiter": 100
    "lr": 0.05
    "stopping_threshold": 0.0001
}
```

- __maxiter (int, optional)__ - the max number of iterations to run in training
- __lr (float, optional)__ - model learning rate
- __stopping\_threshold (float, optional)__ - stopping threshold for training

### UE Training Data File

The UE training data file must be a CSV file with the following format:

|cell_id|avg_rsrp|lon|lat|cell_el_deg|
|---|---|---|---|---|
|cell_1|-80|139.699058|35.644327|0|
|cell_1|-70|139.707889|35.647814|3|
|cell_1|-75|139.700024|35.643857|6|
|...|...|...|...|...|

- __cell\_id (str, required)__ - the ID of the cell unit, the RF Digital Twin will run training/inference on a per-cell unit basis
- __avg\_rsrp (float, required)__ - the RSRP value for this cell, for the lon-lat pixel of data, this is the target value for supervised learning employed by the RF Digital Twin model
- __lon (float, required)__ - the longitude of this pixel of data
- __lat (float, required)__ - the latitude of this pixel of data
- __cell\_el\_deg (float, required)__ - the electrical antenna tilt of the cell

For an example of a UE training data file, see apps/example/ue_training_data.csv

### Topology File

The topology file must be a CSV file with the following format:

|cell_id|cell_name|cell_lat|cell_lon|cell_az_deg|cell_carrier_freq_mhz|
|---|---|---|---|---|---|
|cell_1|Cell1|35.690556|139.691944|0|2100|
|cell_2|Cell2|35.690556|139.691944|120|2100|
|cell_3|Cell3|35.690556|139.691944|240|2100|
|...|...|...|...|...|...|

- __cell\_id (str)__ - the cell identfier
- __cell\_name (str)__ - the name of the cell (optional)
- __cell\_lat (float)__ - the cell latitude
- __cell\_lon (str)__ - the cell longitute
- __cell\_az\_deg (int)__ - the cell azimuth degree
- __cell\_carrier\_freq\_mhz (int)__ - the cell carrier frequency layer

For an example of a topology file, see apps/example/topology.csv

---

## Simulation API

The Simulation API allows a user to run a RIC Simulation event. If called with a valid request, the simulation API will create a simulation event which can be described by calling the [Describe Simulation API](#consume-simulation-output-api). Once the simulation has finished, the outputted results can be consumed by calling the [Consume Simulation Output API](#consume-simulation-output-api)

```python
radp_client.simulation(
    simulation_event=simulation_event,
    ue_data=prediction_data,
    config=prediction_config,
)
```

- __simulation\_event__ - the simulation event object
- __ue\_data__ - the file path of the UE Data file, see [Passing in UE Tracks via UE Data File](#passing-in-ue-tracks-via-ue-data-file); this can also instead by a pandas dataframe
- __config__ - the file path of the Cell Config file, see

### Simulation Event Object

```python
{
    "simulation_time_interval_seconds": 0.01,
    "ue_tracks": <UE Tracks object>,
    "rf_prediction": <RF Prediction Object>,
}
```

- __simulation\_time\_interval\_seconds (float)__ - the simulation time interval in seconds; the time between "ticks" within the simulation - the above example would be 10ms between each tick
- __ue\_tracks__ - the UE Tracks component to the simulation, see [UE Tracks](#ue-tracks)
- __rf\_prediction__ - the RF Prediction component to the simulation, see [RF Prediction](#rf-prediction)

### Cell Config File

|cell_id|cell_el_deg|
|---|---|
|cell_1|10|
|cell_2|10|
|cell_3|12|

- __cell\_id (str, required)__ - the cell identifier
- __cell\_el\_deg (float, required)__ - the electrical antenna tilt of the cell

### Simulation API Output

```python
{
    "simulation_id": 'c88e347c613d1727a1c6ce5b61bc0877'
}
```

- __simulation\_id__ - the simulation identifier used to describe or consume output from this simulation

---

## Describe Model API

The Describe Model API allows a user to describe the training status of a digital twin model. This allows the user to see if a model has finished training (and thus is ready to be used in simulation).

```python
radp_client.describe_model(model_id)
```

- __model\_id__ - the identifier of the model to describe

TO BE COMPLETED

---

## Describe Simulation API

The Describe Simulation API allows a user to describe an existing simulation. The response will signal to the user whether the simulation has finished (and thus whether output results can be consumed).

```python
radp_client.describe_simulation(simulation_id)
```

- __simulation\_id__ - the identifier of the simulation to describe

TO BE COMPLETED

---

## Consume Simulation Output API

The Consume Simulation Output API allows a user to consume the output data from a simulation. The API returns a pandas dataframe object pertaining to simulation output

Example:

```python
rf_dataframe = radp_client.consume_simulation_output(simulation_id)
```

- __simulation\_id (str, required)__ - the ID for the simulation

---

## UE Tracks

UE track data must be either passed in or generated to run a RIC simulation. The `ue_tracks` key must be present in a SimulationEvent object.

### Passing in UE Tracks via UE Data File

Users have the option of passing in their own UE track data to the simulation. To specify that you will pass UE Track data to a simulation you must include the `ue_data_id` key in the `ue_tracks` like the following:

```python
"ue_tracks": {
    "ue_data_id": "ue_data_1"
},
```

The value provided will be used to identify this UE data passed in. It does not need to be the name of the file.

__IMPORTANT__: If two sequential calls to the simulation API contain the same value for this field, the RADP system will assume the provided UE tracks data is the same. The system caches data using a hash of this value. If you wish to pass and run a RIC simulation on a different data set you MUST change the value provided here.

Along with adding the ```ue_data_id``` key to the simulation object, the user must also pass the file path of the UE data to the RADP Client. See the example.py script for an example API call with UE data file path passed in. The UE data file must be a comma-separated csv file with the following case-sensitive columns. See ue_data.csv for an example.

|mock_ue_id|lon|lat|tick|cell_id|
|---|---|---|---|---|
|1|139.699058|35.644327|0|1|
|2|139.707889|35.647814|0|1|
|3|139.700024|35.643857|0|1|
|1|139.699061|35.644322|1|2|
|2|139.707899|35.647813|1|2|
|...|...|...|...|...|

- __lon (float)__ - the longitude of the pixel
- __lat (float)__ - the latitude of the pixel
- __mock_ue_id (str, optional)__ - a fake user equipment ID to associate with this pixel of data
- __tick (int, optional)__ - the relative index of this position with respect to time; if a a certain mock_ue_id has a position (x1,y1) at tick=1, then has position (x2, y2) at tick=2, the simulation sees that this UE has moved from (x1,y1) to (x2,y2) across an interval of time equal to simulation_time_interval_seconds
- __cell\_id (str, optional)__ - the identifier of the cell to attach this data point to. If this column is present then RF prediction will only occur from _this pixel to this cell_. If this column is not provided then RF prediction will occur on a per-cell basis where RX power is outputted for this pixel _for every cell_ in the topology

For an example of a UE Tracks data file, see apps/example/ue_data.csv

### Generating UE Tracks

Users also have the option of generating UE tracks data using RADP's built-in UE Tracks Generation service. The service employs a gaussian markov-based mobility model to generate UE tracks data.

To include UE Tracks Generation in a RIC simulation, set up your UE tracks object as follows:

```python
"ue_tracks": {
    "ue_tracks_generation" : {
        "ue_class_distribution": UEClassDistribution,
        "lat_lon_boundaries": LatLonBoundaries,
        "gauss_markov_params": GaussianMarkovParams,
    }
},
```

### UEClassDistribution

This object specifies the distribution of UE classes as well as mobility parameters for each class. The following UE classes are supported (case-sensitive):
    - stationary
    - pedestrian
    - cyclist
    - car

To include a UE class in the generated UE tracks data, include the class as a key in the UETypes object, with a value of UEType. See below example which includes only pedestrians and cars:

```python
"ue_class_distribution": {
    "pedestrian": UEClassParams,
    "car": UEClassParams,
},
```

### UEClassParams

The basic parameters of the UE Class

- __count (int)__ - the number of UEs of this class to generate in the simulation
- __velocity (float)__ - the average velocity of this UE class (float in meters/second)
- __velocity\_variance (float)__ - the average variance from the average velocity of this UE class (float)

An example UEClassDistribution with UEClassParams for pedestrian and car:

```python
"ue_class_distribution": {
    "pedestrian": {
        "count": 5,
        "velocity": 1,
        "velocity_variance": 1
    },
    "car": {
        "count": 5,
        "velocity": 10,
        "velocity_variance": 0.5
    }
},
```

### LatLonBoundaries

The latitude and longitude boundaries of the grid where the UE Tracks data will be generated. UE device tracks will not stray outside of these boundaries. See the following example:

```python
"lat_lon_boundaries": {
    "min_lat": -90,
    "max_lat": 90,
    "min_lon": -180,
    "max_lon": 180
},
```

### GaussianMarkovParams

This object specifies the Gauss-Markov parameters to tune the randomness of the generated UEs. The params that can be specified are:

- __alpha (float)__ - the tuning parameter used to vary the randomness. Totally random values are obtained by setting it to 0 and linear motion is obtained by setting it to 1
- __variance (float)__ - the randomness variance for the UEs
- __rng\_seed (int)__ - the random generator seed to be used to generate the UE data. The seed is set manually for reproducibility

Below is an example of UE mobility generated using the UE Tracks Generation service.

![Trajectory of 3 UEs for 25 ticks](docs/media/ue_data_gen.png?raw=true")

---

## RF Prediction

As a component of a RIC Simulation, users can specify a previously trained RF Digital Twin model to run RF Prediction.

### Input to RF Prediction

The input to RF prediction is a data set pertaining to UE Tracks, thus to run RF Prediction UE Tracks must first be specified in the SimulationEvent object. The only field required to run RF Prediction is the ID of the trained Digital Twin model to use in RF Prediction.

```python
"rf_prediction": {
    "model_id": "my_trained_digital_twin_1"
}
```

### Output from RSRP

As output, the RF Prediction step will add predicted per-cell RX Power data to the UE Tracks data.

|cell_id|rxpower_dbm|mock_ue_id|lon|lat|tick|
|---|---|---|---|---|---|
|cell_1|-73.58406342082407|0|-22.647772018547954|59.7818913169958|0|
|cell_1|-73.58406342082407|1|119.78045195037765|54.86740569575272|0|
|cell_1|-73.58406342082407|2|72.1170723329875|-20.22066321601217|0|
|cell_1|-73.58406342082407|0|-23.17652796675165|59.49811644458839|1|
|cell_1|-73.58406342082407|1|121.23118177561184|55.20488171003581|1|
|cell_1|-73.58406342082407|2|70.39142709656107|-21.780123291101106|1|
|cell_1|-73.58406342082407|0|-27.928272856142712|60.77581594854604|2|
|cell_1|-73.58406342082407|1|126.61098550003226|58.21946030130971|2|
|cell_1|-73.58406342082407|2|69.80139485065422|-21.980885049368638|2|

- __cell\_id (str)__ - the ID of the cell for which this RF prediction is made
- __rxpower\_dbm (float)__ - the predicted RSRP value (in dBm) for this cell, for the lat-lon position specified

## RADP Service Limitations

This section lays out the known limitations of the service.

### Job Lengths

Below are the timeout thresholds for training and simulation jobs

- __training__ - 12 hours
- __simulation__ - 15 minutes for each stage in a simulation event

To explain, a training job can take up to 12 hours after which it will timeout. A simulation event can handle as much data as is possible without passing a 15 minute length of time for any single stage in the simulation pipeline. The developer team plans to implement batch processing, after which this will no longer be a limitation.

## Something wrong?

For all of these API's you should be able to see relevant output in your terminal tab in which you ran `docker compose`. If you see error outputs for any command then something is wrong!

## Running Notebooks

Please follow [this guide](README-NOTEBOOKS.md) for running notebooks.

## LICENSE

See [license](LICENSE)
