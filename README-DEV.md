# RADP - Development Workflow

Please follow this workflow when making code changes to the system.

## Development Workflow

* Pull the latest version of the __"main"__ branch

* Install the development dependencies

    ```console
    python3 -m venv .venv
    source .venv/bin/activate
    pip3 install --upgrade pip
    pip3 install -r requirements-dev.txt
    ```

* Install all regular dependencies, for unit testing

    ```console
    pip3 install -r radp/client/requirements.txt
    pip3 install -r radp/common/requirements.txt
    pip3 install -r radp/digital_twin/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
    pip3 install -r radp/utility/requirements.txt
    pip3 install -r services/requirements-dev.txt
    pip3 install -r services/api_manager/requirements.txt
    pip3 install -r services/orchestration/requirements.txt
    pip3 install -r services/rf_prediction/requirements.txt
    pip3 install -r services/training/requirements.txt
    pip3 install -r services/ue_tracks_generation/requirements.txt
    ```

* And environment variables if desired:

    ```console
    copy .env-dev .env
    ```

* Checkout to a new development branch
* Make your code changes
* Add/update unit tests for your changes

* Set Python PATH to root of project

    ```console
    export PYTHONPATH="$(pwd)":$PYTHONPATH
    ```

* Run unit tests

    ```console
    pytest
    ```

* Run component tests

    ```console
    python3 tests/run_component_tests.py
    ```

* Start up the RADP service locally

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

    Then you can start RADP in development mode:

    ```console
    docker compose -f dc.yml -f dc-dev.yml up -d --build
    ```

    Or with Nvidia GPU support too:

    ```console
    docker compose -f dc.yml -f dc-dev.yml -f dc-cuda.yml up -d --build
    ```

    __KNOWN ISSUE__

    Docker Desktop for Mac is known to have issues in its new use of VirtioFS for its mount system.
    There are still open issues on GitHub of file truncation and corruption, especially triggered by a rapid sequence of writes.
    This occasionally manifests in crashes for RADP as well, mostly when run under a debugger, which affects its timing.
    If you experience an exception running RADP on DD for Mac, try to switch from VirtioFS back to gRPC FUSE, which is slower but stable.
    Hopefully DD for Linux and Windows do not manifest the same issues.

* Run end-to-end tests

    ```console
    python3 tests/run_end_to_end_tests.py
    ```

* Run pre-commit workflow

    ```console
    pre-commit install
    python3 -m pre_commit run --all-files
    ```

* Commit changes to your local branch, ensure commit name is 3-10 words and summarizes the changes
* Create a Pull Request to merge your changes into the __"main"__ with the following:
  * Clear and concise description of the changes made
  * Testing procedure - how did you test that your code didn't break anything?
  * Any relevant execution logs or screenshots
* Submit the Pull Request

## Testing the Github Workflow

* Test the Github Workflow locally using `act` library

    act depends on docker to run workflows. Install both and test the jobs locally
    1. `brew install act` on MacOS, or follow these [instructions](https://github.com/nektos/act#installation-through-package-managers) for other OS
    2. `act -j end-to-end-tests` and `act -j unit-tests`

## Monitoring Kafka Messages

### dev

```bash
docker exec -u 0 -it radp_dev-kafka-1 bash

kafka-console-consumer --bootstrap-server host.docker.internal:9095 --topic jobs
kafka-console-consumer --bootstrap-server host.docker.internal:9095 --topic outputs
```

```bash
docker exec -u 0 -it radp_dev-kafka-1 bash -c "kafka-console-consumer --bootstrap-server host.docker.internal:9095 --topic jobs"
docker exec -u 0 -it radp_dev-kafka-1 bash -c "kafka-console-consumer --bootstrap-server host.docker.internal:9095 --topic outputs"
```

### prod

```bash
docker exec -u 0 -it radp_prod-kafka-1 bash

kafka-console-consumer --bootstrap-server host.docker.internal:9094 --topic jobs
kafka-console-consumer --bootstrap-server host.docker.internal:9094 --topic outputs
```

```bash
docker exec -u 0 -it radp_dev-kafka-1 bash -c "kafka-console-consumer --bootstrap-server host.docker.internal:9094 --topic jobs"
docker exec -u 0 -it radp_dev-kafka-1 bash -c "kafka-console-consumer --bootstrap-server host.docker.internal:9094 --topic outputs"
```

## LICENSE

See [license](LICENSE)
