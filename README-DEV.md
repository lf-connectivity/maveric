# RADP - Development Workflow

Please follow this workflow when making code changes to the system.

## Development Workflow

- Pull the latest version of the **"main"** branch

- Install the development dependencies

  > **Note:** `setuptools` must be upgraded before other packages to avoid `ModuleNotFoundError: No module named 'pkg_resources'`. If you are using Python 3.11, `fastkml` version 0.13 or higher is required — version 0.12 is not compatible.
  >
  > **Note:** Upgrading `fastkml` to 0.13+ changes its API. The geometry for a `Placemark` must now be passed to the constructor rather than assigned as a property after creation. If you encounter `AttributeError: property 'geometry' of 'Placemark' object has no setter`, update `radp_library.py` accordingly, or pin `fastkml==0.12` in `requirements-dev.txt` if you do not need Python 3.11 support.

  ```console
  python3 -m venv .venv
  source .venv/bin/activate
  pip3 install --upgrade pip setuptools
  pip3 install "fastkml>=0.13"
  pip3 install -r requirements-dev.txt
  ```

- Install all regular dependencies, for unit testing

  > **Note:** Confirm that `radp/__init__.py` exists in the repository before running tests. If it is missing, create an empty file at that path so Python recognises `radp/` as a package. The `requests` and `retry` packages are required by the RADP client — install them if the import step below fails.

  ```console
  pip3 install requests retry
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

  > **Troubleshooting:** If you see `FATAL: Could not import RADP client modules. Ensure project root is in PYTHONPATH`, verify that the project root is in `sys.path` (see the `PYTHONPATH` step below) and that `requests` and `retry` are installed.

- And environment variables if desired:

  ```console
  copy .env-dev .env
  ```

  > **FFmpeg path:** The `.env` file contains a placeholder for the FFmpeg binary path. Replace it with the actual path on your system. To find it, run:
  >
  > ```bash
  > which ffmpeg
  > ```
  >
  > Copy the output (e.g. `/usr/local/bin/ffmpeg`) into the `.env` file. Leaving the placeholder as-is will cause a `FileNotFoundError` when saving animations.

- Checkout to a new development branch
- Make your code changes
- Add/update unit tests for your changes

- Set Python PATH to root of project

  ```console
  export PYTHONPATH="$(pwd)":$PYTHONPATH
  ```

- Run unit tests

  ```console
  pytest
  ```

- Run component tests

  ```console
  python3 tests/run_component_tests.py
  ```

- Start up the RADP service locally

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

  **KNOWN ISSUE**

  Docker Desktop for Mac is known to have issues in its new use of VirtioFS for its mount system.
  There are still open issues on GitHub of file truncation and corruption, especially triggered by a rapid sequence of writes.
  This occasionally manifests in crashes for RADP as well, mostly when run under a debugger, which affects its timing.
  If you experience an exception running RADP on DD for Mac, try to switch from VirtioFS back to gRPC FUSE, which is slower but stable.
  Hopefully DD for Linux and Windows do not manifest the same issues.

- Run end-to-end tests

  ```console
  python3 tests/run_end_to_end_tests.py
  ```

- Run pre-commit workflow

  ```console
  pre-commit install
  python3 -m pre_commit run --all-files
  ```

- Commit changes to your local branch, ensure commit name is 3-10 words and summarizes the changes
- Create a Pull Request to merge your changes into the **"main"** with the following:
  - Clear and concise description of the changes made
  - Testing procedure - how did you test that your code didn't break anything?
  - Any relevant execution logs or screenshots
- Submit the Pull Request

## Testing the Github Workflow

- Test the Github Workflow locally using `act` library

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
