# RADP Notebooks

## Configure Notebook environment

Install the Notebook dependencies:

    ```console
    python3 -m venv .venv
    source .venv/bin/activate
    pip3 install --upgrade pip
    pip3 install -r notebooks/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
    ```

Optional: Enable Jupyter Notebook extensions:

Code folding: `jupyter nbextension enable codefolding/main`

You can find more details about other extensions here:
 <https://jupyter-contrib-nbextensions.readthedocs.io/>

Install ffmpeg:

You can download it from here <https://ffmpeg.org/download.html>

Provide the file path to ffmpeg binary in .env file.

Create .env file in root folder of repo and add below to env

> FFMPEG_PATH="/path_to_ffmpeg/ffmpeg"

## Sample data set

Unpack the sample data set present at `notebooks/data/sim_data.zip` under `notebooks/data/`

## Start Jupyter

Run Jupyter notebook:

    ```console
    jupyter notebook
    ```

This will open Jupyter notebooks in browser. Under Files tab in Jupyter notebooks navigate to `notebooks/coo_demo_with_radp_digital_twin.ipynb` and run the test.

## LICENSE

See [license](LICENSE)
