# Mobility Robustness Optimization Notebook

## Configure Notebook enviroment

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

### Sample data set

Unpack the sample data set present at `notebooks/data/mro_data.zip` under `notebooks/data/`

## Start Jupyter

Run Jupyter notebook:

    ```console
    jupyter notebook
    ```

This will open Jupyter notebooks in browser. Under Files tab in Jupyter notebooks navigate to `notebooks/mro.ipynb` and run the notebook.

## LICENSE

See [license](LICENSE)
