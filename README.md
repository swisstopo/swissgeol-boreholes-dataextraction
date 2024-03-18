# Project Name (Boreholes Extraction?)

Intro to be written. This just serves as a boilerplate file. Documentation will likely be written on the go.

## How to run the code
We use conda to create and manage our virtual environments. Two environments are specified in `environment-dev.yml` and `environment-prod.yml` respectively. The prod environment contains all neccessary dependencies to run the code and extraction pipelines therein. All dependencies that are useful for the development of the code, but not to run it are separated into the dev environment.

TODO: some instructions on how to run the code.
Run the script located at `src/stratigraphy/main.py` to run the extraction pipeline. This will source all pdfs from `data/Benchmark` and create png files in `data/Benchmark/extract`. 

### Folder structure
In order to run the code you have to place the pdf files that are supposed to be analyzed at `data/Benchmark`. 


## Experiment Tracking
We perform experiment tracking using MLFlow. Each developer has his own local MLFlow instance. 

In order to use mlflow, you will need to place a `.env` file at the project root. The required environment variables specified in the `.env` are:

```
MLFLOW_TRACKING="True"
MLFLOW_TRACKING_URI="http://localhost:5000"
```

If the `MLFLOW_TRACKING` variable is not set, no experiments will be logged.

In order to view your experiment, start the mlflow server using `mlflow ui` in your terminal.

Extensive documentation about MLFlow can be found [here](https://mlflow.org/docs/latest/index.html).

## Pre-Commit
We use pre-commit hooks to format our code in a unified way.

Pre-commit comes in the boreholes-dev conda environment. After activating the conda environment you have to install pre-commit by running `pre-commit install` in your terminal. You only have to do this once.

After installing pre-commit, it will trigger 'hooks' upon each `git commit -m ...` command. The hooks will be applied on all the files in the commit. A hook is nothing but a script specified in `.pre-commit-config.yaml`.

The installed hooks are:
- black
- isort.

If you want to skip the hooks, you can use `git commit -m "" --no-verify`.

More information about pre-commit can be found [here](https://pre-commit.com).
