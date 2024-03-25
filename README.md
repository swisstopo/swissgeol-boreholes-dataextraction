# Boreholes Data Extraction

Boreholes Data Extraction is a data extraction pipeline that extracts depth layers with their corresponding material description from borehole profiles in form of pdfs.

Note that the project is under active development and there is no release to this date, nor has the project reached a maturity such that it could be used. The current extractions are solely focused on material descriptions. In the future, the material descriptions should be matched with their corresponding depth layers.

## Installing
We use conda to create and manage the project's dependencies. The project comes with two environments, `environment-dev.yml` and `environment-prod.yml`, respectively. The prod environment contains all neccessary dependencies to run the code and extraction pipelines therein. All dependencies that are useful for the development of the code, but not to run it are separated into the dev environment.

Assuming you have conda installed and cloned the repository, run the following command in your project repository:
```bash
conda env create -f environment-prod.yml
```

If you would like to get all developer functionalities, run:

```bash
conda env create -f environment-dev.yml
```


## Run Data Extraction
This documentation provides a step-by-step guide on how to execute the pipeline, from activating the conda environment to checking the results. To execute the data extraction pipeline, follow these steps:

1. **Activate the Conda Environment**

   If you haven't already, activate the conda environment using the following command:

   ```bash
   conda activate boreholes-prod
   ````

    If you are developing and testing the code, you might want to use the dev environment instead:

    `conda activate boreholes-dev`

2. **Run the Main Script**

    The main script for the extraction pipeline is located at src/stratigraphy/main.py. Run this script to start the extraction process:

    This script will source all PDFs from the data/Benchmark directory and create PNG files in the data/Benchmark/extract directory.

3. **Check the Results**

    Once the script has finished running, you can check the results in the `data/Benchmark/extract` directory. The result is a `predictions.json` file as well as a png file for each pdf in the `data/Benchmark` directory.

Please note that for now the pipeline assumes that all PDF files to be analyzed are placed in the `data/Benchmark` directory. If you want to analyze different files, please place them in this directory.

### Output Structure
The predictions.json file contains the results of a data extraction process from PDF files. Each key in the JSON object is the name of a PDF file, and the value is a list of extracted items in a dictionary like object. The extracted items for now are the material descriptions in their correct order (given by their depths).

Example: predictions.json 
```json
{
    "685256002-bp.pdf": {
        "layers": [
            {
                "description": "grauer, siltig-sandiger Kies (Auffullung)"
            },
            {
                "description": "grauer, lehmiger Kies (Auffullung)"
            },
            {
                "description": "grauer, lehmig-sandiger Kies (Auffullung)"
            },
            {
                "description": "grauer, sandig-lehmiger Kies (Auffullung)"
            },
            {
                "description": "grauer, lehmig-sandiger Kies (Auffullung)"
            }
        ]
    }
}
```

# Developer Guidance
## Project Structure

The project structured and the most important files are as follows:

- `root/` : The root directory of the project.
  - `src/` : The source code of the project.
    - `stratigraphy/` : The main package of the project.
      - `main.py` : The main script of the project. This script runs the data extraction pipeline.
      - `line_detection.py`: This script runs the line detection on provided sample pdfs. Will be deprecated in the future.
      - `util/` : Utility scripts and modules.
      - `benchmark/` : Scripts to evaluate the data extraction.
  - `data/` : The data used by the project.
    - `Benchmark/` : The directory containing the PDF files to be analyzed.
      - `extract/` : The directory where the PNG files are saved.
          - `predictions.json` : The output file of the project, containing the results of the data extraction process.
  - `tests/` : The tests for the project.
  - `README.md` : The README file for the project.


## Main Scripts

- `main.py` : This is the main script of the project. It runs the data extraction pipeline, which analyzes the PDF files in the `data/Benchmark` directory and saves the results in the `predictions.json` file.

- `line_detection.py` : Runs the line detection algorithm on pdfs using `lsd` from opencv. It is meant to find all lines that potentially separate two material descriptions. It is incorporated in the script `main.py` and will be deprecated as a standalone script in the future.

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

We use [ruffs](https://github.com/astral-sh/ruff) [pre-commit package](https://github.com/astral-sh/ruff-pre-commit) for linting and formatting.
The specific linting and formatting settings applied are defined in `pyproject.toml`.

If you want to skip the hooks, you can use `git commit -m "..." --no-verify`.

More information about pre-commit can be found [here](https://pre-commit.com).
