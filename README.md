# Boreholes Data Extraction

Boreholes Data Extraction is a data extraction pipeline that extracts depth layers with their corresponding material description from borehole profiles in form of pdfs.

## Limitations

Note that the project is under active development and there is no release to this date, nor has the project reached a maturity such that it could be used.

The current extractions are solely focused on the depths of the upper and lower limits of each layer, and on the material descriptions of the layers.

The current benchmarking only considers the correctness of the material descriptions. Whether they are linked with the correct depths, is not yet evaluated.

## Installation
We use conda to create and manage the project's dependencies. The project comes with two environments, `environment-dev.yml` and `environment-prod.yml`, respectively. The prod environment contains all necessary dependencies to run the code and extraction pipelines therein. All dependencies that are useful for the development of the code, but not to run it, are separated into the dev environment.

Assuming you have conda installed and cloned the repository, run the following command in your project repository:
```bash
conda env create -f environment-prod.yml
```

If you would like to get all developer functionalities, run:

```bash
conda env create -f environment-dev.yml
```


## Run data extraction
To execute the data extraction pipeline, follow these steps:

1. **Activate the Conda environment**

   If you haven't already, activate the conda environment using the following command:

   ```bash
   conda activate boreholes-prod
   ````

    If you are developing and testing the code, you might want to use the dev environment instead:

    `conda activate boreholes-dev`

2. **Run the extraction script**

    The main script for the extraction pipeline is located at `src/stratigraphy/main.py`. A cli command is created to run this script.

    Run `boreholes-extract-materials` to run the main extraction script. With the default options, the command will source all PDFs from the `data/Benchmark` directory and create PNG files in the `data/Benchmark/extract` directory.

    Use `boreholes-extract-materials --help` to see all options for the extraction script.

3. **Check the results**

    Once the script has finished running, you can check the results in the `data/Benchmark/extract` directory. The result is a `predictions.json` file as well as a png file for each page of each PDF in the `data/Benchmark` directory.

Please note that for now the pipeline assumes that all PDF files to be analyzed are placed in the `data/Benchmark` directory. If you want to analyze different files, please place them in this directory.

### Output Structure
The `predictions.json` file contains the results of a data extraction process from PDF files. Each key in the JSON object is the name of a PDF file, and the value is a list of extracted items in a dictionary like object. The extracted items for now are the material descriptions in their correct order (given by their depths).

Example: predictions.json 
```json
{
    "685256002-bp.pdf": {  # file name
        "page_1": {
            "layers": [  # a layer corresponds to a material layer in the borehole profile
                {
                    "material_description": {  # all information about the complete description of the material of the layer
                        "text": "grauer, siltig-sandiger Kies (Auffullung)",
                        "rect": [
                            232.78799438476562,
                            130.18496704101562,
                            525.6640014648438,
                            153.54295349121094
                        ],
                        "lines": [
                                    {
                                        "text": "grauer, siltig-sandiger Kies (Auffullung)",
                                        "rect": [
                                            232.78799438476562,
                                            130.18496704101562,
                                            525.6640014648438,
                                            153.54295349121094
                                        ]
                                    }
                                ]
                            },
                    "depth_interval": {  # information about the depth of the layer
                        "start": null,
                        "end": {
                            "value": 0.4,
                            "rect": [
                                125.25399780273438,
                                140.2349853515625,
                                146.10398864746094,
                                160.84498596191406
                            ]
                        }
                    }
                },
                ...
            ],
            "depths_materials_column_pairs": [  # information about where on the pdf the information for material description as well as depths are taken.
                {
                    "depth_column": {
                        "rect": [
                            119.05999755859375,
                            140.2349853515625,
                            146.8470001220703,
                            1014.4009399414062
                        ],
                        "entries": [
                            {
                                "value": 0.4,
                                "rect": [
                                    125.25399780273438,
                                    140.2349853515625,
                                    146.10398864746094,
                                    160.84498596191406
                                ]
                            },
                            {
                                "value": 0.6,
                                "rect": [
                                    125.21800231933594,
                                    153.8349609375,
                                    146.0679931640625,
                                    174.44496154785156
                                ]
                            }
                        ]
                    },
                    "material_description_rect": [
                        231.22500610351562,
                        130.18496704101562,
                        540.6109619140625,
                        897.7429809570312
                    ]
                }
            ]
        }
    }
}
```

# Developer Guidance
## Project Structure

The project structure and the most important files are as follows:

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


## Main scripts

- `main.py` : This is the main script of the project. It runs the data extraction pipeline, which analyzes the PDF files in the `data/Benchmark` directory and saves the results in the `predictions.json` file.

- `line_detection.py` : Runs the line detection algorithm on pdfs using `lsd` from opencv. It is meant to find all lines that potentially separate two material descriptions. It is incorporated in the script `main.py` and will be deprecated as a standalone script in the future.

## Experiment Tracking
We perform experiment tracking using MLFlow. Each developer has his own local MLFlow instance. 

In order to use MLFlow, you will need to place a `.env` file at the project root. The required environment variables specified in the `.env` are:

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

## Package Manager
We use [miniforge](https://github.com/conda-forge/miniforge) as a package manager for the project. Miniforge imitates the behavior of conda, with the difference that no packages from the anaconda repository are used. The only channel we use is conda-forge which does not come with license implications.
