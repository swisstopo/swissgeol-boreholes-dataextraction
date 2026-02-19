# Developer Guidance
This README provides an overview of the project structure, main scripts, experiment tracking, development tooling, and the two installable Python packages (`swissgeol_doc_processing` and `extraction`) that power the borehole data extraction and classification pipelines.

## Project Structure

The project structure and the most important files are as follows:

- `root/` : The root directory of the project.
  - `src/` : The source code of the project.
    - `app/`: The API of the project.
      - `main.py`: The main script that launches the API.
      - `common/config.py`: Config file for the API.
      - `v1/`: Contains all the code for the version 1 of the API.
      - `v1/router.py`: Provides an overview of all available endpoints.
    - `extraction/`: The main package of the data extraction pipeline.
      - `annotations/`: Package for drawing the documents.
      - `evaluation/`: Package for evaluating the extracted information.
        - `benchmark/score.py`: Script to score predictions without running the extraction.
      - `features/`: Package containing all the extraction logic.
        - `groundwater/`: Contains the groundwater extraction logic.
        - `metadata/`: Contains the elevation and coordinates extraction logic.
        - `predictions/`: Mainly contains data structures and prediction matching logic.
        - `stratigraphy/`: Contains the stratigraphy extraction logic.
        - `utils/` : Utility package related to extraction.
        - `extract.py`: Contains the main logic for data extraction
      - `main.py` : The main script to run the data extraction pipeline.
    - `classification/`: The main package for the classification of layer descriptions.
      - `classifiers/`: Contains implementations of various classifiers.
      - `evaluation/`: Contains scripts for evaluating classification results.
      - `utils/`: Contains modules used for processing data for classification and all the existing classes.
      - `models/`: Contains modules used by the Bert-base classifier, like the model and the training pipeline.
      - `main.py` : The main script to run the descritption classification pipeline.
    - `utils/` : Utility modules that are used by all pipelines.
    - `scripts/`: Various utility scipts used to download the files or generate the ground truth.
  - `data/` : The data used by the project.
    - `output/` : output of the extraction pipeline
      - `draw/` : The directory where the PNG files are saved.
      - `predictions.json` : The output file of the project, containing the results of the data extraction process.
    - `output_description_classification/`: output of the classification pipeline
  - `config/`: Contains configuration files for the classification pipeline.
    - `baseline/`: folder containing config files for using the baseline classifier.
    - `bert/` : folder containing config files for fine-tuning and infering using the BERT model.
    - `bedrock/`: folder containing config files for using the aws bedrock classifier.
    - `classification_params.yml`: Configuration file containing general variables used for the classification pipeline.
    - `classifier_config_paths.yml`: file containing the paths to all config files, for each classifier and classification system.
    - `line_detection_params.yml`: Configuration file for the line detection.
    - `matching_params.yml`: Configuration file containing variables used for the data extraction pipeline.
  - `tests/` : The tests for the project.
  - `README.md` : The README file for the project.
  - `pyproject.toml`: Contains the Python requirements and configurations specific to the Python environment.
  - `Dockerfile`: Dockerfile to launch the Borehole App as API.


## Main scripts

- `extraction/main.py` : This is the main script of the project. It runs the data extraction pipeline, which analyzes the PDF files in the `data` directory and saves the results in the `predictions.json` file.



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

## Packages

This repository contains two installable Python packages that can be used independently or together:

- **`swissgeol_doc_processing`**: Core document processing utilities (geometry, text extraction, language detection)
- **`extraction`**: Borehole-specific extraction pipeline

Both packages are automatically included when installing the main `swissgeol-boreholes-dataextraction` package.

### Installation

Both packages are part of the `swissgeol-boreholes-dataextraction` repository and can be installed directly from GitHub. They are built using setuptools and setuptools-scm for version management. Published [versions](https://github.com/swisstopo/swissgeol-boreholes-dataextraction/releases) can be installed via:

```bash
pip install https://github.com/swisstopo/swissgeol-boreholes-dataextraction/releases/download/v{VERSION}/swissgeol-boreholes-dataextraction-{VERSION}-py3-none-any.whl
```

The package configuration is defined in `pyproject.toml` under `[tool.setuptools.packages.find]`, which specifies both `swissgeol_doc_processing` and `extraction` as package sources.

---

### swissgeol_doc_processing

A standalone Python library that provides core document processing functionality for geological documents.

#### Modules

- **`geometry`**: Geometric analysis tools including circle detection, line detection, and spatial data structures
- **`text`**: Text extraction and processing utilities for handling PDF text blocks, text lines, and linguistic analysis (stemming, language detection)
- **`utils`**: General utilities for data extraction, file handling, language filtering, and document structure detection (tables, strip logs)

#### Usage

```python
from swissgeol_doc_processing.geometry import line_detection
from swissgeol_doc_processing.text import extract_text
from swissgeol_doc_processing.utils import language_detection

language_detection.detect_language_of_text(
    "This is a sample text.",
    "english",
    ["english", "french", "german"]
)
```

---

### extraction

A Python library for extracting structured data from borehole PDF documents. It provides both CLI commands for running the full extraction pipeline and programmatic APIs for feature extraction suitable for machine learning classification.

#### Modules

- **`features`**: Core extraction logic
  - `metadata`: Borehole metadata extraction (coordinates, elevation, names)
  - `stratigraphy`: Layer and depth extraction
  - `groundwater`: Groundwater level extraction
  - `predictions`: Data structures for extraction results
  - `extract`: Main extraction functions
- **`evaluation`**: Evaluation and benchmarking tools
- **`annotations`**: Visualization and drawing utilities
- **`minimal_pipeline`**: Simplified extraction for ML feature generation
- **`utils`**: Extraction-specific utilities (dynamic matching)

#### Usage

```python
import pymupdf
from extraction.minimal_pipeline import extract_page_features, load_default_params
from extraction.features.metadata.metadata import FileMetadata

params = load_default_params()

with pymupdf.Document("borehole.pdf") as doc:
    file_metadata = FileMetadata.from_document(doc, params["matching_params"])

    for page_index, page in enumerate(doc):
        features = extract_page_features(
            page, page_index, file_metadata.language, **params
        )
```