# Boreholes Data Extraction

Boreholes Data Extraction is a pipeline to extract structured data from borehole profiles in PDF files. Extracted properties (currently coordinates, depths and associated material descriptions) are returned in JSON format, and (optionally) visualized as PNG images. This project was initiated by the Swiss Federal Office of Topography [swisstopo](https://www.swisstopo.admin.ch/), and is developed with support from [Visium](https://www.visium.ch/).

## Motivation

The Federal Office of Topography swisstopo is Switzerland's geoinformation centre. The Swiss Geological Survey at swisstopo is the federal competence centre for the collection, analysis, storage, and provision of geological data of national interest. 

Data from boreholes is an essential source for our knowledge about the subsurface. In order to manage and publish borehole data of national interest, swisstopo has developed the application boreholes.swissgeol.ch (currently for internal use only), part of the [swissgeol.ch](https://swissgeol.ch) platform. As of August 2024, over 30.000 boreholes are registered in the application database, a number that is rapidly increasing thanks to an improved data exchange with cantonal offices, other government agencies and federal corporations such as the Swiss Federal Railways SBB. In the coming years, the number of boreholes in the database is expected to keep increasing to a multiple of the current size. Data is being added from both boreholes that were recently constructed and documented, as well as from older boreholes that were until now only documented in separate databases or in analogue archives. Data from older boreholes can still be very relevant, as geology only changes very slowly, and newer data is often unavailable (and expensive to collect).

In order to use the collected borehole data efficiently, it is critical that both metadata as well as geological information is digitally stored in a structured database. However, the relevant data for most boreholes that are received by swisstopo, is contained in PDF-files that lack a standardized structure. Older data is often only available in the form of a scanned image, obtained from a printed document or from a microfiche. Manually entering all the relevant data from these various sources into a structured database is not feasible, given the large amount of boreholes and the continuous influx of new data.

Therefore, the goal of this project is to automate the extraction of structured data from borehole profiles as much as possible. As far as swisstopo is concerned, the use case is to integrate the data extraction pipeline with the application boreholes.swissgeol.ch ([GitHub](https://github.com/swisstopo/swissgeol-boreholes-suite)), where a user interface for efficient quality control of the automatically extracted data will also be implemented.

All code and documentation is published in this GitHub repository as open source software. All other persons, companies or agencies who manage borehole data of their own, are welcome to use the data extraction pipeline on their own data and to contribute to the project with their own improvements/additions.

### Extracted properties

Below is a list of the most relevant properties for the extraction of structure data from borehole profiles, using the following styles:

- Properties that can already be automatically extracted by the current pipeline implementation are in **bold**.
- Properties for which the implementation of automatic extraction is actively being worked on, are in _italics_.
- Properties that are likely to be added to the data extraction pipeline in the future, but are not under active development yet, are in regular font.

#### Most relevant borehole properties

* Metadata
  * **Coordinates**
  * _Date_
  * _Drilling method_
* Lithology / stratigraphy
  * **Depths** (upper and lower bound of each layer)
  * **Material descriptions** (as plain text)
  * USCS classification, color, consistency, plasticity...
  * Geological interpretations
* Other
  * _Hydrogeology (ground water levels)_
  * Instrumentation
  * Casing
  * Borehole geometry
  * ...


### Related work

Existing work related to this project is mostly focussed on the extraction and classification of specific properties from textual geological descriptions. Notable examples include [GEOBERTje](https://www.arxiv.org/abs/2407.10991) (Belgium), [geo-ner-model](https://github.com/BritishGeologicalSurvey/geo-ner-model) (UK), [GeoVec](https://www.sciencedirect.com/science/article/pii/S0098300419306533) und [dh2loop](https://github.com/Loop3D/dh2loop) (Australia). The scope of this project is considerable wider, in particular regarding the goal of understanding borehole profiles in various languages and with an unknown layout, where the document structure first needs to be understood, before the relevant text fragments can be identified and extracted.

The automatic data extraction pipeline can be considered to belong to the field of [automatic/intelligent document processing](https://en.wikipedia.org/wiki/Document_processing). As such, it involves a combination of methods from multiple fields in data science and machine learning, in particular computer vision (e.g. object detection, line detection) and natural language processing (large language models, named entity recognition). Some of these have already been implemented (e.g. the [Line Segment Detector](https://docs.opencv.org/3.4/db/d73/classcv_1_1LineSegmentDetector.html) algorithm), others are planned as future work.

### Limitations

The project is under active development and there is no release to this date. The quality/accuracy of the results may vary strongly depending on the documents that are used as input.

The input PDF files must contain digital text content. For PDF files that are not _digitally-born_ (e.g. scanned documents), this means that OCR must be performed, and the OCR results stored in the PDF file, before using the file as an input for this data extraction pipeline. The quality of the extracted data is dependent on the quality of the OCR. At swisstopo, we use the [AWS Textract](https://aws.amazon.com/textract/) service together with our own code from the [swissgeol-ocr](https://github.com/swisstopo/swissgeol-ocr) repository for this purpose.

The pipeline has been optimized for and tested on boreholes profiles from Switzerland that have been written in German or (to a more limited extent) in French.

With regard to the extraction of coordinates, the [Swiss coordinate systems](https://de.wikipedia.org/wiki/Schweizer_Landeskoordinaten) LV95 as well as the older LV03 are supported ([visualization of the differences](https://opendata.swiss/de/dataset/bezugsrahmenwechsel-lv03-lv95-koordinatenanderung-lv03-lv95)).

## Main contributors

* Stijn Vermeeren [@stijnvermeeren-swisstopo](https://www.github.com/stijnvermeeren-swisstopo) (swisstopo) - Project Lead
* David Cleres [@dcleres](https://www.github.com/dcleres) (Visium)
* Renato Durrer [@redur](https://www.github.com/redur) (Visium)

## Installation
We use pip to manage the packages dependencies. We recommend using a virtual environment within which to install all dependencies.

The below commands will install the package for you (assuming you have successfully cloned the repository):
```bash
python -m venv env
source env/bin/activate
pip install -e '.[all]'
```

Alternatively you can replace the `pip install -e '.[all]'` command with `pip install git+https://github.com/swisstopo/swissgeol-boreholes-dataextraction.git` in production scenarios.

Adding pip packages can be done by editing the `pyproject.toml` of the project and adding the required package.

## Run data extraction
To execute the data extraction pipeline, follow these steps:

1. **Activate the virtual environment**

    Activate your virtual environment. On unix systems this is

    ``` bash
    source env/bin/activate
    ```

2. **Download the borehole profiles, optional**

    Use `boreholes-download-profiles` to download the files to be processed from an AWS S3 storage. In order to do so, you need to authenticate with aws first. We recommend to use the aws CLI for that purpose. This step is optional, you can continue with step 3 on your own set of borehole profiles.

3. **Run the extraction script**

    The main script for the extraction pipeline is located at `src/stratigraphy/main.py`. A cli command is created to run this script.

    Run `boreholes-extract-all` to run the main extraction script. With the default options, the command will source all PDFs from the `data/Benchmark` directory and create PNG files in the `data/Benchmark/extract` directory.

    Use `boreholes-extract-all --help` to see all options for the extraction script.

4. **Check the results**

    Once the script has finished running, you can check the results in the `data/Benchmark/extract` directory. The result is a `predictions.json` file as well as a png file for each page of each PDF in the `data/Benchmark` directory.

### Output Structure
The `predictions.json` file contains the results of a data extraction process from PDF files. Each key in the JSON object is the name of a PDF file, and the value is a list of extracted items in a dictionary like object. The extracted items for now are the material descriptions in their correct order (given by their depths).

Example: predictions.json
```json
{
    "685256002-bp.pdf": {  # file name
        "language": "de",
        "metadata": {
            "coordinates": null
        },
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
                            ],
                            "page": 1
                        }
                    ],
                    "page": 1
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
                        ],
                        "page": 1
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
                            ],
                            "page": 1
                        },
                        {
                            "value": 0.6,
                            "rect": [
                                125.21800231933594,
                                153.8349609375,
                                146.0679931640625,
                                174.44496154785156
                            ],
                            "page": 1
                        },
                        ...
                    ]
                }
            }
        ],
        "page_dimensions": [
            {
                "height": 1192.0999755859375,
                "width": 842.1500244140625
            }
        ]
    },
}
```

# Developer Guidance
## Project Structure

The project structure and the most important files are as follows:

- `root/` : The root directory of the project.
  - `src/` : The source code of the project.
    - `stratigraphy/` : The main package of the project.
      - `main.py` : The main script of the project. This script runs the data extraction pipeline.
      - `line_detection.py`: Contains functionalities for line detection on pdf pages.
      - `util/` : Utility scripts and modules.
      - `benchmark/` : Scripts to evaluate the data extraction.
  - `data/` : The data used by the project.
    - `output/` : 
      - `draw/` : The directory where the PNG files are saved.
      - `predictions.json` : The output file of the project, containing the results of the data extraction process.
  - `tests/` : The tests for the project.
  - `README.md` : The README file for the project.


## Main scripts

- `main.py` : This is the main script of the project. It runs the data extraction pipeline, which analyzes the PDF files in the `data/Benchmark` directory and saves the results in the `predictions.json` file.


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