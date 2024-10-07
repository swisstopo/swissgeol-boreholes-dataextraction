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

#### Project Status
The project is under active development and there is no release to this date. The quality/accuracy of the results may vary strongly depending on the documents that are used as input.

#### Requirements on the input PDFs
The input PDF files must contain digital text content. For PDF files that are not _digitally-born_ (e.g. scanned documents), this means that OCR must be performed, and the OCR results stored in the PDF file, before using the file as an input for this data extraction pipeline. The quality of the extracted data is dependent on the quality of the OCR. At swisstopo, we use the [AWS Textract](https://aws.amazon.com/textract/) service together with our own code from the [swissgeol-ocr](https://github.com/swisstopo/swissgeol-ocr) repository for this purpose.

#### Test Regions and Languages
The pipeline has been optimized for and tested on boreholes profiles from Switzerland that have been written in German or (to a more limited extent) in French.

#### Coordinates
With regard to the extraction of coordinates, the [Swiss coordinate systems](https://de.wikipedia.org/wiki/Schweizer_Landeskoordinaten) LV95 as well as the older LV03 are supported ([visualization of the differences](https://opendata.swiss/de/dataset/bezugsrahmenwechsel-lv03-lv95-koordinatenanderung-lv03-lv95)).

#### Groundwater
With the current version of the code, groundwater can only be found at depth smaller than 200 meters. This threshold is defined in `src/stratigraphy/groundwater/groundwater_extraction.py` by the constant `MAX_DEPTH`. 

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

    Run `boreholes-extract-all` to run the main extraction script. You need to specify the input directory or a single PDF file using the `-i` or `--input-directory` flag. 
 The script will source all PDFs from the specified directory and create PNG files in the `data/output/draw` directory.

    Use `boreholes-extract-all --help` to see all options for the extraction script.

4. **Check the results**

    Once the script has finished running, you can check the results in the `data/output/draw` directory. The result is a `predictions.json` file as well as a png file for each page of each PDF in the specified input directory.

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
    - `app/`: The API of the project.
      - `main.py`: The main script that launches the API.
      - `common/config.py`: Config file for the API.
      - `v1/`: Contain all the code for the version 1 of the API.
      - `v1/router.py`: Presents at a glance all the available endpoints.
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
  - `pyproject.toml`: Contain the python requirements and provides specific for a python environment.
  - `Dockerfile`: Dockerfile to launch the Borehole App as API.


## Main scripts

- `main.py` : This is the main script of the project. It runs the data extraction pipeline, which analyzes the PDF files in the `data/Benchmark` directory and saves the results in the `predictions.json` file.

## API

The API for this project is built using FastAPI, a modern, fast (high-performance), web framework for building APIs with Python.

To launch the API and access its endpoints, follow these steps:

1. **Activate the virtual environment**

    Activate your virtual environment. On Unix systems, this can be done with the following command:

    ```bash
    source env/bin/activate
    ```

2. **Environment variables**

    Please make sure to define the environment variables needed for the API to access the S3 Bucket of interest.

    ```python
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_key_access = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_session_token = os.environ.get("AWS_SESSION_TOKEN")
    aws_endpoint = os.environ.get("AWS_ENDPOINT")
    ```

3. **Start the FastAPI server**

    Run the following command to start the FastAPI server:

    ```bash
    uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8002
    ```

    This will start the server on port 8002 of the localhost and enable automatic reloading whenever changes are made to the code. You can see the OpenAPI Specification (formerly Swagger Specification) by opening: `http://127.0.0.1:8002/docs#/` in your favorite browser. 

4. **Access the API endpoints**

    Once the server is running, you can access the API endpoints using a web browser or an API testing tool like Postman.

    The main endpoint for the data extraction pipeline is `http://localhost:8000/extract-data`. You can send a POST request to this endpoint with the PDF file you want to extract data from.

    Additional endpoints and their functionalities can be found in the project's source code.

    **Note:** Make sure to replace `localhost` with the appropriate hostname or IP address if you are running the server on a remote machine.

5. **Stop the server**

    To stop the FastAPI server, press `Ctrl + C` in the terminal where the server is running. Please refer to the [FastAPI documentation](https://fastapi.tiangolo.com) for more information on how to work with FastAPI and build APIs using this framework.


## Build API as Local Docker Image

The borehole application offers a given amount of functionalities (extract text, number, and coordinates) through an API. To build this API using a Docker Container, you can run the following commands. 

1. **Navigate to the project directory**

    Change your current directory to the project directory:

    ```bash
    cd swissgeol-boreholes-dataextraction
    ```

2. **Build the Docker image**

    Build the Docker image using the following command:

    ```bash
    docker build -t borehole-api . -f Dockerfile
    ```

    ```bash
    docker build --platform linux/amd64 -t borehole-api:test .
    ```

    This command will build the Docker image with the tag `borehole-api`.

3. **Verify the Docker image**

    Verify that the Docker image has been successfully built by running the following command:

    ```bash
    docker images
    ```

    You should see the `borehole-api` image listed in the output.

4. **Run the Docker container**

    To run the Docker container, use the following command:

    ```bash
    docker run -p 8000:8000 borehole-api
    ```

    This command will start the container and map port 8000 of the container to port 8000 of the host machine.

5. **Run the docker image with the AWS credentials**

If you have the AWS credentials configured locally in the `~/.aws` file, you can run the following command to forward your AWS credentials to the docker container 

    ```bash

    docker run -v ~/.aws:/root/.aws -d -p 8000:8000 borehole-api
    ```

    ```bash 
    docker run --platform linux/amd64 -v ~/.aws:/root/.aws -d -p 8000:8000 borehole-api:test
    ```


6. **Access the API**

    Once the container is running, you can access the API by opening a web browser and navigating to `http://localhost:8000`.

    You can also use an API testing tool like Postman to send requests to the API endpoints.

    **Note:** If you are running Docker on a remote machine, replace `localhost` with the appropriate hostname or IP address.


7. **Query the API**

```bash
    curl -X 'POST' \
    'http://localhost:8000/api/V1/create_pngs' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "filename": "10021.pdf"
    }'
```

8. **Stop the Docker container**

    To stop the Docker container, press `Ctrl + C` in the terminal where the container is running.

    Alternatively, you can use the following command to stop the container:

    ```bash
    docker stop <container_id>
    ```

    Replace `<container_id>` with the ID of the running container, which can be obtained by running `docker ps`.


## Use the Docker Image from the GitHub Container Registry

This repository provides a Docker image hosted in the GitHub Container Registry (GHCR) that can be used to run the application easily. Below are the steps to pull and run the Docker image.

1. Pull the Docker Image

```bash
docker pull ghcr.io/dcleres/swissgeol-boreholes-dataextraction-api:edge
```

2. a. Run the docker image from the Terminal

```bash
docker run -d --name swissgeol-boreholes-dataextraction-api -e AWS_ACCESS_KEY_ID=XXX -e AWS_SECRET_ACCESS_KEY=YYY -e AWS_ENDPOINT=ZZZ -p 8080:8080 ghcr.io/dcleres/swissgeol-boreholes-dataextraction-api:TAG
```

Adjust the port mapping (8080:8080) based on the app's requirements.

NOTE: Do not forget to specify your AWS Credentials.

1. b. Run the docker image from the Docker Desktop App

Open the Docker Desktop app and navigate to `Images`, you should be able to see the image you just pulled from GHCR. Click on the image and click on the `Run` button on the top right of the screen. 

![](assets/img/docker-1.png){ width=400px }

Then open the `Optional Settings` menu and specify the port and the AWS credentials

![](assets/img/docker-2.png){ width=800px }


1.  Verify the Container is Running
   
To check if the container is running, use:

```bash
docker ps
```

## AWS Lambda Deployment

AWS Lambda is a serverless computing service provided by Amazon Web Services that allows you to run code without managing servers. It automatically scales your applications by executing code in response to triggers. You only pay for the compute time used.

In this project we are using Mangum to wrap the FastAPI with a handler that we will package and deploy as a Lambda function in AWS. Then using AWS API Gateway we will route all incoming requests to invoke the lambda and handle the routing internally within our application.

We created a script that should make it possible for you to deploy the FastAPI in AWS lambda using a single command. The script is creating all the required AWS resources to run the API. The resources that will be created for you are: 
- AWS Lambda Function
- AWS IAM user with the right to execute lambda functions and to read & write on S3 buckets
- AWS CloudWatch log group to monitor the API
- AWS API Gateway

To deploy the staging version of the FastPI, run the following command: 

```shell
IMAGE=borehole-fastapi ENV=stage AWS_PROFILE=dcleres-visium AWS_S3_BUCKET=dcleres-boreholes-integration-tmp ./deploy_api_aws_lambda.sh
```


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