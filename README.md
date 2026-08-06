# Boreholes Data Extraction

Boreholes Data Extraction is a pipeline to extract structured data from borehole profiles in PDF files. Extracted properties (currently coordinates, depths and associated material descriptions) are returned in JSON format, and (optionally) visualized as PNG images. This project was initiated by the Swiss Federal Office of Topography [swisstopo](https://www.swisstopo.admin.ch/), and is developed with support from [Visium](https://www.visium.com/).

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

Existing work related to this project is mostly focussed on the extraction and classification of specific properties from textual geological descriptions. Notable examples include [GEOBERTje](https://www.arxiv.org/abs/2407.10991) (Belgium), [geo-ner-model](https://github.com/BritishGeologicalSurvey/geo-ner-model) (UK), [GeoVec](https://www.sciencedirect.com/science/article/pii/S0098300419306533) und [dh2loop](https://github.com/Loop3D/dh2loop) (Australia). The scope of this project is considerable wider, in particular regarding the goal of understanding borehole profiles in various languages and with an unknown layout, where the document structure first needs to be understood, before the relevant text fragments can be identified and extracted. A commercial solution with similar goals is [Civils.ai](https://civils.ai/geotechnical-engineering-ai-automation) ([Youtube video](https://www.youtube.com/watch?v=WkAttZWbjBk)).

The automatic data extraction pipeline can be considered to belong to the field of [automatic/intelligent document processing](https://en.wikipedia.org/wiki/Document_processing). As such, it involves a combination of methods from multiple fields in data science and machine learning, in particular computer vision (e.g. object detection, line detection) and natural language processing (large language models, named entity recognition). Some of these have already been implemented (e.g. the [Line Segment Detector](https://docs.opencv.org/3.4/db/d73/classcv_1_1LineSegmentDetector.html) algorithm), others are planned as future work.

### Limitations

#### Project Status
The project is under active development and there is no release to this date. The quality/accuracy of the results may vary strongly depending on the documents that are used as input.

#### Requirements on the input PDFs
The input PDF files must contain digital text content. For PDF files that are not _digitally-born_ (e.g. scanned documents), this means that OCR must be performed, and the OCR results stored in the PDF file, before using the file as an input for this data extraction pipeline. The quality of the extracted data is dependent on the quality of the OCR. At swisstopo, we use the [AWS Textract](https://aws.amazon.com/textract/) service together with our own code from the [swissgeol-ocr](https://github.com/swisstopo/swissgeol-ocr) repository for this purpose.

We also provide the script `src/scripts/deskew_pdf.py` that can unskew document that have been rotated or warped during their scanning. This script is applied to all documents in a given folder, and will produce a copy of those documents, unskewed. It is important to note that any ocr done before this step will be lost, and that it must be redone after the deskewing.

#### Test Regions and Languages
The pipeline has been optimized for and tested on boreholes profiles from Switzerland that have been written in German or (to a more limited extent) in French.

#### Coordinates
With regard to the extraction of coordinates, the [Swiss coordinate systems](https://de.wikipedia.org/wiki/Schweizer_Landeskoordinaten) LV95 as well as the older LV03 are supported ([visualization of the differences](https://opendata.swiss/de/dataset/bezugsrahmenwechsel-lv03-lv95-koordinatenanderung-lv03-lv95)).

#### Groundwater
With the current version of the code, groundwater can only be found at depth smaller than 200 meters. This threshold is defined in `src/extraction/features/groundwater/groundwater_extraction.py` by the constant `MAX_DEPTH`.

The groundwater is extracted in two main ways from the borehole documents. The first one aims to match a groundwater-related keyword in the text extracted from the document (e.g., groundwater, groundwater-level). The second technique focuses on extracting the groundwater-related symbol from the document. It aims at finding 2 horizontal lines, on top of each other, satisfying some visual requirement that the groundwater symbol has.


## Project management

This project is managed and financed by the Swiss Federal Office of Topography [swisstopo](https://www.swisstopo.admin.ch/). Many contributions come from [Visium](https://www.visium.com/), in their role as contractor for swisstopo for this project.

This project is released as open source software, under the principle of "_public money, public code_", in accordance with the 2023 federal law "[_EMBAG_](https://www.fedlex.admin.ch/eli/fga/2023/787/de)", and following the guidance of the [tools for OSS published by the Federal Chancellery](https://www.bk.admin.ch/bk/en/home/digitale-transformation-ikt-lenkung/bundesarchitektur/open_source_software/hilfsmittel_oss.html).

We welcome feedback, bug reports and code contributions (provided they are compatible with swisstopo's roadmap) from third parties. Feature requests and support requests can only be fulfilled as long as they are compatible with swisstopo's legal mandate. Other organisations (both within Switzerland or internationally) who manage their own borehole data and are interested in using and/or contributing to this project, are encouraged to contact us, in order to discuss the possibility of establishing a partnership for collaborating more closely on this project.

### Main contributors

* Stijn Vermeeren [@stijnvermeeren-swisstopo](https://www.github.com/stijnvermeeren-swisstopo) (swisstopo) - Project Lead
* [Visium](https://www.visium.com/)

See [contributers](https://github.com/swisstopo/swissgeol-boreholes-dataextraction/graphs/contributors) for all individual contributors.

### License

The source code is licensed under the [AGPL-3.0-only License](LICENSE). This is due to the licensing of certain dependencies, most notably [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright), which is only available under either the AGPL license or a commercial license. If this dependency is removed in the future, we will switch to a more permissive license for this project.

## Installation
We use [uv](https://docs.astral.sh/uv/) to manage package dependencies. Install uv first if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

The below commands will install the package for you (assuming you have successfully cloned the repository):
```bash
uv sync --all-extras
```

Then activate your environment:
```bash
source .venv/bin/activate
```

Alternatively, you can install directly from GitHub in production scenarios:
```bash
uv pip install git+https://github.com/swisstopo/swissgeol-boreholes-dataextraction.git
```

Adding packages can be done by editing the `pyproject.toml` of the project and adding the required package, then running `uv lock` to update the lock file.

## Run data extraction
To execute the data extraction pipeline, follow these steps:

### 1. Download the borehole profiles, optional

Use `boreholes-download-profiles` to download the files to be processed from an AWS S3 storage. In order to do so, you need to authenticate with aws first. We recommend using the aws CLI for that purpose, or storing your credentials in the ~/.aws configuration files This step is optional, you can continue with step 3 on your own set of borehole profiles.

Alternatively, you can download the data directly using the AWS CLI:
```bash
brew install awscli
aws s3 sync s3://stijnvermeeren-boreholes-data ./data
```

If you choose to use the ~/.aws files, then they should look like this:

**~/.aws/config**

  ```
  [default]
  region=eu-central-1
  output=json
  ```

 **~/.aws/credentials**

  ```
  [default]
  aws_access_key_id=YOUR_ACCESS_KEY
  aws_secret_access_key=YOUR_SECRET_KEY
  ```

## 2. Run the extraction script

The main script for the extraction pipeline is located at `src/extraction/main.py`. A cli command is created to run this script.

Run `boreholes-extract-all` to run the main extraction script. You need to specify the input directory or a single PDF file using the `-i` or `--input-directory` flag.
The script will source all PDFs from the specified directory and create PNG files in the `data/output/draw` directory.

```bash
boreholes-extract-metadata -i data/pdfs/
```

Run`boreholes-extract-metadata` to run only the metadata part of the pipeline (coordinates, elevation, borehole names), skipping stratigraphy extraction.
Accepts the same options as `boreholes-extract-all`.



Use `boreholes-extract-all --help` to see all options for the extraction script.

To run the extraction pipeline on multiple datasets in one command and to receive an overview of all child runs on parent level, use the `--benchmark` flag on `boreholes-extract-all`.
Use repeatable benchmark specs with syntax `"<name>:<input_path>:<ground_truth_path>"`, e.g.
```bash
boreholes-extract-all \
  --benchmark "zurich:data/zurich:data/zurich_ground_truth.json" \
  --benchmark "bern:data/bern:data/bern_ground_truth.json"
```

To apply custom settings, generate a local copy of the configuration files using the package helper. Any values you change locally will override the package defaults.

```python
from swissgeol_doc_processing.utils.file_utils import expose_configs

# Create a local "config/" folder populated with the default settings
expose_configs()
```

This will create a `config/` directory at the root of your project containing all configuration files that can be safely edited.

### 3. Check the results

The script produces output in different formats:
- A file `data/output/predictions.json` that contains all extracted data in a machine-readable format. The structure of this file is documented in [README.predictions-json.md](docs/README.predictions-json.md).
- More concise output files `data/output/predictions_coordinates.json`, `data/output/predictions_elevation.json` and `data/output/predictions_name.json` with the extracted values (optionally evaluated against ground truth values) for individual borehole metadata properties (coordinates, elevation and borehole name respectively).
- A PNG image of each processed PDF page in the `data/output/draw` directory, where the extracted data is highlighted.

## Run Layer Description Classification

To execute the layer description classification, follow these steps:

### 1. Setup

Repeat steps 1 and 2 of the [data extraction pipeline](#run-data-extraction) to set up the environment and download the data.

Pre-trained models are available in two ways:

- **From HuggingFace (recommended):** Published models are hosted on [HuggingFace](https://huggingface.co/swissgeol). No manual download is needed — pass the HuggingFace model ID (e.g. `swissgeol/lithology`) directly via the `-p` flag at runtime and the model is loaded automatically.

- **From S3 (swisstopo internal):** If you have access to the swisstopo S3 bucket, you can download models directly (e.g.For models not yet published on HuggingFace), and pass the local path via `-p`:
  ```bash
  aws s3 sync s3://stijnvermeeren-boreholes-models models/uscs/your_model_folder
  ```

### 2. Run the Classification Pipeline

The main script for the classification pipeline is located at `src/classification/main.py`. A CLI command is available to run this script. Use **Python: Run boreholes-classify** (VSCode launch config) or the equivalent CLI command to classify a single input file:

```bash
# Classify predictions from the extraction pipeline using the Bedrock classifier
boreholes-classify-descriptions -f data/output/predictions.json -c bedrock -cs en_main
```

```bash
# Classify using ground truth as input and evaluate with a baseline classifier
boreholes-classify-descriptions -f data/geoquat_ground_truth.json -c baseline
```

To run across multiple datasets in one command and receive an overview of all child runs at parent level:

```bash
boreholes-classify-descriptions \
  --benchmark "geoquat_gt:data/geoquat_ground_truth.json" \
  --benchmark "geoquat_pred:data/output/predictions.json" \
  -c bedrock -cs en_main
```


**Key options:**

- Use the `-f` or `--file-path` flag to specify the input JSON file (ground truth or predictions from extraction). When a ground truth JSON is provided, only the **test set** defined within it is classified. This ensures evaluation results are consistent with the train/test split used during BERT training. For document-level systems (e.g. `borehole_type`), `-f` can instead point at a PDF file or a directory of PDFs.
- For document-level systems when `-f` is ground truth JSON rather than PDFs, use `-dt` or `--document-texts` to supply the filename → text mapping: either a precomputed `*_filtered_text.json` (see `scripts.extract_full_text_csv`) or a PDF file/directory, extracted live. Repeatable and mixable, like `-f`. Ignored for layer-level systems.
- Use the `-c` or `--classifier-type` option to choose the classifier: `dummy`, `baseline`, `bert`, or `bedrock`.
- If you are using the classifier `bert`, specify the model path using `-p` or `--model-path`:
  - **Full model:** pass either a HuggingFace model ID (downloaded automatically at runtime) or a local directory path (contains `config.json`, `model.safetensors`, tokenizer files, etc.):
    ```bash
    boreholes-classify-descriptions -f data/geoquat_ground_truth.json \
      -c bert -p swissgeol/lithology -cs lithology
    ```
  - **Split model (backbone + head):** pass the head directory via `-p` and the shared backbone via `-b` or `--backbone-path`. This is the recommended approach when using models downloaded from S3:
    ```bash
    boreholes-classify-descriptions -f data/geoquat_ground_truth.json \
      -c bert -p models/lithology_head -b models/backbone/backbone.safetensors -cs lithology
    ```

- Use `--classification-system` or `-cs` to specify the classification system. Supported values:
  `accessory_components`, `alteration_degree_consolidated`, `alteration_degree_unconsolidated`, `borehole_type`, `cementation`, `color_consolidated`, `color_unconsolidated`, `debris`, `en_main`, `en_secondary`, `grain_angularity`, `grain_shape`, `lithology`, `mineral_components`, `organic_components`, `uscs`.
- Use `-o` and `-ob` to specify the output directory and bedrock output directory respectively.

The script will classify all given descriptions and write the predictions to the `data/output_description_classification` directory (or `data/output_description_classification_bedrock` for Bedrock outputs).

Run `boreholes-classify-descriptions --help` to see all available options.

---

## Further information
- [API_and_Docker.md](docs/API_and_Docker.md) documents how to start the API server and how to build the API as a Docker Image.

- [For_Developers.md](docs/For_Developers.md) documents project structure and practical tools and best practices like pre-commit which may be useful for developers.

- [groundtruth-json.md](docs/groundtruth-json.md) documents the expected structure of the ground truth file needed for evaluation.

- [train_BERT.md](docs/train_BERT.md) documents how to fine-tune a BERT model on your own data.

- [CICD.md](docs/CICD.md) overview of the deployment process and CICD pipeline

- [Experiment_Results.md](docs/Experiment_Results.md) overview of the classification performances on internal datasets.
