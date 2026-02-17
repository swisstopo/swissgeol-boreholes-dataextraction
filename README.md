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
With the current version of the code, groundwater can only be found at depth smaller than 200 meters. This threshold is defined in `src/stratigraphy/groundwater/groundwater_extraction.py` by the constant `MAX_DEPTH`. 

The groundwater is extracted in two main ways from the borehole documents. The first one aims to match a groundwater-related keyword in the text extracted from the document (e.g., groundwater, groundwater-level). The second technique focuses on extracting the groundwater-related symbol from the document. It aims at finding 2 horizontal lines, on top of each other, satisfying some visual requirement that the groundwater symbol has.


## Project management

This project is managed and financed by the Swiss Federal Office of Topography [swisstopo](https://www.swisstopo.admin.ch/). Many contributions come from [Visium](https://www.visium.ch/), in their role as contractor for swisstopo for this project.

This project is released as open source software, under the principle of "_public money, public code_", in accordance with the 2023 federal law "[_EMBAG_](https://www.fedlex.admin.ch/eli/fga/2023/787/de)", and following the guidance of the [tools for OSS published by the Federal Chancellery](https://www.bk.admin.ch/bk/en/home/digitale-transformation-ikt-lenkung/bundesarchitektur/open_source_software/hilfsmittel_oss.html).

We welcome feedback, bug reports and code contributions (provided they are compatible with swisstopo's roadmap) from third parties. Feature requests and support requests can only be fulfilled as long as they are compatible with swisstopo's legal mandate. Other organisations (both within Switzerland or internationally) who manage their own borehole data and are interested in using and/or contributing to this project, are encouraged to contact us, in order to discuss the possibility of establishing a partnership for collaborating more closely on this project.

### Main contributors

* Stijn Vermeeren [@stijnvermeeren-swisstopo](https://www.github.com/stijnvermeeren-swisstopo) (swisstopo) - Project Lead
* David Cleres [@dcleres](https://www.github.com/dcleres) (Visium)
* Renato Durrer [@redur](https://www.github.com/redur) (Visium)

### License

The source code of this project is licensed under the [MIT License](LICENSE). Certain libraries in the project dependencies might be distributed under more restrictive open source licenses. Most notably, [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright) is available under either the AGPL license or a commercial license. These more restrictive licenses would apply when distributing this project in a single package together with its dependencies.

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

If you are using a version of Python newer than 3.12 (e.g. 3.13), you may need to use the command ` python3.12 -m venv env `instead.
It might also be usefull to use `python3.12 -m pip install -e '.[all]'` if some error appear.

## Run data extraction
To execute the data extraction pipeline, follow these steps:

### 1. Activate the virtual environment

Activate your virtual environment. On unix systems this is

``` bash
source env/bin/activate
```

### 2. Download the borehole profiles, optional

Use `boreholes-download-profiles` to download the files to be processed from an AWS S3 storage. In order to do so, you need to authenticate with aws first. We recommend using the aws CLI for that purpose, or storing your credentials in the ~/.aws configuration files This step is optional, you can continue with step 3 on your own set of borehole profiles.

Alternativelly, you can download the data directly using the AWS CLI:
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

## 3. Run the extraction script

The main script for the extraction pipeline is located at `src/stratigraphy/main.py`. A cli command is created to run this script.

Run `boreholes-extract-all` to run the main extraction script. You need to specify the input directory or a single PDF file using the `-i` or `--input-directory` flag. 
The script will source all PDFs from the specified directory and create PNG files in the `data/output/draw` directory.

Use `boreholes-extract-all --help` to see all options for the extraction script.

To run the extraction pipeline on multiple datasets in one command and to receive an overview of all chilf runs on parent level,  you can run `boreholes-extract-multi-benchmark`. 
Use repeatable benchmark specs with syntax `"<name>:<input_path>:<ground_truth_path>"` , e.g. 
```python 
"--benchmark",
"zurich:data/zurich:data/zurich_ground_truth.json",
```

To apply custom settings, generate a local copy of the configuration files using the package helper. Any values you change locally will override the package defaults.

```python
from swissgeol_doc_processing.utils.file_utils import expose_configs

# Create a local "config/" folder populated with the default settings
expose_configs()
```

This will create a `config/` directory at the root of your project containing all configuration files that can be safely edited.

### 4. Check the results

The script produces output in two different formats:
- A file `data/output/predictions.json` that contains all extracted data in a machine-readable format. The structure of this file is documented in [README.predictions-json.md](README.predictions-json.md).
- A PNG image of each processed PDF page in the `data/output/draw` directory, where the extracted data is highlighted.

## Run Layer Description Classification  

To execute the layer description classification, follow these steps:  

### 1. Setup  

Repeat steps 1 and 2 of the [data extraction pipeline](#run-data-extraction) to set up the environment and download the data.  

### 2. Run the Classification Pipeline  

The main script for the classification pipeline is located at `src/description_classification/main.py`. A CLI command is available to run this script:  

```bash
boreholes-classify-descriptions -f data/geoquat_ground_truth.json -s data/geoquat/validation -c baseline
```  

- Use the `-f` or `--file-path` flag to specify the path to the JSON file containing the layer description and USCS ground truth.  
- You can optionally provide a folder containing a subset of files by using the `-s` or `--file-subset-directory` flag. Only the filenames in this folder will be loaded from the JSON file for classification.
- Use the `-c` or `--classifier` option to choose the classifier type from `dummy`, `baseline` or `bert`.
- If you are using the classifier `bert` with a trained local [model](#train-bert-model), specify its folder path using the `-p` or `--model-path` flag. The folder has to contain all files generated when saving a model checkpoint with the transformers library.
- Use `--classification-system` or `-cs` to specify the system you want to classify the description into. Currently supports `uscs`, `lithology` and `en_main`.

The script will classify all given descriptions and write the predictions to the `data/output_description_classification` directory.  

Run `boreholes-classify-descriptions --help` to see all available options.  

To run the classificaton pipeline on multiple datasets in one command and to receive an overview of all chilf runs on parent level, you can run `boreholes-classify-descriptions (multi, single-file).`or `Run boreholes-classify-predictions (multi, predictions+GT)`
Use repeatable benchmark specs with syntax `"<name>:<file_path>:<subset_dir>"` or `"<name>:<predictions_path>:<ground_truth_path>"`respectively. 
E.g. 
```python 
"--benchmark", "val:data/geoquat_ground_truth.json:data/geoquat/validation",
"--benchmark", "test:data/geoquat_ground_truth.json:data/geoquat/test",
```
or 
```python
"--benchmark", "pred_geoquat:path_to_prediction/predictions.json:data/geoquat_ground_truth.json",
"--benchmark", "pred_zurich:path_to_prediction/predictions.json:data/zurich_ground_truth.json",
```
The arguments `-c`, `-p`, and `-cs` are applicable as well. 
Additionally, `-o`and `-ob`can be used to specify the ouput and bedrock output directories. 

---  

## Train BERT Model  

To fine-tune BERT on your data, follow these steps:  

### 1. Setup  

Repeat steps 1 and 2 of the [data extraction pipeline](#run-data-extraction) to set up the environment and download the data.  

### 2. Choose Hyperparameters  

Modify the file `config/bert_config_uscs.yml` to set the hyperparameters for training and data processing. Data sources used for training and validation are specified in this file.

### 3. Train the Model  

To fine-tune BERT from the base model on Hugging Face, run:  

```bash
fine-tune-bert -cf bert_config_uscs.yml
```  

- Use `-cf` or `--config-file-path` to specify the config file containing the training parameters.
- By default, the initial model is the one specified in the config file (loaded from hugging face). However, you can continue training from a specific checkpoint by providing a local model path with `-c` or `--model-checkpoint`.  

The pipeline stores a checkpoint of the model after each epoch and logs training details in the `models` directory. The model name corresponds to the timestamp when training was launched.  


# Further information 
[README.API_and_Docker.md](README.API_and_Docker.md) documents how to start the API server and how to build the API as a Docker Image. 
[README.For_Developers.md](README.For_Developers.md) documents project structure and practical tools and best practices like pre-commit which may be usefull for developers. 
[README.groundtruth-json.md](README.groundtruth-json.md) documents the expexted structure of the ground truth file needed for evaluation. 
