"""This module defines the FastAPI endpoint for extracting all boreholes with their stratigraphy."""

from app.common.aws import load_pdf_from_aws
from app.common.helpers import load_png
from app.common.schemas import (
    BoreholeExtractionSchema,
    BoreholeLayerSchema,
    ExtractStratigraphyResponse,
)
from extraction.features.extract import extract_page
from extraction.features.stratigraphy.layer.continuation_detection import merge_boreholes
from extraction.features.stratigraphy.layer.layer import LayersInDocument
from swissgeol_doc_processing.geometry.line_detection import extract_lines
from swissgeol_doc_processing.text.extract_text import extract_text_lines
from swissgeol_doc_processing.utils.file_utils import read_params
from swissgeol_doc_processing.utils.language_detection import detect_language_of_document
from swissgeol_doc_processing.utils.strip_log_detection import detect_strip_logs
from swissgeol_doc_processing.utils.table_detection import (
    detect_table_structures,
)

config_path = "config"
matching_params = read_params("matching_params.yml", config_path)
line_detection_params = read_params("line_detection_params.yml", config_path)
table_detection_params = read_params("table_detection_params.yml", config_path)


def extract_stratigraphy(filename: str) -> ExtractStratigraphyResponse:
    """Extract all boreholes with stratigraphy from the entire PDF file.

    This endpoint scans all pages of the PDF and returns all boreholes found,
    each with page numbers and a list of stratigraphy layers (material descriptions,
    start depths, end depths), including bounding boxes with page references.

    Args:
        filename (str): filename of the PDF.

    Returns:
        List[BoreholeExtraction]: List of boreholes with their stratigraphy layers and bounding boxes.
    """
    # 1. load the pdf
    document = load_pdf_from_aws(filename)

    language = detect_language_of_document(
        document, matching_params["default_language"], matching_params["material_description"].keys()
    )

    pdf_img_scalings = []

    boreholes_per_page = []
    for page_index, page in enumerate(document):
        # 2. load the png image to infer the scaling, MUST have been generated before
        png_page = load_png(filename, page_index + 1)  # page number is 1-indexed
        pdf_img_scalings.append((png_page.shape[1] / page.rect.width, png_page.shape[0] / page.rect.height))

        # 3. extract layers
        text_lines = extract_text_lines(page)
        geometric_lines = extract_lines(page, line_detection_params)

        # Detect table structures on the page
        table_structures = detect_table_structures(
            page_index, document, geometric_lines, text_lines, table_detection_params
        )

        # Detect strip logs on the page
        strip_logs = detect_strip_logs(
            page, geometric_lines, text_lines, line_detection_params, table_detection_params
        )

        boreholes_per_page.append(
            extract_page(
                text_lines,
                geometric_lines,
                table_structures,
                strip_logs,
                language,
                page_index,
                document,
                line_detection_params,
                None,
                **matching_params,
            )
        )

    layers_with_bb_in_document = LayersInDocument(merge_boreholes(boreholes_per_page, matching_params), filename)
    extracted_stratigraphy = create_response_object(layers_with_bb_in_document, pdf_img_scalings)

    return extracted_stratigraphy


def create_response_object(
    layers_with_bb: LayersInDocument, pdf_img_scalings: list[tuple[float]]
) -> ExtractStratigraphyResponse:
    """Create a response object from the extracted layers with bounding boxes.

    Args:
        layers_with_bb (LayersInDocument): Object containing borehole layers with bounding boxes.
        pdf_img_scalings (list): list of 2-tuples containing the height and width scalings to convert the coordinates
            from pdf to png. This scaling is chosen at the png creation, and could potentially be simplified.

    Returns:
        ExtractStratigraphyResponse: Response object containing the extracted stratigraphy data.
    """
    boreholes: list[BoreholeExtractionSchema] = []

    for borehole in layers_with_bb.boreholes_layers_with_bb:
        layers: list[BoreholeLayerSchema] = []
        page_numbers = set()

        for prediction in borehole.predictions:
            layer = BoreholeLayerSchema.from_prediction(prediction, pdf_img_scalings)
            layers.append(layer)

            if layer.material_description:
                page_numbers.update(bb.page_number for bb in layer.material_description.bounding_boxes)

        boreholes.append(
            BoreholeExtractionSchema(
                page_numbers=sorted(page_numbers),
                layers=layers,
            )
        )

    return ExtractStratigraphyResponse(boreholes=boreholes)
