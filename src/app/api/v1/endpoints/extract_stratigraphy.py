"""This module defines the FastAPI endpoint for extracting all boreholes with their stratigraphy."""

from app.common.aws import load_pdf_from_aws
from app.common.schemas import (
    BoreholeExtractionSchema,
    BoreholeLayerSchema,
    ExtractStratigraphyResponse,
)
from extraction.features.extract import process_page
from extraction.features.stratigraphy.layer.duplicate_detection import remove_duplicate_layers
from extraction.features.stratigraphy.layer.layer import LayersInDocument
from extraction.features.utils.geometry.line_detection import extract_lines
from extraction.features.utils.text.extract_text import extract_text_lines
from fastapi import HTTPException
from utils.file_utils import read_params
from utils.language_detection import detect_language_of_document

matching_params = read_params("matching_params.yml")
line_detection_params = read_params("line_detection_params.yml")


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
    try:
        document = load_pdf_from_aws(filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Cannot open PDF file") from e

    language = detect_language_of_document(
        document, matching_params["default_language"], matching_params["material_description"].keys()
    )

    layers_with_bb_in_document = LayersInDocument([], filename)
    for page_index, page in enumerate(document):
        # TODO should be funcs
        page_number = page_index + 1
        text_lines = extract_text_lines(page)
        geometric_lines = extract_lines(page, line_detection_params)
        extracted_boreholes = process_page(text_lines, geometric_lines, language, page_number, **matching_params)
        processed_page_results = LayersInDocument(extracted_boreholes, filename)
        if page_index > 0:
            layer_with_bb_predictions = remove_duplicate_layers(
                previous_page=document[page_index - 1],
                current_page=page,
                previous_layers_with_bb=layers_with_bb_in_document,
                current_layers_with_bb=processed_page_results,
                img_template_probability_threshold=matching_params["img_template_probability_threshold"],
            )
        else:
            layer_with_bb_predictions = processed_page_results.boreholes_layers_with_bb
        layers_with_bb_in_document.assign_layers_to_boreholes(layer_with_bb_predictions)

    if not layers_with_bb_in_document.boreholes_layers_with_bb:
        raise HTTPException(status_code=404, detail="No boreholes found in PDF.")

    extracted_stratigraphy = create_response_object(layers_with_bb_in_document)

    return extracted_stratigraphy


def create_response_object(layers_with_bb: LayersInDocument) -> ExtractStratigraphyResponse:
    """Create a response object from the extracted layers with bounding boxes.

    Args:
        layers_with_bb (LayersInDocument): Object containing borehole layers with bounding boxes.

    Returns:
        ExtractStratigraphyResponse: Response object containing the extracted stratigraphy data.
    """
    boreholes: list[BoreholeExtractionSchema] = []

    for borehole in layers_with_bb.boreholes_layers_with_bb:
        layers: list[BoreholeLayerSchema] = []
        page_numbers = set()

        for prediction in borehole.predictions:
            layer = BoreholeLayerSchema.from_prediction(prediction)
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
