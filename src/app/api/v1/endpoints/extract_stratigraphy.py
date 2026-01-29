"""This module defines the FastAPI endpoint for extracting all boreholes with their stratigraphy."""

from app.common.aws import load_pdf_from_aws
from app.common.helpers import load_png
from app.common.schemas import (
    BoreholeExtractionSchema,
    BoreholeLayerSchema,
    ExtractStratigraphyResponse,
    GroundwaterSchema,
)
from extraction.features.extract import extract_page
from extraction.features.groundwater.groundwater_extraction import (
    GroundwaterInDocument,
    GroundwaterLevelExtractor,
)
from extraction.features.metadata.borehole_name_extraction import NameInDocument
from extraction.features.predictions.borehole_predictions import BoreholePredictions
from extraction.features.predictions.predictions import BoreholeListBuilder
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

matching_params = read_params("matching_params.yml")
line_detection_params = read_params("line_detection_params.yml")
table_detection_params = read_params("table_detection_params.yml")
striplog_detection_params = read_params("striplog_detection_params.yml")


def extract_stratigraphy(filename: str, include_groundwater: bool = False) -> ExtractStratigraphyResponse:
    """Extract all boreholes with stratigraphy from the entire PDF file.

    Optionally includes groundwater measurements in the response. When enabled,
    groundwater measurements are linked to specific boreholes based on spatial
    proximity.

    This endpoint scans all pages of the PDF and returns all boreholes found,
    each with page numbers and a list of stratigraphy layers (material descriptions,
    start depths, end depths), including bounding boxes with page references.

    When include_groundwater is True, also extracts groundwater measurements
    (depth, date, elevation) and includes them per-borehole in the response.

    Args:
        filename (str): Filename of the PDF.
        include_groundwater (bool): If True, include groundwater data in response.
            Default is False.

    Returns:
        ExtractStratigraphyResponse: Response with boreholes (with their stratigraphy layers,
            bounding boxes, and optionally groundwater per borehole).
    """
    # 1. load the pdf
    document = load_pdf_from_aws(filename)

    language = detect_language_of_document(
        document, matching_params["default_language"], matching_params["material_description"].keys()
    )

    pdf_img_scalings = []

    # Initialize groundwater extractor and collection if needed
    groundwater_extractor = None
    groundwater_in_doc = None
    if include_groundwater:
        groundwater_extractor = GroundwaterLevelExtractor(language, matching_params)
        groundwater_in_doc = GroundwaterInDocument([], filename)

    boreholes_per_page = []
    for page_index, page in enumerate(document):
        # 2. load the png image to infer the scaling, MUST have been generated before
        png_page = load_png(filename, page_index + 1)  # page number is 1-indexed
        pdf_img_scalings.append((png_page.shape[1] / page.rect.width, png_page.shape[0] / page.rect.height))

        # 3. extract layers
        text_lines = extract_text_lines(page)
        long_or_horizontal_lines, all_geometric_lines = extract_lines(page, line_detection_params)

        # Detect table structures on the page
        table_structures = detect_table_structures(page, long_or_horizontal_lines, text_lines, table_detection_params)

        # Detect strip logs on the page
        strip_logs = detect_strip_logs(page, text_lines, striplog_detection_params)

        extracted_boreholes = extract_page(
            text_lines,
            long_or_horizontal_lines,
            all_geometric_lines,
            table_structures,
            strip_logs,
            language,
            page_index,
            page,
            line_detection_params,
            None,
            **matching_params,
        )
        boreholes_per_page.append(extracted_boreholes)

        # Extract groundwater if requested (uses the extracted boreholes as context)
        if include_groundwater and groundwater_extractor and groundwater_in_doc:
            groundwater_entries = groundwater_extractor.extract_groundwater(
                page_number=page_index + 1,
                text_lines=text_lines,
                geometric_lines=long_or_horizontal_lines,
                extracted_boreholes=extracted_boreholes,
            )
            groundwater_in_doc.groundwater_feature_list.extend(groundwater_entries)

    layers_with_bb_in_document = LayersInDocument(merge_boreholes(boreholes_per_page, matching_params), filename)

    # Match groundwater entries to boreholes if requested
    borehole_predictions_list = (
        build_borehole_predictions(layers_with_bb_in_document, filename, groundwater_in_doc)
        if groundwater_in_doc
        else None
    )

    return create_response_object(layers_with_bb_in_document, pdf_img_scalings, borehole_predictions_list)


def build_borehole_predictions(
    layers_with_bb_in_document: LayersInDocument,
    filename: str,
    groundwater_in_doc: GroundwaterInDocument,
) -> list[BoreholePredictions]:
    """Build borehole predictions with groundwater matched to boreholes.

    Uses BoreholeListBuilder to spatially match groundwater entries to their closest
    boreholes, consistent with the logic in main.py.

    Args:
        layers_with_bb_in_document (LayersInDocument): Extracted layers with bounding boxes.
        filename (str): Name of the PDF file.
        groundwater_in_doc (GroundwaterInDocument): Groundwater entries to match to boreholes.

    Returns:
        list[BoreholePredictions]: List of borehole predictions with matched groundwater.
    """
    # Create empty NameInDocument for future use (names, elevations, coordinates left empty for now)
    names_in_doc = NameInDocument([], filename)

    borehole_predictions_list = BoreholeListBuilder(
        layers_with_bb_in_document=layers_with_bb_in_document,
        file_name=filename,
        groundwater_in_doc=groundwater_in_doc,
        names_in_doc=names_in_doc,
        elevations_list=[],
        coordinates_list=[],
    ).build()

    # Remove duplicates and infer depth/elevation for groundwater entries
    for borehole in borehole_predictions_list:
        borehole.filter_groundwater_entries()

    return borehole_predictions_list


def create_response_object(
    layers_with_bb: LayersInDocument,
    pdf_img_scalings: list[tuple[float]],
    borehole_predictions_list: list[BoreholePredictions] | None = None,
) -> ExtractStratigraphyResponse:
    """Create a response object from the extracted layers with bounding boxes.

    Args:
        layers_with_bb (LayersInDocument): Object containing borehole layers with bounding boxes.
        pdf_img_scalings (list): list of 2-tuples containing the height and width scalings to convert the coordinates
            from pdf to png. This scaling is chosen at the png creation, and could potentially be simplified.
        borehole_predictions_list (list[BoreholePredictions] | None): Optional list of borehole predictions
            containing matched groundwater entries. If provided, groundwater will be included per borehole.

    Returns:
        ExtractStratigraphyResponse: Response object containing the extracted stratigraphy data.
    """
    boreholes: list[BoreholeExtractionSchema] = []

    for borehole_index, borehole in enumerate(layers_with_bb.boreholes_layers_with_bb):
        layers: list[BoreholeLayerSchema] = []
        page_numbers = set()

        for prediction in borehole.predictions:
            layer = BoreholeLayerSchema.from_prediction(prediction, pdf_img_scalings)
            layers.append(layer)

            if layer.material_description:
                page_numbers.update(bb.page_number for bb in layer.material_description.bounding_boxes)

        # Get groundwater for this borehole if available
        groundwater_list = None
        if borehole_predictions_list:
            borehole_prediction = borehole_predictions_list[borehole_index]
            groundwater_list = [
                GroundwaterSchema.from_prediction(gw_entry, pdf_img_scalings)
                for gw_entry in borehole_prediction.groundwater_in_borehole.groundwater_feature_list
            ]

        boreholes.append(
            BoreholeExtractionSchema(
                page_numbers=sorted(page_numbers),
                layers=layers,
                groundwater=groundwater_list,
            )
        )

    return ExtractStratigraphyResponse(boreholes=boreholes)
