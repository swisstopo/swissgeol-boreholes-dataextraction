"""Main router for the app."""

from typing import Annotated

from fastapi import APIRouter, Body, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.api.v1.endpoints.bounding_boxes import bounding_boxes
from app.api.v1.endpoints.classify_all import classify
from app.api.v1.endpoints.classify_borehole_type import classify_borehole_type
from app.api.v1.endpoints.create_pngs import create_pngs
from app.api.v1.endpoints.extract_data import extract_data
from app.api.v1.endpoints.extract_stratigraphy import extract_stratigraphy
from app.common.schemas import (
    BoundingBoxesRequest,
    BoundingBoxesResponse,
    ClassifyBoreholeTypeResponse,
    ClassifyRequest,
    ClassifyResponse,
    ExtractCoordinatesResponse,
    ExtractDataRequest,
    ExtractNumberResponse,
    ExtractStratigraphyRequest,
    ExtractStratigraphyResponse,
    ExtractTextResponse,
    PNGRequest,
    PNGResponse,
)

router = APIRouter(prefix="/api/V1")

_CLASSIFY_REQUEST_EXAMPLES = {
    "unconsolidated_silt": {
        "summary": "Unconsolidated sediment (silt)",
        "value": {
            "description": (
                "Silt, calcareous, argillaceous, grey, with thin light grey interlayers and "
                "yellowish olive reduction patches, with gravels, angular to sub-rounded, very "
                "poorly to poorly sorted; from 2 m to 4 m: some plant roots; from 6 m to 8 m: "
                "one chert nodule."
            )
        },
    },
    "consolidated_limestone": {
        "summary": "Consolidated rock (limestone)",
        "value": {
            "description": (
                "Peloidal bioclastic limestone, fine to medium grained, slightly oolitic, yellow "
                "to orange, finely sandy, with pyrite and glauconite."
            )
        },
    },
}

_CLASSIFY_RESPONSE_EXAMPLES = {
    "unconsolidated_silt": {
        "summary": "Unconsolidated sediment (silt)",
        "value": {
            "consolidation": "unconsolidated",
            "en_main": "si",
            "en_secondary": ["si", "cl", "gr"],
            "uscs": "not_specified",
            "debris": ["not_specified"],
            "color": "grey",
            "grain_angularity": ["angular", "sub_angular", "sub_rounded"],
            "grain_shape": ["not_specified"],
            "organic_components": ["roots"],
        },
    },
    "consolidated_limestone": {
        "summary": "Consolidated rock (limestone)",
        "value": {
            "consolidation": "consolidated",
            "lithology": "limestone",
            "alteration_degree_consolidated": "not_specified",
            "cementation": "not_specified",
            "color": "yellowish_orange",
            "mineral_components": ["pyrite", "glauconite"],
            "accessory_components": ["ooids", "pellets"],
        },
    },
}


class BadRequestResponse(BaseModel):
    """Response schema for the extract_data endpoint."""

    detail: str


####################################################################################################
### Create PNGs
####################################################################################################
@router.post(
    "/create_pngs",
    tags=["create_pngs"],
    responses={
        400: {"model": BadRequestResponse, "description": "Bad request"},
        404: {
            "model": BadRequestResponse,
            "description": "Failed to load PDF document. The filename is not found in the bucket.",
        },
        500: {"model": BadRequestResponse, "description": "Internal server error"},
    },
)
def post_create_pngs(request: PNGRequest) -> PNGResponse:
    """Create PNG images from a PDF stored in the S3 bucket.

    This endpoint generates PNG images from each page of a specified PDF document stored in the AWS S3 bucket.
    The PDF file must be accessible in the bucket with a valid filename provided in the request.

    ### Request Body
    - **request** (`PNGRequest`): Contains the `filename` of the PDF document in the S3 bucket from which PNGs
    should be generated.

    ### Returns
    - **PNGResponse**: Response containing a list of keys (filenames) for the generated PNG images stored in the
    S3 bucket.

    ### Status Codes
    - **200 OK**: PNG images were successfully created and stored in the S3 bucket.
    - **400 Bad Request**: The request format or content is invalid. Verify that `filename` is correctly specified.
    - **404 Not Found**: PDF file not found in S3 bucket.
    - **500 Internal Server Error**: An error occurred on the server while creating PNGs.

    ### Additional Information
    - The endpoint connects to AWS S3 to retrieve the specified PDF, converts its pages to PNGs, and stores
    the generated images back in S3. Ensure the PDF file exists in the S3 bucket and is accessible before
    making a request.
    """
    return create_pngs(request.filename)


####################################################################################################
### Bounding Boxes for Words
####################################################################################################
@router.post(
    "/bounding_boxes",
    tags=["bounding_boxes"],
    responses={
        400: {"model": BadRequestResponse, "description": "Bad request"},
        404: {
            "model": BadRequestResponse,
            "description": "Failed to load PDF document. The filename is not found in the bucket.",
        },
        500: {"model": BadRequestResponse, "description": "Internal server error"},
    },
)
def get_bounding_boxes(request: BoundingBoxesRequest) -> BoundingBoxesResponse:
    """Obtain bounding boxes (in PNG pixel coordinates) for all words that are found on the requested PDF page.

    ### Prerequisites
    The text in the PDF must be digitally readable, i.e. either a digitally-born PDF, or a PDF where OCR has already
    been executed.

    Ensure that the PDF file has been processed by the create_pngs endpoint first.

    ### Request Body
    - **request** (`BoundingBoxesRequest`): Contains the `filename` of the PDF document in the S3 bucket and the page
    number.

    ### Returns
    - **BoundingBoxesResponse**: Response containing a list of bounding boxes.

    ### Status Codes
    - **200 OK**: The bounding were successfully found.
    - **400 Bad Request**: The request format or content is invalid. Verify that `filename` is correctly specified.
    - **404 Not Found**: PDF file not found in S3 bucket.
    - **500 Internal Server Error**: An error occurred on the server while obtaining the bounding boxes.
    """
    try:
        return bounding_boxes(request.filename, request.page_number)
    except ValueError as e:
        # Handle a known ValueError and return a 400 status
        raise HTTPException(status_code=400, detail=str(e)) from None


####################################################################################################
### Extract Data
####################################################################################################
@router.post(
    "/extract_data",
    tags=["extract_data"],
    response_model=ExtractCoordinatesResponse | ExtractTextResponse | ExtractNumberResponse,
    responses={
        404: {"model": BadRequestResponse, "description": "Coordinates/Text/Number not found"},
        400: {"model": BadRequestResponse, "description": "Bad request"},
        500: {"model": BadRequestResponse, "description": "Internal server error"},
    },
)
def post_extract_data(
    extract_data_request: ExtractDataRequest,
) -> ExtractCoordinatesResponse | ExtractTextResponse | ExtractNumberResponse:
    """Extract specified data from a given document based on the bounding box coordinates and format.

    Text is extracted on a word-by-word basis, whereby a word is included if its center point is within the bounding
    box that is provided by the user in the request.

    ### Prerequisites
    Ensure that the PDF file has been processed by the create_pngs endpoint first.

    ### Request Body
    - **extract_data_request**: Instance of `ExtractDataRequest`, containing file details, page number, bounding
    box, and data format. The bounding box in PNG coordinates helps locate the region to extract data from.

    ### Returns
    The endpoint responds with one of the following response models based on the extracted data:
    - **ExtractCoordinatesResponse**: If geographic coordinates are extracted.
    - **ExtractTextResponse**: If text content is extracted.
    - **ExtractNumberResponse**: If numerical data is extracted.

    ### Status Codes
    - **200 OK**: Successful extraction, returning the specified data type.
    - **400 Bad Request**: Input request was invalid, typically due to misformatted or missing parameters.
    - **401 Unauthorized**: Wrong or incomplete credentials.
    - **404 Not Found**: Requested data could not be found within the specified bounding box or page.
    - **500 Internal Server Error**: An error occurred on the server side during data extraction.

    ### Error Handling
    Known `ValueError`s (e.g., invalid input data) result in a `400 Bad Request` response with a relevant error
    message.
    For other errors, the endpoint returns a `500 Internal Server Error`.
    """
    try:
        # Extract the data based on the request
        response = extract_data(extract_data_request)
        return response

    except ValueError as e:
        # Handle a known ValueError and return a 400 status
        raise HTTPException(status_code=400, detail=str(e)) from None


####################################################################################################
### Extract Stratigraphy
@router.post(
    "/extract_stratigraphy",
    response_model=ExtractStratigraphyResponse,
    tags=["extract_stratigraphy"],
    responses={
        400: {"model": BadRequestResponse, "description": "Bad request"},
        401: {"model": BadRequestResponse, "description": "Unauthorized"},
        404: {"model": BadRequestResponse, "description": "No boreholes found in PDF"},
        500: {"model": BadRequestResponse, "description": "Internal server error"},
    },
)
def post_extract_stratigraphy(request: ExtractStratigraphyRequest) -> ExtractStratigraphyResponse:
    """Extract all boreholes with stratigraphy (depths and material descriptions) from the entire PDF file.

    Optionally includes groundwater measurements when requested via the `include_groundwater` parameter.

    Scans all pages of the PDF and returns all boreholes found,
    each containing page numbers and stratigraphy layers with bounding boxes.
    When include_groundwater is True, also returns groundwater measurements (depth, date, elevation).

    ### Request Body
    - **filename**: The PDF filename to process.
    - **include_groundwater** (optional): If True, include groundwater data in response. Default is False.

    ### Returns
    - List of boreholes with layers and bounding boxes.
    - Optional list of groundwater measurements (if include_groundwater=True).

    ### Status Codes
    - 200: Successful extraction
    - 400: Invalid request or unable to open PDF
    - 401: Unauthorized: Wrong or incomplete credentials.
    - 404: No boreholes found or PNG files not generated
    - 500: Internal error

    ### Notes
    - Groundwater depth values are limited to MAX_DEPTH (200m) to avoid confusion with elevation values
    - Date and elevation fields may be null if not detected in the document
    - Bounding boxes are in PNG pixel coordinates (scaled 3x from PDF coordinates)
    """
    return extract_stratigraphy(request.filename, request.include_groundwater)


####################################################################################################
### Classify (unified multi-task)
####################################################################################################
@router.post(
    "/classify",
    tags=["classify"],
    response_model=ClassifyResponse,
    response_model_exclude_unset=True,
    responses={
        200: {"content": {"application/json": {"examples": _CLASSIFY_RESPONSE_EXAMPLES}}},
        400: {"model": BadRequestResponse, "description": "Bad request"},
        500: {"model": BadRequestResponse, "description": "Internal server error"},
        503: {"model": BadRequestResponse, "description": "BERT models not loaded (set BERT_ENABLED=true)"},
    },
)
def post_classify(
    request: Annotated[ClassifyRequest, Body(openapi_examples=_CLASSIFY_REQUEST_EXAMPLES)],
    http_request: Request,
) -> ClassifyResponse:
    """Classify a plain-text material description across all relevant tasks in one forward pass.

    The backbone embedding is computed once from the description, then fed independently into each
    task-specific classification head. The lithology head determines whether the material is consolidated
    or unconsolidated; only tasks relevant to that rock type are returned.

    ### Request Body
    - **description**: Plain-text material description (e.g. `"schwach tonig-siltiger Sand und Kies,
      brau-beige, Komponenten vorw. eckig"`).

    ### Returns
    - **consolidation**: `"consolidated"` or `"unconsolidated"`, as determined by the lithology head;
      indicates which of the fields are populated.
    - One field per classification task. A field is `null` if that task isn't relevant to the inferred
      rock type; otherwise it holds the predicted class name (single-label tasks, e.g. `en_main`, `uscs`,
      `color`) or class names (multi-label/rank tasks, e.g. `en_secondary`, `grain_angularity`, `grain_shape`,
      `organic_components`, `accessory_components`, `debris`, `mineral_components`).

    ### Consolidated rock tasks
    `lithology`, `alteration_degree_consolidated`, `cementation`, `color`, `mineral_components`, `accessory_components`

    ### Unconsolidated sediment tasks
    `en_main`, `en_secondary`, `uscs`, `debris`, `color`, `grain_angularity`, `grain_shape`, `organic_components`

    ### Status Codes
    - **200 OK**: Classification completed successfully.
    - **400 Bad Request**: Invalid request parameters.
    - **500 Internal Server Error**: Model loading or inference failure.
    - **503 Service Unavailable**: BERT models were not loaded at startup (`BERT_ENABLED=false`).
    """
    if http_request.app.state.bert_models is None:
        raise HTTPException(
            status_code=503,
            detail="Classification endpoint is disabled. Set BERT_ENABLED=true to enable BERT model loading.",
        )
    return classify(request, http_request.app.state.bert_models)


####################################################################################################
### Classify Borehole Type
####################################################################################################
@router.post(
    "/classify_borehole_type",
    tags=["classify_borehole_type"],
    response_model=ClassifyBoreholeTypeResponse,
    responses={
        400: {"model": BadRequestResponse, "description": "Bad request (e.g. not a PDF)"},
        500: {"model": BadRequestResponse, "description": "Internal server error"},
        503: {"model": BadRequestResponse, "description": "BERT models not loaded (set BERT_ENABLED=true)"},
    },
)
async def post_classify_borehole_type(
    http_request: Request,
    file: UploadFile = File(..., description="The PDF document to classify."),  # noqa: B008
) -> ClassifyBoreholeTypeResponse:
    """Classify the borehole type of every borehole detected in an uploaded PDF document.

    This endpoint takes a raw PDF file upload:
    it runs the extraction pipeline to detect each borehole in the document, extracts header-like
    text scoped to each borehole's own pages, and classifies each one independently with a single full
    forward pass through the `borehole_type` model.

    ### Request

    A `multipart/form-data` upload with a single `file` field containing the PDF.

    ### Returns
    - **boreholes**: One `{borehole_index, class_name}` entry per borehole detected in the document.

    ### Status Codes
    - **200 OK**: Classification completed successfully.
    - **400 Bad Request**: The uploaded file is not a PDF.
    - **500 Internal Server Error**: Model loading or inference failure.
    - **503 Service Unavailable**: BERT models were not loaded at startup (`BERT_ENABLED=false`).
    """
    if http_request.app.state.bert_models is None:
        raise HTTPException(
            status_code=503,
            detail="Classification endpoint is disabled. Set BERT_ENABLED=true to enable BERT model loading.",
        )
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid request. The uploaded file must be a PDF.")

    data = await file.read()
    return classify_borehole_type(data, file.filename, http_request.app.state.bert_models)
