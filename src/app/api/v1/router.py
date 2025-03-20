"""Main router for the app."""

from app.api.v1.endpoints.bounding_boxes import bounding_boxes
from app.api.v1.endpoints.create_pngs import create_pngs
from app.api.v1.endpoints.extract_data import extract_data
from app.common.schemas import (
    BoundingBoxesRequest,
    BoundingBoxesResponse,
    ExtractCoordinatesResponse,
    ExtractDataRequest,
    ExtractNumberResponse,
    ExtractTextResponse,
    PNGRequest,
    PNGResponse,
)
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/V1")


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
    return bounding_boxes(request.filename, request.page_number)


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
