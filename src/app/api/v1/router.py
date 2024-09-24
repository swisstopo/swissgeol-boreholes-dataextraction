"""Main router for the app."""

from app.api.v1.endpoints.create_pngs import create_pngs
from app.api.v1.endpoints.extract_data import extract_data
from app.common.schemas import (
    ExtractCoordinatesResponse,
    ExtractDataRequest,
    ExtractNumberResponse,
    ExtractTextResponse,
    PNGRequest,
    PNGResponse,
)
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/V1")


####################################################################################################
### Create PNGs
####################################################################################################
@router.post("/create_pngs", tags=["create_pngs"])
def post_create_pngs(request: PNGRequest) -> PNGResponse:
    """Create PNGs from the given data."""
    return create_pngs(request.filename)


####################################################################################################
### Extract Data
####################################################################################################
@router.post(
    "/extract_data",
    tags=["extract_data"],
    response_model=ExtractCoordinatesResponse | ExtractTextResponse | ExtractNumberResponse,
)
def post_extract_data(
    extract_data_request: ExtractDataRequest,
) -> ExtractCoordinatesResponse | ExtractTextResponse | ExtractNumberResponse:
    """Extract data from the given PNGs."""
    try:
        # Extract the data based on the request
        response = extract_data(extract_data_request)
        if isinstance(response, ExtractCoordinatesResponse | ExtractTextResponse | ExtractNumberResponse):
            return response  # Ensure this is a valid response model
        else:
            # Handle an HTTPException response type
            raise response

    except ValueError as e:
        # Handle a known ValueError and return a 400 status
        raise HTTPException(status_code=400, detail=str(e)) from None
