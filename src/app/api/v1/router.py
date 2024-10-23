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
    """Create PNGs from the given data."""
    return create_pngs(request.filename)


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
    """Extract data from the given PNGs."""
    try:
        # Extract the data based on the request
        response = extract_data(extract_data_request)
        return response

    except ValueError as e:
        # Handle a known ValueError and return a 400 status
        raise HTTPException(status_code=400, detail=str(e)) from None
