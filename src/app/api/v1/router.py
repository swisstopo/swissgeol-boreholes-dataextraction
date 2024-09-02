"""Main router for the app."""

from app.api.v1.endpoints.create_pngs import create_pngs
from app.api.v1.endpoints.extract_data import extract_data
from app.common.schemas import (
    ExtractCoordinatesResponse,
    ExtractDataRequest,
    ExtractDataResponse,
    ExtractElevationResponse,
    ExtractNumberResponse,
    ExtractTextResponse,
    PNGRequest,
    PNGResponse,
)
from fastapi import APIRouter

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
    response_model=ExtractCoordinatesResponse | ExtractElevationResponse | ExtractTextResponse | ExtractNumberResponse,
)
def post_extract_data(extract_data_request: ExtractDataRequest) -> ExtractDataResponse:
    """Extract data from the given PNGs."""
    return extract_data(extract_data_request)
