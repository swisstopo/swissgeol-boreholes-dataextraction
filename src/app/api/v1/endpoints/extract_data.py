"""This module defines the FastAPI endpoint for extracting information from PNG images."""

from app.common.schemas import (
    BoundingBox,
    Coordinates,
    ExtractCoordinatesResponse,
    ExtractDataRequest,
    ExtractDataResponse,
    ExtractElevationResponse,
    FormatTypes,
)


def extract_data(extract_data_request: ExtractDataRequest) -> ExtractDataResponse:
    """Extract information from PNG images.

    Args:
        extract_data_request (ExtractDataRequest): The request data.

    Returns:
        ExtractDataResponse: The extracted information.
    """
    # Extract the information based on the format type
    if extract_data_request.format == FormatTypes.COORDINATES:
        return extract_coordinates(extract_data_request)
    elif extract_data_request.format == FormatTypes.ELEVATION:
        return extract_elevation(extract_data_request)
    else:
        raise ValueError("Invalid format type.")


def extract_coordinates(extract_data_request: ExtractDataRequest) -> ExtractDataResponse:
    """Extract coordinates from a PNG image.

    Args:
        extract_data_request (ExtractDataRequest): The request data.

    Returns:
        ExtractDataResponse: The extracted coordinates.
    """
    # Extract the coordinates
    east_coordinate = 1.0
    north_coordinate = 2.0
    bbox = BoundingBox(x0=20.0, y0=40.0, x1=60.0, y1=80.0)
    coordinates = Coordinates(east=east_coordinate, north=north_coordinate, page=1)
    return ExtractCoordinatesResponse(bbox=bbox, coordinates=coordinates)


def extract_elevation(extract_data_request: ExtractDataRequest) -> ExtractDataResponse:
    """Extract the elevation from a PNG image.

    Args:
        extract_data_request (ExtractDataRequest): The request data.

    Returns:
        ExtractDataResponse: The extracted elevation.
    """
    # Extract the elevation
    elevation = 444
    bbox = BoundingBox(x0=20.0, y0=40.0, x1=60.0, y1=80.0)
    return ExtractElevationResponse(bbox=bbox, elevation=elevation)
