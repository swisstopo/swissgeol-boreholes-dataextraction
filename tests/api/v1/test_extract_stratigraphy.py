"""Tests for the extract_stratigraphy endpoint."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.common.schemas import ExtractStratigraphyRequest

# Get the project root directory (3 levels up from this file in tests/api/v1/)
PROJECT_ROOT = Path(__file__).parents[3]

# Define test file paths
TEST_PDF_KEY = "sample.pdf"
TEST_PDF_PATH = str(PROJECT_ROOT / "example" / "example_borehole_profile.pdf")
TEST_PNG_KEY = "dataextraction/sample-1.png"
TEST_PNG_PATH = str(PROJECT_ROOT / "example" / "sample-1.png")


@pytest.fixture(scope="function")
def upload_test_pdf(s3_client, test_bucket_name):
    """Upload a test PDF file to S3."""
    s3_client.upload_file(Filename=str(TEST_PDF_PATH), Bucket=test_bucket_name, Key=TEST_PDF_KEY)


@pytest.fixture(scope="function")
def upload_test_png(s3_client, upload_test_pdf, test_bucket_name):
    """Upload a test PNG file to S3."""
    s3_client.upload_file(Filename=str(TEST_PNG_PATH), Bucket=test_bucket_name, Key=str(TEST_PNG_KEY))


def test_extract_stratigraphy_success(test_client: TestClient, upload_test_pdf, upload_test_png):
    """Test successful extraction of stratigraphy."""
    request = ExtractStratigraphyRequest(filename=TEST_PDF_KEY)
    response = test_client.post("/api/V1/extract_stratigraphy", content=request.model_dump_json())
    assert response.status_code == 200
    json_response = response.json()

    # Verify the response structure
    assert "boreholes" in json_response
    assert isinstance(json_response["boreholes"], list)

    if json_response["boreholes"]:
        borehole = json_response["boreholes"][0]
        assert "layers" in borehole
        assert isinstance(borehole["layers"], list)

        if borehole["layers"]:
            layer = borehole["layers"][0]
            # Verify layer structure
            assert "material_description" in layer
            assert "start" in layer
            assert "end" in layer


def test_extract_stratigraphy_nonexistent_pdf(test_client: TestClient):
    """Test request with nonexistent PDF file."""
    request = ExtractStratigraphyRequest(filename="nonexistent.pdf")
    response = test_client.post("/api/V1/extract_stratigraphy", content=request.model_dump_json())
    assert response.status_code == 404
    assert response.json() == {"detail": "Document nonexistent.pdf not found in S3 bucket."}


def test_extract_stratigraphy_nonexistent_png(test_client: TestClient, upload_test_pdf):
    """Test request with nonexistent PNG file."""
    request = ExtractStratigraphyRequest(filename=TEST_PDF_KEY)
    response = test_client.post("/api/V1/extract_stratigraphy", content=request.model_dump_json())
    assert response.status_code == 404
    assert response.json() == {
        "detail": (
            "PNG file dataextraction/sample-1.png not found. The PNG files"
            " need to be generated first using the create_pngs endpoint."
        )
    }


def test_extract_stratigraphy_invalid_filename(test_client: TestClient):
    """Test request with invalid filename."""
    # Test with non-PDF file
    request = ExtractStratigraphyRequest(filename="test.txt")
    response = test_client.post("/api/V1/extract_stratigraphy", content=request.model_dump_json())
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid request. The filename must end with '.pdf'."}


def test_extract_stratigraphy_default_no_groundwater(test_client: TestClient, upload_test_pdf, upload_test_png):
    """Test that default behavior does NOT include groundwater (backward compatibility)."""
    request = ExtractStratigraphyRequest(filename=TEST_PDF_KEY)
    response = test_client.post("/api/V1/extract_stratigraphy", content=request.model_dump_json())
    assert response.status_code == 200
    json_response = response.json()

    # Verify stratigraphy is present
    assert "boreholes" in json_response
    assert isinstance(json_response["boreholes"], list)

    # Verify groundwater is NOT present (or is None)
    assert json_response.get("groundwater") is None


def test_extract_stratigraphy_explicit_false(test_client: TestClient, upload_test_pdf, upload_test_png):
    """Test that include_groundwater=False does not include groundwater."""
    request = ExtractStratigraphyRequest(filename=TEST_PDF_KEY, include_groundwater=False)
    response = test_client.post("/api/V1/extract_stratigraphy", content=request.model_dump_json())
    assert response.status_code == 200
    json_response = response.json()

    assert "boreholes" in json_response
    assert json_response.get("groundwater") is None


def test_extract_stratigraphy_with_groundwater(test_client: TestClient, upload_test_pdf, upload_test_png):
    """Test that include_groundwater=True includes groundwater data."""
    request = ExtractStratigraphyRequest(filename=TEST_PDF_KEY, include_groundwater=True)
    response = test_client.post("/api/V1/extract_stratigraphy", content=request.model_dump_json())
    assert response.status_code == 200
    json_response = response.json()

    # Verify stratigraphy is present
    assert "boreholes" in json_response
    assert isinstance(json_response["boreholes"], list)

    # Verify groundwater field is present (may be empty or populated)
    assert "groundwater" in json_response
    assert isinstance(json_response["groundwater"], list)

    # If groundwater found, verify structure
    if len(json_response["groundwater"]) > 0:
        gw = json_response["groundwater"][0]
        assert "depth" in gw
        assert "date" in gw  # May be null
        assert "elevation" in gw  # May be null
        assert "bounding_box" in gw
        assert "page_number" in gw["bounding_box"]
        assert isinstance(gw["depth"], int | float)

        # Verify bounding box structure
        bbox = gw["bounding_box"]
        assert "x0" in bbox and "y0" in bbox and "x1" in bbox and "y1" in bbox


def test_extract_stratigraphy_with_groundwater_missing_png(test_client: TestClient, upload_test_pdf):
    """Test that include_groundwater=True fails gracefully when PNG missing."""
    request = ExtractStratigraphyRequest(filename=TEST_PDF_KEY, include_groundwater=True)
    response = test_client.post("/api/V1/extract_stratigraphy", content=request.model_dump_json())
    assert response.status_code == 404
    assert "PNG file" in response.json()["detail"]
