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
