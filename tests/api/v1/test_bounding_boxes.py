"""Tests for the bounding_boxes endpoint."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.common.schemas import BoundingBoxesRequest

# Get the project root directory (3 levels up from this file in tests/api/v1/)
PROJECT_ROOT = Path(__file__).parents[3]

# Reuse test constants from existing tests
TEST_PDF_KEY = "sample.pdf"
TEST_PDF_PATH = str(PROJECT_ROOT / "example" / "example_borehole_profile.pdf")
TEST_PNG_KEY = "dataextraction/sample-1.png"
TEST_PNG_PATH = str(PROJECT_ROOT / "example" / "sample-1.png")


@pytest.fixture(scope="function")
def upload_test_pdf(s3_client, test_bucket_name):
    """Upload a test PDF file to S3."""
    s3_client.upload_file(Filename=str(TEST_PDF_PATH), Bucket=test_bucket_name, Key=TEST_PDF_KEY)


@pytest.fixture(scope="function")
def upload_test_png(s3_client, test_bucket_name):
    """Upload a test PNG file to S3."""
    s3_client.upload_file(Filename=str(TEST_PNG_PATH), Bucket=test_bucket_name, Key=TEST_PNG_KEY)


def test_bounding_boxes_success(test_client: TestClient, upload_test_pdf, upload_test_png):
    """Test successful retrieval of bounding boxes."""
    request = BoundingBoxesRequest(filename=TEST_PDF_KEY, page_number=1)
    response = test_client.post("/api/V1/bounding_boxes", content=request.model_dump_json())
    assert response.status_code == 200
    json_response = response.json()
    assert "bounding_boxes" in json_response
    assert isinstance(json_response["bounding_boxes"], list)
    # Verify box structure
    if json_response["bounding_boxes"]:
        box = json_response["bounding_boxes"][0]
        assert all(key in box for key in ["x0", "y0", "x1", "y1"])


def test_bounding_boxes_nonexistent_pdf(test_client: TestClient):
    """Test request with nonexistent PDF file."""
    request = BoundingBoxesRequest(filename="nonexistent.pdf", page_number=1)
    response = test_client.post("/api/V1/bounding_boxes", content=request.model_dump_json())
    assert response.status_code == 404
    assert response.json() == {"detail": "Document nonexistent.pdf not found in S3 bucket."}

    # responses={
    #     400: {"model": BadRequestResponse, "description": "Bad request"},
    #     404: {
    #         "model": BadRequestResponse,
    #         "description": "Failed to load PDF document. The filename is not found in the bucket.",
    #     },
    #     500: {"model": BadRequestResponse, "description": "Internal server error"},
    # },


def test_bounding_boxes_invalid_page(test_client: TestClient, upload_test_pdf, upload_test_png):
    """Test request with invalid page number."""
    request = BoundingBoxesRequest(filename=TEST_PDF_KEY, page_number=999)  # Non-existent page
    response = test_client.post("/api/V1/bounding_boxes", content=request.model_dump_json())
    assert response.status_code == 400
    assert response.json() == {"detail": "page not in document"}


def test_bounding_boxes_invalid_filename(test_client: TestClient):
    """Test request with invalid filename."""
    # Test with non-PDF file
    request = BoundingBoxesRequest(filename="test.txt", page_number=1)
    response = test_client.post("/api/V1/bounding_boxes", content=request.model_dump_json())
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid request. The filename must end with '.pdf'."}
