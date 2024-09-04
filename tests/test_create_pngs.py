"""Tests for the create_pngs endpoint.

To see the objects that are being created before the tests are run, you can look at tests/conftest.py. The
test_client fixture is created by the TestClient(app) call, which creates a test client for the FastAPI app.
The s3_client fixture is created by the mock_aws decorator, which mocks the AWS S3 client using Moto.

"""

from pathlib import Path

import pytest
from app.common.config import config
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient

TEST_PDF_KEY = "pdfs/sample.pdf"
TEST_PDF_PATH = Path(__file__).parent.parent / "example" / "example_borehole_profile.pdf"
TEST_PNG_KEY = "pngs/sample-1.png"
TEST_PNG_PATH = Path(__file__).parent.parent / "example" / "sample-1.png"


@pytest.fixture(scope="function")
def upload_test_pdf(s3_client):
    """Upload a test PDF file to S3."""
    s3_client.upload_file(Filename=str(TEST_PDF_PATH), Bucket=config.test_bucket_name, Key=TEST_PDF_KEY)


def test_upload_png_to_s3(s3_client):
    """Test uploading a PNG file to mocked AWS S3."""
    # The method under test
    s3_client.upload_file(
        TEST_PNG_PATH,  # The local path to the file to upload
        config.bucket_name,  # The S3 bucket name (this will be mocked)
        TEST_PNG_KEY,  # The key (name) of the file in the bucket
    )

    # Assert that the file is in the test bucket
    response = s3_client.get_object(Bucket=config.test_bucket_name, Key=TEST_PNG_KEY)
    assert response is not None


def test_create_pngs_success(test_client: TestClient, s3_client, upload_test_pdf):
    """Test the create_pngs endpoint with a valid request."""
    response = test_client.post("/api/V1/create_pngs", json={"filename": TEST_PDF_KEY.rsplit("/", 1)[-1]})
    assert response.status_code == 200
    json_response = response.json()
    assert "png_urls" in json_response
    assert len(json_response["png_urls"]) > 0

    # Verify that PNG files are uploaded to S3
    for png_url in json_response["png_urls"]:
        png_key = png_url.split("/", 3)[-1]
        try:
            s3_client.head_object(Bucket=config.test_bucket_name, Key=png_key)
        except ClientError:
            pytest.fail(f"PNG file {png_key} not found in S3.")


def test_create_pngs_invalid_filename(test_client: TestClient):
    """Test the create_pngs endpoint with an invalid request."""
    response = test_client.post("/api/V1/create_pngs", json={"filename": ""})
    assert response.status_code == 422
    assert response.json() == {
        "detail": [
            {
                "loc": ["body", "filename"],
                "msg": "String should have at least 1 character",
                "type": "string_too_short",
                "ctx": {"min_length": 1},
                "input": "",
            }
        ]
    }


def test_create_pngs_nonexistent_pdf(test_client: TestClient):
    """Test the create_pngs endpoint with a nonexistent PDF file."""
    response = test_client.post("/api/V1/create_pngs", json={"filename": "nonexistent.pdf"})
    assert response.status_code == 404
    assert response.json() == {"detail": "Failed to load PDF document. The filename is not found in the bucket."}


def test_create_pngs_missing_pdf_extension(test_client: TestClient):
    """Test the create_pngs endpoint with a file not ending in .pdf."""
    response = test_client.post("/api/V1/create_pngs", json={"filename": "nonexistent"})
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid request. The filename must end with '.pdf'."}


def test_create_pngs_invalid_filename_format(test_client: TestClient):
    """Test the create_pngs endpoint with an invalid filename format."""
    response = test_client.post("/api/V1/create_pngs", json={"filename": "nonexistent.png"})
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid request. The filename must end with '.pdf'."}
