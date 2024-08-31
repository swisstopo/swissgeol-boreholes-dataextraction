# tests/test_create_pngs.py
from pathlib import Path

import pytest
from app.common.config import config
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient

TEST_PDF_KEY = "test-pdfs/sample.pdf"
TEST_PDF_PATH = Path(__file__).parent / "example" / "example_borehole_profile.pdf"


@pytest.fixture(scope="module")
def upload_test_pdf(s3_client):
    s3_client.upload_file(Filename=str(TEST_PDF_PATH), Bucket=config.bucket_name, Key=TEST_PDF_KEY)


def test_create_pngs_success(test_client: TestClient, s3_client, upload_test_pdf):
    response = test_client.post("/create_pngs", data={"filename": TEST_PDF_KEY})
    assert response.status_code == 200
    json_response = response.json()
    assert "png_urls" in json_response
    assert len(json_response["png_urls"]) > 0

    # Verify that PNG files are uploaded to S3
    for png_url in json_response["png_urls"]:
        png_key = png_url.split("/", 3)[-1]
        try:
            s3_client.head_object(Bucket=config.bucket_name, Key=png_key)
        except ClientError:
            pytest.fail(f"PNG file {png_key} not found in S3.")


def test_create_pngs_invalid_filename(test_client: TestClient):
    response = test_client.post("/create_pngs", data={"filename": ""})
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid request. 'filename' parameter is required and must be a string."}


def test_create_pngs_nonexistent_pdf(test_client: TestClient):
    response = test_client.post("/create_pngs", data={"filename": "nonexistent.pdf"})
    assert response.status_code == 404
    assert response.json() == {"detail": "PDF document not found in S3 bucket."}
