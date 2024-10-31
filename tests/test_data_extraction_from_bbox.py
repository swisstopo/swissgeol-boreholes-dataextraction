"""Tests for the extract_data endpoint.

To see the objects that are being created before the tests are run, you can look at tests/conftest.py. The
test_client fixture is created by the TestClient(app) call, which creates a test client for the FastAPI app.
The s3_client fixture is created by the mock_aws decorator, which mocks the AWS S3 client using Moto.

"""

import json
from pathlib import Path

import fitz
import pytest
from app.common.aws import load_pdf_from_aws
from app.common.config import config
from app.common.schemas import ExtractDataRequest, FormatTypes
from fastapi.testclient import TestClient

TEST_PDF_KEY = Path("sample.pdf")
TEST_PDF_PATH = Path(__file__).parent.parent / "example" / "example_borehole_profile.pdf"
TEST_PNG_KEY = Path("dataextraction/sample-1.png")
TEST_PNG_PATH = Path(__file__).parent.parent / "example" / "sample-1.png"

TEST_ROTATED_PNG_KEY = Path("dataextraction/16132-1.png")
TEST_ROTATED_PNG_PATH = Path(__file__).parent.parent / "example" / "16132-1.png"
TEST_ROTATED_PDF_KEY = Path("16132.pdf")
TEST_ROTATED_PDF_PATH = Path(__file__).parent.parent / "example" / "16132.pdf"  # Rotated PDF of 270 degrees

TEST_CLIPPING_BEHAVIOR_PDF_PATH = Path(__file__).parent.parent / "example" / "clipping_test.pdf"
TEST_CLIPPING_BEHAVIOR_PDF_KEY = Path("clipping_test.pdf")
TEST_CLIPPING_BEHAVIOR_PNG_PATH = Path(__file__).parent.parent / "example" / "clipping_test-1.png"
TEST_CLIPPING_BEHAVIOR_PNG_KEY = Path("dataextraction/clipping_test-1.png")


def get_default_small_coordinate_request():
    """Return a default ExtractDataRequest for coordinates."""
    return ExtractDataRequest(
        filename=TEST_PDF_KEY.name,
        page_number=1,
        bbox={"x0": 0, "y0": 0, "x1": 100, "y1": 100},
        format=FormatTypes.COORDINATES,
    )


def get_default_coordinate_request():
    """Return a default ExtractDataRequest for coordinates."""
    return ExtractDataRequest(
        filename=TEST_PDF_KEY.name,
        page_number=1,
        bbox={"x0": 0, "y0": 0, "x1": 3000, "y1": 3000},
        format=FormatTypes.COORDINATES,
    )


def get_text_request_on_rotated_pdf():
    """Return a default ExtractDataRequest for text on a rotated PDF."""
    return ExtractDataRequest(
        filename=TEST_ROTATED_PDF_KEY.name,
        page_number=1,
        bbox={"x0": 0, "y0": 0, "x1": 600, "y1": 150},
        format=FormatTypes.TEXT,
    )


@pytest.fixture(scope="function")
def upload_test_pdf(s3_client):
    """Upload a test PDF file to S3."""
    s3_client.upload_file(Filename=str(TEST_PDF_PATH), Bucket=config.test_bucket_name, Key=str(TEST_PDF_KEY))
    s3_client.upload_file(
        Filename=str(TEST_ROTATED_PDF_PATH), Bucket=config.test_bucket_name, Key=str(TEST_ROTATED_PDF_KEY)
    )
    s3_client.upload_file(
        Filename=str(TEST_CLIPPING_BEHAVIOR_PDF_PATH),
        Bucket=config.test_bucket_name,
        Key=str(TEST_CLIPPING_BEHAVIOR_PDF_KEY),
    )


@pytest.fixture(scope="function")
def upload_test_png(s3_client, upload_test_pdf):
    """Upload a test PNG file to S3."""
    s3_client.upload_file(Filename=str(TEST_PNG_PATH), Bucket=config.test_bucket_name, Key=str(TEST_PNG_KEY))
    s3_client.upload_file(
        Filename=str(TEST_ROTATED_PNG_PATH), Bucket=config.test_bucket_name, Key=str(TEST_ROTATED_PNG_KEY)
    )
    s3_client.upload_file(
        Filename=str(TEST_CLIPPING_BEHAVIOR_PNG_PATH),
        Bucket=config.test_bucket_name,
        Key=str(TEST_CLIPPING_BEHAVIOR_PNG_KEY),
    )


def test_load_pdf_from_aws(upload_test_pdf):
    """Test loading a PDF from mocked AWS S3."""
    pdf_document = load_pdf_from_aws(Path(TEST_PDF_KEY.name))
    assert pdf_document is not None
    assert isinstance(pdf_document, fitz.Document)


def test_extract_coordinate_fail(test_client: TestClient, upload_test_pdf, upload_test_png):
    """Test the extract_data endpoint with a valid request."""
    request = get_default_small_coordinate_request()
    response = test_client.post("/api/V1/extract_data", content=request.model_dump_json())
    assert response.status_code == 404
    json_response = response.json()
    assert "detail" in json_response
    assert json_response["detail"] == "Coordinates not found."


def test_extract_text_success(test_client: TestClient, upload_test_pdf, upload_test_png):
    """Test the extract_data endpoint with a valid request."""
    ####################################################################################################
    ### Extract Data on Normal PDF
    ####################################################################################################
    target_text = (
        "BLS AlpTransit AG Lötschberg-Basislinie Sondierbohrung : SST KB 5 "
        "Bauherrschaft: BLS AlpTransit "
        "Bohrfirma : jet injectobohr AG "
        "Bohrmeister : Dragnic "
        "Ausführungsdatum 2.-3. 9. 1995 "
        "Koordinaten : 615 790 / 157 500 "
        "Kote Bezugspunkt: ~788,6 m ü. M."
    )

    request = ExtractDataRequest(
        filename=TEST_PDF_KEY.name,
        page_number=1,
        bbox={"x0": 0, "y0": 0, "x1": 1000, "y1": 1000},
        format=FormatTypes.TEXT,
    )
    response = test_client.post("/api/V1/extract_data", content=request.model_dump_json())
    assert response.status_code == 200
    json_response = response.json()
    assert "bbox" in json_response
    assert json_response["text"] == target_text

    ####################################################################################################
    ### Extract Data on Rotated PDF
    ####################################################################################################
    target_text = "OUVRAGE EMPLACEMENT Le Mégaror Lancy ENTREPRISE ISR Injectobohr SA"
    request = get_text_request_on_rotated_pdf()
    response = test_client.post("/api/V1/extract_data", content=request.model_dump_json())
    assert response.status_code == 200
    json_response = response.json()
    assert "bbox" in json_response
    assert json_response["text"] == target_text


def test_clipping_behavior(test_client: TestClient, upload_test_pdf, upload_test_png):
    """Test the extract_data endpoint with a valid request."""
    ####################################################################################################
    ### Extract Data on Normal PDF with bounding box with all text inside
    ####################################################################################################
    target_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut"

    request = ExtractDataRequest(
        filename=TEST_CLIPPING_BEHAVIOR_PDF_KEY.name,
        page_number=1,
        bbox={"x0": 311, "y0": 269, "x1": 821, "y1": 704},  # pixels
        format=FormatTypes.TEXT,
    )
    response = test_client.post("/api/V1/extract_data", content=request.model_dump_json())
    assert response.status_code == 200
    json_response = response.json()
    assert "bbox" in json_response
    assert json_response["text"] == target_text

    ####################################################################################################
    ### Extract Data on Normal PDF with bounding box with text on the boundary (e.g., the bounding box line is on the
    ### text)
    ####################################################################################################
    target_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut"

    request = ExtractDataRequest(
        filename=TEST_CLIPPING_BEHAVIOR_PDF_KEY.name,
        page_number=1,
        bbox={"x0": 311, "y0": 299, "x1": 813, "y1": 704},  # pixels
        format=FormatTypes.TEXT,
    )
    response = test_client.post("/api/V1/extract_data", content=request.model_dump_json())
    assert response.status_code == 200
    json_response = response.json()
    assert "bbox" in json_response
    assert json_response["text"] == target_text

    ####################################################################################################
    ### Extract Data on Normal PDF with bounding box with only a few words selected out of the text.
    ### Here the text was done with multiple Text Boxes (e.g., each line is a different Text Box).
    ####################################################################################################
    target_text = "Lorem ipsum"

    request = ExtractDataRequest(
        filename=TEST_CLIPPING_BEHAVIOR_PDF_KEY.name,
        page_number=1,
        bbox={"x0": 311, "y0": 269, "x1": 611, "y1": 336},  # pixels
        format=FormatTypes.TEXT,
    )
    response = test_client.post("/api/V1/extract_data", content=request.model_dump_json())
    assert response.status_code == 200
    json_response = response.json()
    assert "bbox" in json_response
    assert json_response["text"] == target_text

    ####################################################################################################
    ### Extract Data on Normal PDF with bounding box with only a few words selected out of the text.
    ### Here the text was done with one Text Box.
    ####################################################################################################
    target_text = "Lorem ipsum"

    request = ExtractDataRequest(
        filename=TEST_CLIPPING_BEHAVIOR_PDF_KEY.name,
        page_number=1,
        bbox={"x0": 1848, "y0": 242, "x1": 2145, "y1": 303},  # pixels
        format=FormatTypes.TEXT,
    )
    response = test_client.post("/api/V1/extract_data", content=request.model_dump_json())
    assert response.status_code == 200
    json_response = response.json()
    assert "bbox" in json_response
    assert json_response["text"] == target_text

    ####################################################################################################
    ### Extract Data on Normal PDF with bounding box with only one part of one word selected out of the text.
    ####################################################################################################
    target_text = "Lo"

    request = ExtractDataRequest(
        filename=TEST_CLIPPING_BEHAVIOR_PDF_KEY.name,
        page_number=1,
        bbox={"x0": 315, "y0": 281, "x1": 371, "y1": 330},  # pixels
        format=FormatTypes.TEXT,
    )
    response = test_client.post("/api/V1/extract_data", content=request.model_dump_json())
    assert response.status_code == 200
    json_response = response.json()
    assert "bbox" in json_response
    assert json_response["text"] == target_text

    target_text = "Lorem"

    request = ExtractDataRequest(
        filename=TEST_CLIPPING_BEHAVIOR_PDF_KEY.name,
        page_number=1,
        bbox={"x0": 315, "y0": 300, "x1": 465, "y1": 330},  # pixels
        format=FormatTypes.TEXT,
    )
    response = test_client.post("/api/V1/extract_data", content=request.model_dump_json())
    assert response.status_code == 200
    json_response = response.json()
    assert "bbox" in json_response
    assert json_response["text"] == target_text


def test_extract_text_empty(test_client: TestClient, upload_test_pdf, upload_test_png):
    """Test the extract_data endpoint with a valid request."""
    request = ExtractDataRequest(
        filename=TEST_PDF_KEY.name,
        page_number=1,
        bbox={"x0": 0, "y0": 0, "x1": 100, "y1": 100},
        format=FormatTypes.TEXT,
    )
    response = test_client.post("/api/V1/extract_data", content=request.model_dump_json())
    assert response.status_code == 404
    json_response = response.json()
    assert "detail" in json_response
    assert json_response["detail"] == "Text not found."


def test_extract_coordinate_success(test_client: TestClient, upload_test_pdf, upload_test_png):
    """Test the extract_data endpoint with a valid request."""
    ####################################################################################################
    ### Extract Data on Normal PDF with LV03 coordinates
    ####################################################################################################
    request = get_default_coordinate_request()
    response = test_client.post("/api/V1/extract_data", content=request.model_dump_json())
    assert response.status_code == 200
    json_response = response.json()
    assert "bbox" in json_response
    assert "coordinates" in json_response
    assert json_response["coordinates"]["east"] == 615790
    assert json_response["coordinates"]["north"] == 157500
    assert json_response["coordinates"]["projection"] == "LV03"

    ####################################################################################################
    ### Extract Data on Rotated PDF with LV03 coordinates
    ####################################################################################################
    request = ExtractDataRequest(
        filename=TEST_CLIPPING_BEHAVIOR_PDF_KEY.name,
        page_number=1,
        bbox={"x0": 1625, "y0": 900, "x1": 2819, "y1": 968},  # pixels
        format=FormatTypes.COORDINATES,
    )
    response = test_client.post("/api/V1/extract_data", content=request.model_dump_json())
    assert response.status_code == 200
    json_response = response.json()
    assert "bbox" in json_response
    assert "coordinates" in json_response
    assert json_response["coordinates"]["east"] == 684592.0
    assert json_response["coordinates"]["north"] == 252857.0
    assert json_response["coordinates"]["projection"] == "LV03"

    ####################################################################################################
    ### Extract Data on Rotated PDF with LV95 coordinates
    ####################################################################################################
    request = ExtractDataRequest(
        filename=TEST_CLIPPING_BEHAVIOR_PDF_KEY.name,
        page_number=1,
        bbox={"x0": 1625, "y0": 1000, "x1": 2819, "y1": 1068},  # pixels
        format=FormatTypes.COORDINATES,
    )
    response = test_client.post("/api/V1/extract_data", content=request.model_dump_json())
    assert response.status_code == 200
    json_response = response.json()
    assert "bbox" in json_response
    assert "coordinates" in json_response
    assert json_response["coordinates"]["east"] == 2682834.0
    assert json_response["coordinates"]["north"] == 1253400.0
    assert json_response["coordinates"]["projection"] == "LV95"


def test_incomplete_request(test_client: TestClient, upload_test_pdf, upload_test_png):
    """Test the extract_data endpoint with an incomplete request."""
    request = get_default_coordinate_request()
    request_json = json.loads(request.model_dump_json())
    del request_json["bbox"]
    response = test_client.post("/api/V1/extract_data", json=request_json)
    assert response.status_code == 400
    assert response.json() == {"detail": "bbox field - Field required"}


def test_page_number_out_of_range(test_client: TestClient, upload_test_pdf, upload_test_png):
    """Test the extract_data endpoint with an out-of-range page number."""
    request = get_default_coordinate_request()
    request_json = json.loads(request.model_dump_json())
    request_json["page_number"] = 2
    response = test_client.post("/api/V1/extract_data", json=request_json)
    assert response.status_code == 400
    assert response.json() == {"detail": "page not in document"}

    request_json["page_number"] = 0
    response = test_client.post("/api/V1/extract_data", json=request_json)
    assert response.status_code == 400
    assert response.json() == {"detail": "Page number must be a positive integer"}


def test_invalid_format(test_client: TestClient, upload_test_pdf, upload_test_png):
    """Test the extract_data endpoint with an invalid format."""
    request = get_default_coordinate_request()
    request_json = json.loads(request.model_dump_json())
    request_json["format"] = "invalid"
    response = test_client.post("/api/V1/extract_data", json=request_json)
    assert response.status_code == 400
    assert response.json() == {"detail": "format field - Input should be 'text', 'number' or 'coordinates'"}


def test_invalid_bbox(test_client: TestClient, upload_test_pdf, upload_test_png):
    """Test the extract_data endpoint with an invalid bounding box."""
    request = get_default_coordinate_request()
    request_json = json.loads(request.model_dump_json())
    request_json["bbox"] = {"x0": 0, "y0": 0, "x1": 100, "y1": -100.0}
    response = test_client.post("/api/V1/extract_data", json=request_json)
    assert response.status_code == 400
    assert response.json() == {"detail": "Bounding box coordinates must be positive"}


def test_invalid_pdf(test_client: TestClient, upload_test_pdf, upload_test_png):
    """Test the extract_data endpoint with an invalid PDF."""
    request = get_default_coordinate_request()
    request_json = json.loads(request.model_dump_json())
    request_json["filename"] = "invalid.pdf"
    response = test_client.post("/api/V1/extract_data", json=request_json)
    assert response.status_code == 404
    assert response.json() == {"detail": "Document invalid.pdf not found in S3 bucket."}


def test_number_extraction(test_client: TestClient, upload_test_pdf, upload_test_png):
    """Test the extract_data endpoint with a valid request."""
    ####################################################################################################
    ### Extract Data on Normal PDF
    ####################################################################################################
    request = get_default_coordinate_request()
    request.format = FormatTypes.NUMBER

    response = test_client.post("/api/V1/extract_data", content=request.model_dump_json())
    assert response.status_code == 200
    json_response = response.json()
    assert "bbox" in json_response
    assert "number" in json_response
    assert json_response["number"] == 5  # from STT KB5

    ####################################################################################################
    ### Extract Data on Rotated PDF
    ####################################################################################################
    request = get_text_request_on_rotated_pdf()
    request.format = FormatTypes.NUMBER
    request.bbox = {"x0": 0, "y0": 0, "x1": 300, "y1": 300}

    response = test_client.post("/api/V1/extract_data", content=request.model_dump_json())
    assert response.status_code == 200
    json_response = response.json()
    assert "bbox" in json_response
    assert "number" in json_response
    assert json_response["number"] == 0.0


def test_number_extraction_failure(test_client: TestClient, upload_test_pdf, upload_test_png):
    """Test the extract_data endpoint with a valid request."""
    request = ExtractDataRequest(
        filename=TEST_PDF_KEY.name,
        page_number=1,
        bbox={"x0": 0, "y0": 850, "x1": 1000, "y1": 950},  # Line with the coordinates in the document
        format=FormatTypes.NUMBER,
    )

    response = test_client.post("/api/V1/extract_data", content=request.model_dump_json())
    assert response.status_code == 400
    json_response = response.json()
    assert "detail" in json_response
    assert json_response["detail"] == "Multiple numbers found in the text."
