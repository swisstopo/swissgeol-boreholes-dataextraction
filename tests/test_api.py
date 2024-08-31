# """This file contains the tests for the FastAPI endpoints.

# The tests are written using the pytest framework and the FastAPI TestClient.

# The tests cover the following scenarios:
# 1. Test the create_pngs endpoint with a valid request.
# 2. Test the create_pngs endpoint with an invalid request.
# 3. Test the extract_data endpoint with a valid request for extracting text.
# 4. Test the extract_data endpoint with an invalid format type.
# 5. Test the extract_data endpoint with a missing PDF file.
# """

# from app.common.config import config
# from app.main import app  # Replace with the actual path to your FastAPI app
# from fastapi import HTTPException
# from fastapi.testclient import TestClient

# client = TestClient(app)
# bucket_name = config.bucket_name  # Replace with the actual bucket name


# def test_create_pngs_valid_request():
#     """Test the create_pngs endpoint with a valid request."""
#     pdf_filename_s3 = "pdfs/geoquat/train/10012.pdf"
#     response = client.post("/create_pngs", data={"filename": pdf_filename_s3})
#     assert response.status_code == 200
#     assert response.json() == {"png_urls": [f"https://{bucket_name}.s3.amazonaws.com/pngs/geoquat/train/10012-1.png"]}


# # test for files with multiple pages
# def test_create_pngs_valid_request_multiple_pages():
#     """Test the create_pngs endpoint with a valid request for a PDF with multiple pages."""
#     response = client.post("/create_pngs", data={"filename": "pdfs/geoquat/train/298.pdf"})
#     assert response.status_code == 200
#     assert response.json() == {
#         "png_urls": [
#             f"https://{bucket_name}.s3.amazonaws.com/pngs/geoquat/train/298-1.png",
#             f"https://{bucket_name}.s3.amazonaws.com/pngs/geoquat/train/298-2.png",
#             f"https://{bucket_name}.s3.amazonaws.com/pngs/geoquat/train/298-3.png",
#             f"https://{bucket_name}.s3.amazonaws.com/pngs/geoquat/train/298-4.png",
#         ]
#     }


# def test_create_pngs_invalid_request():
#     """Test the create_pngs endpoint with an invalid request."""
#     response = client.post("/create_pngs", data={"filename": ""})
#     assert response.status_code == 400
#     assert response.json() == {"detail": "Invalid request. 'filename' parameter is required and must be a string."}


# def test_extract_data_valid_text_request():
#     """Test the extract_data endpoint with a valid request for extracting text."""
#     extract_data_request = {
#         "filename": "geoquat/train/10012",
#         "page_number": 0,
#         "bbox": {"x0": 10, "y0": 20, "x1": 30, "y1": 40},
#         "format": "TEXT",
#     }

#     response = client.post("/extract_data", json=extract_data_request)
#     assert response.status_code == 200
#     assert response.json() == {"bbox": {"x0": 10, "y0": 20, "x1": 30, "y1": 40}, "text": "Extracted text goes here"}


# def test_extract_data_invalid_bbox_request():
#     """Test the extract_data endpoint with a valid request for extracting coordinates."""
#     extract_data_request = {
#         "filename": "geoquat/train/10012",
#         "page_number": 0,
#         "bbox": {"x_min": 10, "y_min": 20, "x_max": 30, "y_max": 40},
#         "format": "TEXT",
#     }

#     response = client.post("/extract_data", json=extract_data_request)
#     assert response.status_code == 200
#     assert response.json() == {"bbox": {"x0": 10, "y0": 20, "x1": 30, "y1": 40}, "text": "Extracted text goes here"}


# def test_extract_data_zero_bound_page_request():
#     """Test the extract_data endpoint with a zero-bound page number."""
#     extract_data_request = {
#         "filename": "geoquat/train/10012",
#         "page_number": 0,
#         "bbox": {"x0": 10, "y0": 20, "x1": 30, "y1": 40},
#         "format": "TEXT",
#     }

#     response = client.post("/extract_data", json=extract_data_request)
#     assert response.status_code == 200
#     assert response.json() == {"bbox": {"x0": 10, "y0": 20, "x1": 30, "y1": 40}, "text": "Extracted text goes here"}


# def test_extract_data_invalid_format():
#     """Test the extract_data endpoint with an invalid format type."""
#     extract_data_request = {
#         "filename": "geoquat/train/10012",
#         "page_number": 0,
#         "bbox": {"x0": 10, "y0": 20, "x1": 30, "y1": 40},
#         "format": "INVALID_FORMAT",
#     }

#     response = client.post("/extract_data", json=extract_data_request)
#     assert response.status_code == 400
#     assert response.json() == {"detail": "Invalid format type."}


# def test_extract_data_missing_file():
#     """Test the extract_data endpoint with a missing PDF file."""
#     extract_data_request = {
#         "filename": "geoquat/train/invalid_file",
#         "page_number": 0,
#         "bbox": {"x0": 10, "y0": 20, "x1": 30, "y1": 40},
#         "format": "TEXT",
#     }

#     response = client.post("/extract_data", json=extract_data_request)
#     assert response.status_code == 404
#     assert response.json() == {"detail": "PDF document not found in S3 bucket."}


# def main():
#     test_create_pngs_valid_request()
#     test_create_pngs_valid_request_multiple_pages()
#     test_create_pngs_invalid_request()
#     test_extract_data_valid_text_request()
#     test_extract_data_invalid_bbox_request()
#     test_extract_data_zero_bound_page_request()
#     test_extract_data_invalid_format()
#     test_extract_data_missing_file()
