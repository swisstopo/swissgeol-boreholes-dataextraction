"""Module for pdf modification trought image."""

import os
from pathlib import Path

import cv2
import fitz
import numpy as np


def deskew(doc: fitz.Document) -> fitz.Document:
    """Deskew a scanned PDF document by detecting and correcting slight skews in each page.

    Args:
        doc: Input fitz.Document object containing scanned pages

    Returns:
        New fitz.Document object with deskewed pages
    """
    new_doc = fitz.open()

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Get image directly as numpy array
        cv_img = _page_to_cv_image(page)

        # Deskew the image
        deskewed_img = _deskew_image(cv_img)

        # Insert deskewed image into new document
        _insert_cv_image_to_page(new_doc, deskewed_img, page.rect)

    return new_doc


def _page_to_cv_image(page: fitz.Page) -> np.ndarray:
    """Convert a fitz page to OpenCV image format."""
    # Render with 2x resolution for better quality
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)

    # Convert directly to numpy array (RGB format)
    img_array = np.frombuffer(pix.samples, dtype=np.uint8)
    img_array = img_array.reshape(pix.height, pix.width, pix.n)

    # Convert RGB to BGR for OpenCV
    if pix.n == 3:  # RGB
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    elif pix.n == 4:  # RGBA
        return cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    else:  # Grayscale
        return cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)


def _insert_cv_image_to_page(doc: fitz.Document, cv_img: np.ndarray, original_rect: fitz.Rect):
    """Insert OpenCV image into a new page in the document."""
    # Convert BGR back to RGB
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    # Convert to bytes
    success, img_encoded = cv2.imencode(".png", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError("Failed to encode image")

    img_bytes = img_encoded.tobytes()

    # Create new page and insert image
    new_page = doc.new_page(width=original_rect.width, height=original_rect.height)
    new_page.insert_image(new_page.rect, stream=img_bytes)


def _deskew_image(image: np.ndarray, croping_allowed: bool = False) -> np.ndarray:
    """Detect and correct skew in an image using OpenCV.

    Args:
        image: Input image as numpy array (BGR format)
        croping_allowed (bool): whether to crop the image and keep the same zoom level or the opposite.

    Returns:
        Deskewed image as numpy array
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply threshold to get binary image
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert if necessary (text should be white on black background for line detection)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    # Apply morphological operations to enhance line detection
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Detect lines using HoughLinesP
    lines = cv2.HoughLinesP(binary, rho=1, theta=np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)

    if lines is None or len(lines) == 0:
        # No lines detected, return original image
        return image

    # Calculate angles of all detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

        # Normalize angle to [-45, 45] range
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90

        # Only consider lines that are roughly horizontal (within ±45 degrees)
        if abs(angle) <= 45:
            angles.append(angle)

    if not angles:
        # No suitable lines found, return original image
        return image

    # Calculate the median angle (more robust than mean for outliers)
    skew_angle = np.median(angles)

    # Only correct if skew is significant (> 0.1 degrees)
    if abs(skew_angle) < 0.1:
        return image

    # Get image dimensions
    h, w = image.shape[:2]

    # Calculate rotation matrix
    center = (w // 2, h // 2)

    if croping_allowed:
        rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)

        # Calculate new bounding dimensions after rotation
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        new_w = int((h * sin_angle) + (w * cos_angle))
        new_h = int((h * cos_angle) + (w * sin_angle))

        # Adjust the rotation matrix to account for translation
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        # Apply rotation with white background
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        # Crop back to original size if the rotated image is larger
        if new_w > w or new_h > h:
            start_x = max(0, (new_w - w) // 2)
            start_y = max(0, (new_h - h) // 2)
            end_x = min(new_w, start_x + w)
            end_y = min(new_h, start_y + h)
            rotated = rotated[start_y:end_y, start_x:end_x]
    else:
        # Calculate scale factor to fit rotated content within original dimensions
        cos_angle = abs(np.cos(np.radians(skew_angle)))
        sin_angle = abs(np.sin(np.radians(skew_angle)))
        scale_factor = min(w / (w * cos_angle + h * sin_angle), h / (w * sin_angle + h * cos_angle))

        # Apply rotation with scaling to fit original dimensions
        rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, scale_factor)

        # Apply rotation with white background, keeping original dimensions
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

    return rotated


# Example usage:
if __name__ == "__main__":
    folder = "data/deepwells_test"
    output_folder = "data/deepwells_deskewed"

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process all PDF files in the folder
    folder_path = Path(folder)
    for file_path in folder_path.glob("*.pdf"):
        print(f"Processing: {file_path.name}")

        try:
            # Load PDF document
            input_doc = fitz.open(str(file_path))

            # Deskew the document
            deskewed_doc = deskew(input_doc)

            # Create output filename
            output_path = Path(output_folder) / f"deskewed_{file_path.name}"

            # Save the result
            deskewed_doc.save(str(output_path))
            print(f"Saved: {output_path.name}")

            # Close documents
            input_doc.close()
            deskewed_doc.close()

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    print("Processing complete!")
