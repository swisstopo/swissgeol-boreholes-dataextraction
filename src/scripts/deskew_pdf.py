"""Module for pdf modification trought image."""

import os
from pathlib import Path

import cv2
import numpy as np
import pymupdf


def deskew_doc(doc: pymupdf.Document) -> pymupdf.Document:
    """Deskew a scanned PDF document by detecting and correcting slight skews in each page.

    Args:
        doc: Input pymupdf.Document object containing scanned pages

    Returns:
        New pymupdf.Document object with deskewed pages
    """
    new_doc = pymupdf.open()

    for page_num in range(len(doc)):
        page = doc[page_num]

        cv_img = _page_to_cv_image(page)

        deskewed_img = _deskew_image(cv_img)

        _insert_cv_image_to_page(new_doc, deskewed_img, page.rect)

    return new_doc


def _page_to_cv_image(page: pymupdf.Page) -> np.ndarray:
    """Convert a pymupdf page to OpenCV image format.

    Args:
        page: pymupdf.Page object to convert

    Returns:
        np.ndarray: Image in OpenCV format (BGR)
    """
    # Render with 2x resolution for better quality
    mat = pymupdf.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)

    # Convert directly to numpy array (RGB format)
    img_array = np.frombuffer(pix.samples, dtype=np.uint8)
    img_array = img_array.reshape(pix.height, pix.width, pix.n)

    if pix.n == 3:  # RGB
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    elif pix.n == 4:  # RGBA
        return cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    else:  # Grayscale
        return cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)


def _insert_cv_image_to_page(
    doc: pymupdf.Document, cv_img: np.ndarray, original_rect: pymupdf.Rect, jpeg_quality: int = 85
) -> None:
    """Insert OpenCV image into a new page in the document.

    Args:
        doc: pymupdf.Document object to insert the image into
        cv_img: OpenCV image in BGR format
        original_rect: Original rectangle dimensions for the new page
        jpeg_quality: JPEG quality for image encoding (default is 85)
    """
    # Convert BGR back to RGB
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    success, img_encoded = cv2.imencode(".jpg", rgb_img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not success:
        raise ValueError("Failed to encode image")

    img_bytes = img_encoded.tobytes()

    # Create new page and insert image
    new_page = doc.new_page(width=original_rect.width, height=original_rect.height)
    new_page.insert_image(new_page.rect, stream=img_bytes)


def _deskew_image(image: np.ndarray) -> np.ndarray:
    """Detect and correct skew in an image using OpenCV.

    First, the image is cropped from its black scanned background. This croped image is then used to determine the
    deskewed angle that should be applied to the original image.

    Args:
        image: Input image as numpy array (BGR format)

    Returns:
        Deskewed image as numpy array
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    largest_contour = find_outside_contour(gray)

    cropped = crop_outside(gray, largest_contour)
    skew_angle = find_lines_angle(cropped)
    # rotate the image to align the contour and match the image used for the skew detection
    image = rotate_out_contour(image, largest_contour)
    # rotate the image to align the text lines based on the skew angle
    image = rotate_image(skew_angle, image)
    return image


def find_outside_contour(gray: np.ndarray) -> np.ndarray | None:
    """Find the document out contour, which is likely the contour of the page on the background.

    This function finds the largest white contour present in the image.

    Args:
        gray: Input grayscale image as numpy array
    Returns:
        np.ndarray: Largest contour found in the image, or None if no contours are found
    """
    # Threshold to get binary image
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # remove noise and small artifacts like text with morphological operation
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours of the white page on the black background
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Returns the largest contour
    return max(contours, key=cv2.contourArea)


def crop_outside(gray: np.ndarray, largest_contour: np.ndarray | None) -> np.ndarray:
    """Crop the image to the bounding box of the largest contour.

    Args:
        gray: Input grayscale image as numpy array
        largest_contour: Largest contour found in the image, or None if no contours are found

    Returns:
        np.ndarray: Cropped image containing only the area inside the largest contour
    """
    original = gray.copy()
    if largest_contour is None:
        return original

    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Order the box points in top-left, top-right, bottom-right, bottom-left order
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    ordered_box = order_points(box)

    # Compute width and height of new image
    (tl, tr, br, bl) = ordered_box
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    # Destination points for perspective transform
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(ordered_box, dst)
    return cv2.warpPerspective(original, M, (maxWidth, maxHeight))


def rotate_image(angle: float, image: np.ndarray) -> np.ndarray:
    """Rotate an image by a specified angle while maintaining the original dimensions.

    Args:
        angle: Angle in degrees to rotate the image
        image: Input image as numpy array (BGR format)

    Returns:
        np.ndarray: Rotated image as numpy array
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    cos_angle = abs(np.cos(np.radians(angle)))
    sin_angle = abs(np.sin(np.radians(angle)))
    scale_factor = min(w / (w * cos_angle + h * sin_angle), h / (w * sin_angle + h * cos_angle))

    # Apply rotation with scaling to fit original dimensions
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale_factor)

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


def rotate_out_contour(gray: np.ndarray, largest_contour: np.ndarray | None, fill_white: bool = False) -> np.ndarray:
    """Rotate the image based on the largest contour found.

    If fill_white is True, the area outside the contour will be filled with white, otherwise it will be left unchanged.

    Args:
        gray: Input grayscale image as numpy array
        largest_contour: Largest contour found in the image, or None if no contours are found
        fill_white: If True, fill the area outside the contour with white

    Returns:
        np.ndarray: Rotated image as numpy array
    """
    if largest_contour is None:
        return gray.copy()
    original = gray.copy()
    if fill_white:
        original = fill_outside(original, largest_contour)
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[-1]
    # limit angle to +/- 45 deg
    if angle < -45:
        angle += 90

    if angle > 45:
        angle -= 90

    # Limit angle to avoid flipping
    if abs(angle) > 30:
        angle = 0  # Skip rotation

    rotated = rotate_image(angle, original)
    return rotated


def fill_outside(gray: np.ndarray, largest_contour: np.ndarray) -> np.ndarray:
    """Fill the area outside the largest contour with white.

    Args:
        gray: Input grayscale image as numpy array
        largest_contour: Largest contour found in the image

    Returns:
        np.ndarray: Image with the area outside the contour filled with white
    """
    # Read image
    original = gray.copy()

    # Create mask from the largest contour
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillPoly(mask, [largest_contour], 255)

    # Create result image - start with all white
    result = np.ones_like(gray) * 255

    # Copy original image only inside the mask
    result[mask > 0] = original[mask > 0]

    return result


def find_lines_angle(gray: np.ndarray) -> float:
    """Find the median angle of lines in a grayscale image using Hough Transform.

    This enhance horizontal text line detection by applying thresholding and morphologic closing.
    The Hough Transform is then applied to detect lines and calculates their angles.

    Args:
        gray: Input grayscale image as numpy array
    Returns:
        float: The median angle of detected lines in degrees, or 0 if no significant skew is found
    """
    smoothed = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(smoothed, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    binary = cv2.bitwise_not(binary)  # makes the text white on black background

    # Apply morphological operations to enhance text line detection
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))  # very horizontal
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # Detect lines using HoughLinesP
    lines = cv2.HoughLinesP(binary, rho=1, theta=np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)

    if lines is None or len(lines) == 0:
        return 0.0

    # too many lines usually means a grid is on the page, we can't rely on hugh transform to deskew.
    if len(lines) > 1000:
        print("Too many lines, skipped.")
        return 0.0

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

        # Only consider lines that are roughly horizontal (within ±20 degrees)
        if abs(angle) <= 20:
            angles.append(angle)

    if not angles:
        return 0.0

    skew_angle = np.median(angles)

    # Only correct if skew is significant (> 0.1 degrees) and reasonable (< 3 degreees)
    return skew_angle if 0.1 < abs(skew_angle) < 3 else 0.0


def is_digitally_born(doc: pymupdf.Document) -> bool:
    """Check if the PDF document is digitally born (not scanned).

    Args:
        doc: Input pymupdf.Document object

    Returns:
        bool: True if the document is digitally born, False if it is scanned
    """
    bboxes = [bbox for page in doc for bbox in page.get_bboxlog()]

    for boxType, rectangle in bboxes:
        if (boxType == "fill-text" or boxType == "stroke-text") and not pymupdf.Rect(rectangle).is_empty:
            print("Document is digitally born, no need to deskew.")
            return True
    return False


# Example usage:
if __name__ == "__main__":
    folder = "data/deepwells"
    output_folder = "data/deepwells_deskewed_hugh"
    # folder = "data/_test_solo"
    # folder = "data/_test_to_deskew"
    # output_folder = "data/_test_deskewed"

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process all PDF files in the folder
    folder_path = Path(folder)
    for file_path in folder_path.glob("*.pdf"):
        print(f"Processing: {file_path.name}")

        try:
            with pymupdf.open(file_path) as input_doc:
                # only try deskewing if the document was scanned
                deskewed_doc = input_doc if is_digitally_born(input_doc) else deskew_doc(input_doc)

                deskewed_doc.save(Path(output_folder) / file_path.name)
                deskewed_doc.close()
                print("Saved.")

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    print("Processing complete!")
