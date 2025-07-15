"""Script to compare PDFs side by side."""

from pathlib import Path

import cv2
import numpy as np
import pymupdf


def pdf_to_images(pdf_path: Path, zoom: float = 2.0) -> list[np.ndarray]:
    """Convert all pages of a PDF into a list of images.

    Args:
        pdf_path (str or Path): Path to the PDF file.
        zoom (float): Zoom factor for rendering the PDF pages.

    Returns:
        list[np.ndarray]: List of images, one for each page in the PDF.
    """
    with pymupdf.open(pdf_path) as doc:
        images = []
        mat = pymupdf.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            img = np.frombuffer(pix.samples, dtype=np.uint8).copy().reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            images.append(img)
    return images


def pad_to_same_height(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pad two images to the same height by adding white borders.

    Args:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.

    Returns:
        tuple: Two images padded to the same height.
    """
    h1, h2 = img1.shape[0], img2.shape[0]
    max_h = max(h1, h2)
    img1_padded = cv2.copyMakeBorder(img1, 0, max_h - h1, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    img2_padded = cv2.copyMakeBorder(img2, 0, max_h - h2, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return img1_padded, img2_padded


def compare_pdfs(folder1: str, folder2: str) -> None:
    """Compare PDFs in two folders with the same filenames, displaying them side by side.

    Args:
        folder1 (str): Path to the first folder containing PDFs.
        folder2 (str): Path to the second folder containing PDFs.
    """
    folder1 = Path(folder1)
    folder2 = Path(folder2)

    files1 = {f.name for f in folder1.glob("*.pdf")}
    files2 = {f.name for f in folder2.glob("*.pdf")}
    common_files = sorted(files1 & files2)

    if not common_files:
        print("No matching PDF filenames found.")
        return

    for idx, filename in enumerate(common_files):
        if idx < 54:
            continue
        path1 = folder1 / filename
        path2 = folder2 / filename

        try:
            imgs1 = pdf_to_images(path1)
            imgs2 = pdf_to_images(path2)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        max_pages = max(len(imgs1), len(imgs2))
        page_num = 0

        while page_num < max_pages:
            img1 = imgs1[page_num] if page_num < len(imgs1) else np.full_like(imgs2[0], 255)
            img2 = imgs2[page_num] if page_num < len(imgs2) else np.full_like(imgs1[0], 255)

            img1, img2 = pad_to_same_height(img1, img2)
            combined = np.hstack((img1, img2))

            title = f"{filename} - Page {page_num + 1}/{max_pages} ({idx + 1}/{len(common_files)})"
            cv2.imshow(title, combined)

            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)

            if key in [ord("n"), 13]:  # Next page
                page_num += 1
            elif key == ord("s"):  # Skip document
                break
            elif key in [ord("q"), 27]:  # Quit
                return


if __name__ == "__main__":
    folder1 = "data/deepwells_s3"
    folder2 = "data/deepwells_deskewed"
    compare_pdfs(folder1.strip(), folder2.strip())
