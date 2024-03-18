""" Temporary script for line detection"""

import os
from collections import deque
from typing import Dict

import cv2
import fitz
import numpy as np
from dotenv import load_dotenv
from numpy.typing import ArrayLike

from stratigraphy import DATAPATH
from stratigraphy.util.dataclasses import Line
from stratigraphy.util.geometric_line_utilities import (
    drop_vertical_lines,
    merge_parallel_lines_approximately,
    merge_parallel_lines_efficiently,
)
from stratigraphy.util.plot_utils import plot_lines
from stratigraphy.util.textblock import TextBlock
from stratigraphy.util.util import flatten, line_from_array, read_params

load_dotenv()

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled


line_detection_params = read_params("line_detection_params.yml")


def detect_lines_lsd(page: fitz.Page, scale_factor=2, lsd_params={}) -> ArrayLike:
    """Given a file path, detect lines in the pdf using the Line Segment Detector (LSD) algorithm.

    Publication of the algorithm can be found here: http://www.ipol.im/pub/art/2012/gjmr-lsd/article.pdf
    Note: As of now the function only works for pdfs with a single page.
          For now the function displays each pdf with the lines detected using opencv.
          This behavior will be changed in the future.

    Args:
        page (fitz.Page): The page to detect lines in.
        scale_factor (float, optional): The scale factor to scale the pdf page. Defaults to 2.

    Returns:
        list[Line]: The lines detected in the pdf.
    """
    pix = page.get_pixmap(matrix=fitz.Matrix(scale_factor, scale_factor))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create default line segment detector
    lsd = cv2.createLineSegmentDetector(**lsd_params)
    #  Documentation for the parameters can be found here:
    #  https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#gae0bba3b867a5f44d1b823aef4f57ee8d

    # Detect lines in the image
    lines = lsd.detect(gray)[0]
    return [line_from_array(line, scale_factor) for line in lines]


def extract_lines(page: fitz.Page, line_detection_params: Dict) -> list[Line]:
    lines = detect_lines_lsd(
        page,
        lsd_params=line_detection_params["lsd"],
        scale_factor=line_detection_params["pdf_scale_factor"],
    )
    lines = drop_vertical_lines(lines, threshold=line_detection_params["vertical_lines_threshold"])
    merging_params = line_detection_params["line_merging_params"]
    if merging_params["use_clustering"]:
        lines = merge_parallel_lines_approximately(
            lines,
            tol=merging_params["merging_tolerance"],
            eps=merging_params["clustering_threshold"],
            angle_threshold=merging_params["angle_threshold"],
        )

    else:
        lines = merge_parallel_lines_efficiently(
            lines, tol=merging_params["merging_tolerance"], angle_threshold=merging_params["angle_threshold"]
        )
    return lines


if __name__ == "__main__":
    # Some test pdfs
    selected_pdfs = [
        "270124083-bp.pdf",
        "268124307-bp.pdf",
        "268125268-bp.pdf",
        "267125378-bp.pdf",
        "268124435-bp.pdf",
        "267123060-bp.pdf",
        "268124635-bp.pdf",
        "675230002-bp.pdf",
        "268125592-bp.pdf",
        "267124070-bp.pdf",
    ]

    if mlflow_tracking:
        import mlflow

        mlflow.set_experiment("LineDetection")
        mlflow.start_run()
        mlflow.log_params(flatten(line_detection_params))
    lines = {}
    for pdf in selected_pdfs:
        doc = fitz.open(DATAPATH / "Benchmark" / pdf)

        for page in doc:
            lines[pdf] = extract_lines(page, line_detection_params)
            img = plot_lines(page, lines[pdf], scale_factor=line_detection_params["pdf_scale_factor"])
            if mlflow_tracking:
                mlflow.log_image(img, f"lines_{pdf}.png")
