"""This script converts pdf files to png files."""

from pathlib import Path

import click
import fitz
from tqdm import tqdm


@click.command()
@click.option("--input-directory", type=click.Path(path_type=Path), help="The directory containing the pdf files.")
@click.option("--output-directory", type=click.Path(path_type=Path), help="The directory to save the png files.")
def convert_pdf_to_png(input_directory: Path, output_directory: Path):
    """Convert pdf files to png files.

    Args:
        input_directory (Path): The input directory containing the pdf files.
        output_directory (Path): The output directory to save the png files.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    for pdf_file in tqdm(input_directory.glob("*.pdf")):
        document = fitz.open(pdf_file)
        for page_number in range(document.page_count):
            page = document.load_page(page_number)
            image = page.get_pixmap(matrix=fitz.Matrix(3, 3))
            image.save(output_directory / f"{pdf_file.stem}_{page_number}.png")


if __name__ == "__main__":
    convert_pdf_to_png()
