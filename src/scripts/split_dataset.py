"""Split input folder to train. validation, and test folders."""

import hashlib
import logging
import shutil
from pathlib import Path

import click

logger = logging.getLogger(__name__)


def deterministic_hash_ratio(text: str) -> float:
    """Get hash ratio from input string.

    Args:
        text (str): String to hash.

    Returns:
        float: Hashed string to ratio [0, 1[.
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # Take first 8 hexa pairs and divide it by range (16^16 == 2^64)
    return int.from_bytes(h[:8]) / 16**16


@click.command()
@click.option("-i", "--input-directory", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output-directory", type=click.Path(path_type=Path))
@click.option("-rt", "--rtest", default=0.15, type=float)
@click.option("-rv", "--rvalid", default=0.15, type=float)
def split_data(input_directory: Path, output_directory: Path, rtest: float, rvalid: float) -> None:
    """Split input files and gt to train, valid and test.

    Converts an input directory containing N files into train, validation, and test folders, using the
    ratios specified via input flags.

    The split is deterministic and based on a hash of each filename. As a result, adding new files to
    the input directory does not change the assignment of previously processed files. Ranges are defines as
    test: '[0, rtest[', valid: '[rtest, rtest+rvalid[', and train: '[rtest+rvalid, 1['

    input_directory
    ├── file_1.pdf
    ├── ...
    └── file_n.pdf

    output_directory
    ├── train
    │   ├── file_1.pdf
    │   └── ...
    ├── validation
    │   ├── file_2.pdf
    │   └── ...
    └── test
        ├── file_3.pdf
        └── ...

    Args:
        input_directory (Path): Input directory containing *.pdf files
        output_directory (Path): Ouptut durectory to stor splits to.
        rtest (float): Fraction of test data. Defaults to 0.15.
        rvalid (float): Fraction of validation data. Defaults to 0.15.
    """
    # Check that valid and test are coherent
    assert rvalid + rtest < 1

    # Read input files
    files = [file.name for file in input_directory.iterdir() if file.suffix == ".pdf"]
    files_ratio = [deterministic_hash_ratio(file) for file in files]

    # Get split into sets.
    splits = {"train": [], "validation": [], "test": []}
    for file, ratio in zip(files, files_ratio, strict=True):
        if ratio < rtest:
            splits["test"].append(file)
        elif ratio < rtest + rvalid:
            splits["validation"].append(file)
        else:
            splits["train"].append(file)

    logger.info(
        "Files train: {}, validation: {}, test: {}".format(
            len(splits["train"]), len(splits["validation"]), len(splits["test"])
        )
    )

    # Prepare outputs
    for key in splits:
        # Create output directory
        (output_directory / key).mkdir(parents=True, exist_ok=True)
        # Write to output
        [shutil.copyfile(input_directory / file, output_directory / key / file) for file in splits[key]]

    logger.info("Done.")


if __name__ == "__main__":
    split_data()
