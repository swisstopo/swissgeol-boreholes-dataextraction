"""Script to download the borehole profiles from the S3 bucket."""

import os
from pathlib import Path

import boto3
import click
from tqdm import tqdm

from stratigraphy import DATAPATH


@click.command()
@click.option("--bucket-name", default="stijnvermeeren-boreholes-data", help="The name of the bucket.")
@click.option(
    "--remote-directory-name",
    default="data_v2/validation",
    help="The name of the directory in the bucket to be downloaded.",
)
@click.option(
    "--output-path", default=DATAPATH, type=click.Path(path_type=Path), help="The path to save the downloaded files."
)
def download_directory_froms3(
    bucket_name: str = "stijnvermeeren-boreholes-data",
    remote_directory_name: str = "data_v2/validation",
    output_path: Path = DATAPATH,
):
    """Download a directory from S3 bucket.

    Donwloads and saves the folder to disk.

    \f
    Args:
        bucketName (str): The name of the bucket.
        remoteDirectoryName (str): The name of the directory in the bucket to be downloaded.
    """  # noqa: D301
    s3_resource = boto3.resource("s3")
    bucket = s3_resource.Bucket(bucket_name)
    total_files = sum(1 for _ in bucket.objects.filter(Prefix=remote_directory_name))  # this is fast
    for obj in tqdm(bucket.objects.filter(Prefix=remote_directory_name), total=total_files):
        Path(output_path / obj.key).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(os.path.dirname(obj.key)):
            os.makedirs(os.path.dirname(obj.key))
        bucket.download_file(obj.key, output_path / obj.key)  # save to same path


if __name__ == "__main__":
    download_directory_froms3()
