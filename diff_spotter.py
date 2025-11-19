"""Check for differences in metrics."""

# from memory_profiler import profile

# from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

from fast_langdetect import LangDetectConfig, LangDetector


# @profile
def test():
    config = LangDetectConfig(max_input_length=1000)
    detector = LangDetector(config)
    language = detector.detect("jojlj", model="lite")[0].get("lang", None)
    print(language)


from pathlib import Path

import click
import pandas as pd
import json
import numpy as np


# @click.command()
# @click.option("--source-file", default="a.csv", help="Reference metric file")
# @click.option("--target-file", default="b.csv", help="Updated metric file")
# @click.option("--index", default="document_name", help="Index column name")
# def main(source_file: Path, target_file: Path, index: str):
#     """Compare two csv files for differences. Comparison if made on column ``.

#     Args:
#         source_file (Path): Source file before modifications.
#         target_file (Path): Target file after modifications.
#         index (str): Column to use as index.
#     """
#     # Laod data
#     df_src = pd.read_csv(source_file).set_index(index)
#     df_tar = pd.read_csv(target_file).set_index(index)

#     # Check entries that differs
#     matched = (df_src == df_tar).all(axis=1)

#     df_diff = df_src[~matched] - df_tar[~matched]

#     # Print results
#     print(df_diff)
#     for index, row in df_diff.iterrows():
#         print(f"{index}: \n{row[row != 0].to_dict()}")


# @click.command()
# @click.option("--source-file", default="geoquat_langdetect.json", help="Reference metric file")
# @click.option("--target-file", default="geoquat_fast_langdetect_lite_1000.json", help="Updated metric file")
# @click.option("--index", default="document_name", help="Index column name")
# def main(source_file: Path, target_file: Path, index: str):
#     """Compare two csv files for differences. Comparison if made on column ``.

#     Args:
#         source_file (Path): Source file before modifications.
#         target_file (Path): Target file after modifications.
#         index (str): Column to use as index.
#     """
#     # load data
#     with open(source_file, encoding="utf-8") as f:
#         data_a = json.load(f)
#         files_a = list(data_a.keys())

#     with open(target_file, encoding="utf-8") as f:
#         data_b = json.load(f)

#     pairs = [
#         (
#             data_a.get(key, {}).get("file_metadata", {}).get("language", ""),
#             data_b.get(key, {}).get("file_metadata", {}).get("language", ""),
#         )
#         for key in data_a
#     ]

#     correct = [la == lb for la, lb in pairs]
#     incorrect_files = np.array(files_a)[np.nonzero(~np.array(correct))[0]]
#     incorrect_pairs = np.array(pairs)[np.nonzero(~np.array(correct))[0]]
#     # Print results
#     # print(f"{len(correct)} Acc: {np.mean(correct):.3f}, Diff: {incorrect_files}, Pairs: {incorrect_pairs}")


if __name__ == "__main__":
    test()
