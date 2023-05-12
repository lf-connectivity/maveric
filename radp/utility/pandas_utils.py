# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import pandas as pd

from radp.common.helpers.file_system_safety import atomic_write

logger = logging.getLogger(__name__)


def read_feather_df(file_path: str) -> pd.DataFrame:
    """Read a pandas dataframe from feather format."""
    try:
        with open(file_path, "rb") as feather_file:
            return pd.read_feather(feather_file)
    except Exception as e:
        logger.exception(f"Exception occurred while reading {file_path}: {e}")
        raise e


def write_feather_df(file_path: str, df: pd.DataFrame) -> None:
    """Write a pandas dataframe to feather format file"""
    try:
        with atomic_write(file_path, "wb") as feather_file:
            df.to_feather(feather_file)
    except Exception as e:
        logger.exception(f"Exception occurred while writing to {file_path}: {e}")
        raise e


def cross_replicate(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """Cross replicate two pandas dataframes"""
    # raise exception if the dfs share a column name
    if any(df_a.columns.intersection(df_b.columns)):
        raise ValueError("Cannot call cross_replicate on dataframes with shared column names")

    size_a, size_b = df_a.shape[0], df_b.shape[0]

    # multiply df_a using concat
    df_a_multiplied = pd.concat([df_a] * size_b, ignore_index=True)

    # multiple df_b using repeat
    df_b_multiplied = pd.DataFrame(np.repeat(df_b.values, size_a, axis=0))
    df_b_multiplied.columns = df_b.columns

    return pd.concat([df_b_multiplied, df_a_multiplied], axis=1)
