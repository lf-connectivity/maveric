# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager
from os import fsync, replace
from uuid import uuid1


@contextmanager
def atomic_write(file_path: str, mode: str):
    """Write to unique temp file, then overwrite original"""
    try:
        tmp_file_path = f"{file_path}.{uuid1()}.tmp"
        file = open(tmp_file_path, mode)
        yield file
    finally:
        file.flush()
        fsync(file.fileno())
        file.close()
        replace(tmp_file_path, file_path)
