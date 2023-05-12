# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict


def add_params_to_url(url: str, url_params: Dict) -> str:
    """Helper method to generate url with params embedded"""
    param_list = [f"{key}={val}" for key, val in url_params.items()]
    return url + f"?{'&'.join(param_list)}"
