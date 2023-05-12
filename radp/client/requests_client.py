# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import requests

from radp.client.utils import add_params_to_url


class RequestsClient:
    """Generic requests client"""

    def __init__(self, ip: str, port: int):
        # TODO: validate that ip and port provided have valid format
        self.endpoint = f"http://{ip}:{port}"

    def _send_get_request(self, path: str, url_params: Dict) -> requests.Response:
        url = f"{self.endpoint}/{path}"

        # embed url params if present
        if url_params:
            url = add_params_to_url(url, url_params)
        return requests.get(url)

    def _send_post_request(self, path, **kwargs) -> requests.Response:
        url = f"{self.endpoint}/{path}"
        return requests.post(url, **kwargs)

    def _get_request_url(self, path: str, url_params: Dict) -> str:
        url = f"{self.endpoint}/{path}"

        # embed url params if present
        if url_params:
            url = add_params_to_url(url, url_params)
        return url
