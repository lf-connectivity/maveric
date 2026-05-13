# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from api_manager.exceptions.validation_exception import ValidationException
from api_manager.validators.base_validator import SchemaValidator


class TestSchemaValidator(unittest.TestCase):
    def test_validate_valid_schema(self):
        validator = SchemaValidator(
            {
                "name": {"required": True, "type": str, "pattern": r"^[a-z_]+$"},
                "count": {"required": True, "type": int, "min": 1, "max": 10},
                "mode": {"required": False, "type": str, "enum": {"a", "b"}},
            },
            allow_unknown=False,
        )

        validator.validate({"name": "valid_name", "count": 5, "mode": "a"})

    def test_validate_accumulates_errors(self):
        validator = SchemaValidator(
            {
                "name": {"required": True, "type": str, "min_length": 3},
                "count": {"required": True, "type": int, "min": 1},
            },
            allow_unknown=False,
        )

        with self.assertRaises(ValidationException) as ctx:
            validator.validate({"name": "x", "count": 0, "extra": True})

        fields = {error["field"] for error in ctx.exception.validation_errors}
        self.assertEqual(fields, {"name", "count", "extra"})

    def test_validate_rejects_non_object(self):
        validator = SchemaValidator({"name": {"required": True, "type": str}})

        with self.assertRaises(ValidationException) as ctx:
            validator.validate(["not", "a", "dict"])

        self.assertEqual(ctx.exception.validation_errors[0]["field"], "request")
