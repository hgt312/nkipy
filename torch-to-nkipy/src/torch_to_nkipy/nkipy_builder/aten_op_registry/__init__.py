# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .base import AtenOpRegistry

from . import batch_register

from pathlib import Path
import importlib

# Automatically import all modules that start with "aten_"
_current_dir = Path(__file__).parent
for _file in _current_dir.glob("aten_*.py"):
    importlib.import_module(f".{_file.stem}", package=__name__)
