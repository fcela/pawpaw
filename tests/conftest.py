"""Shared pytest fixtures and config."""
from __future__ import annotations

import os
import pytest


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="set RUN_GPU_TESTS=1 to enable")
    if os.environ.get("RUN_GPU_TESTS") != "1":
        for item in items:
            if "integration" in str(item.fspath):
                item.add_marker(skip_gpu)
