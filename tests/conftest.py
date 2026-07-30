from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def upstream_png():
    distribution = importlib.metadata.distribution("pypng")
    source = Path(distribution.locate_file("png.py"))
    spec = importlib.util.spec_from_file_location("_upstream_pypng", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
