from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mock_api import MockSferenceServer


@pytest.fixture(scope="function")
def mock_sference_server() -> MockSferenceServer:
    server = MockSferenceServer()
    server.start()
    yield server
    server.stop()
