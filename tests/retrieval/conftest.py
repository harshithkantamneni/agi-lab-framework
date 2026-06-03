"""Make the retrieval test package hermetic.

Model-dependent tests must never reach the Hugging Face Hub: the weights are
cached locally and the lab is local-only by constraint. A live network call on
import/load adds ~90s+ of latency and variance (and "unauthenticated request"
warnings), which makes these tests slow and flaky. Setting the offline env at
session start also propagates to any subprocess (server/worker) the tests spawn.
"""
import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _offline_models():
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ.setdefault(key, "1")
    yield
