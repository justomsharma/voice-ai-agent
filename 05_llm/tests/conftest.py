"""Every test runs as if no API key exists, whatever is in your real .env.

Tests must never read real credentials, let alone use them: a test that
"accidentally works" with your key would make paid calls and hide bugs.
"""

import pytest


@pytest.fixture(autouse=True)
def no_real_keys(monkeypatch, tmp_path):
    monkeypatch.setattr("llm.ENV_FILE", tmp_path / "no.env")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
