"""Suite-wide: the suite exercises Claude services, so it opts in to the paid Claude API
(tests never send real requests). test_claude_api_policy.py clears it to test the refusal."""
import pytest

from llm_utils.claude_api_policy import ALLOW_ENV


@pytest.fixture(autouse=True)
def _allow_claude_api(monkeypatch):
    monkeypatch.setenv(ALLOW_ENV, "1")
    # A developer's active broker must never receive requests from the suite.
    monkeypatch.delenv("LLM_UTILS_BROKER_URL", raising=False)
