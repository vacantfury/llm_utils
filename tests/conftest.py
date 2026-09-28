"""Suite-wide: the suite exercises Claude services, so it opts in to the paid Claude API
(tests never send real requests). test_claude_api_policy.py clears it to test the refusal.
Each test starts with zero run spend, and the suite runs uncapped unless a test sets a cap:
recorded fake costs must not leak from one test into another's cap check."""
import pytest

from llm_utils import config, spend
from llm_utils.claude_api_policy import ALLOW_ENV


@pytest.fixture(autouse=True)
def _allow_claude_api(monkeypatch):
    monkeypatch.setenv(ALLOW_ENV, "1")
    # A developer's active broker must never receive requests from the suite.
    monkeypatch.delenv("LLM_UTILS_BROKER_URL", raising=False)


@pytest.fixture(autouse=True)
def _fresh_spend(monkeypatch):
    monkeypatch.setenv(spend.MAX_USD_ENV, "none")
    monkeypatch.delenv(config.CONFIG_ENV, raising=False)
    config.reload_config()
    spend.reset_run_spend()
    yield
    spend.reset_run_spend()
    config.reload_config()
