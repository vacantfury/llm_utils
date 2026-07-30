"""Base-layer helpers: mechanism-error sentinel, error classifiers, backoff,
UsageStats. Pure functions — no network, no keys."""

from llm_utils.base_llm_service import (
    MECHANISM_ERROR_SENTINEL,
    UsageStats,
    _backoff_seconds,
    is_credit_exhausted_error,
    is_invalid_credential_error,
    is_mechanism_error,
    is_rate_limit_error,
    make_mechanism_error,
    strip_mechanism_error,
)


class TestMechanismError:
    def test_round_trip(self):
        wrapped = make_mechanism_error("connection reset")
        assert is_mechanism_error(wrapped)
        assert strip_mechanism_error(wrapped) == "connection reset"

    def test_plain_text_is_not_mechanism_error(self):
        assert not is_mechanism_error("I refuse to answer that.")
        assert not is_mechanism_error("")

    def test_non_string_is_not_mechanism_error(self):
        assert not is_mechanism_error(None)
        assert not is_mechanism_error(42)

    def test_strip_is_identity_on_plain_text(self):
        assert strip_mechanism_error("hello") == "hello"

    def test_sentinel_contains_null_byte(self):
        # The null byte is the collision guard against real model output.
        assert "\x00" in MECHANISM_ERROR_SENTINEL


class TestErrorClassifiers:
    def test_rate_limit_patterns(self):
        for msg in (
            "Error code: 429 - too many requests",
            "RESOURCE_EXHAUSTED: quota exceeded",
            "The server is currently overloaded",
            "Request was throttled",
        ):
            assert is_rate_limit_error(Exception(msg)), msg

    def test_ordinary_error_is_not_rate_limit(self):
        assert not is_rate_limit_error(Exception("connection reset by peer"))

    def test_invalid_credential_patterns(self):
        for msg in (
            "Incorrect API key provided: sk-...",
            "invalid x-api-key",
            "API key not valid. Please pass a valid API key.",
            "Error code: 401 - authentication_error",
        ):
            assert is_invalid_credential_error(Exception(msg)), msg

    def test_credit_exhausted_patterns(self):
        for msg in (
            "Error code: 429 - insufficient_quota",
            "You exceeded your current quota",
            "Your credit balance is too low",
            "Insufficient Balance",
            "Error code: 402 - payment required",
        ):
            assert is_credit_exhausted_error(Exception(msg)), msg

    def test_openai_billing_429_is_both(self):
        # OpenAI surfaces billing exhaustion as a 429: it matches the
        # rate-limit classifier TOO, which is why services must consult
        # is_credit_exhausted_error BEFORE the retry branch.
        exc = Exception("Error code: 429 - insufficient_quota")
        assert is_credit_exhausted_error(exc)
        assert is_rate_limit_error(exc)


class TestBackoff:
    def test_grows_then_caps(self):
        waits = [_backoff_seconds(a, max_wait=60.0) for a in range(10)]
        assert waits[0] < 10
        assert all(w <= 60.0 for w in waits)
        assert waits[-1] == 60.0


class TestUsageStats:
    def test_record_accumulates(self):
        s = UsageStats()
        s.record(100, 50, 0.01)
        s.record(10, 5, 0.001)
        assert s.inference_count == 2
        assert s.input_tokens == 110
        assert s.output_tokens == 55
        assert abs(s.cost - 0.011) < 1e-12

    def test_to_dict_shape(self):
        d = UsageStats().to_dict()
        assert set(d) == {"inference_count", "input_tokens", "output_tokens", "cost"}
