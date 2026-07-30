"""Claude/Google batch result collection: batch-price cost recording,
per-item errors with detail, missing-item fill, thinking-token billing,
harvest idempotence. Provider clients are faked — no network."""

from types import SimpleNamespace

import pytest

from llm_utils import LLMModel, is_mechanism_error
from llm_utils.llm_services import ClaudeService, GoogleService

KEY = {"api_key": "test-key-not-real"}


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------

def _claude_ok_entry(cid, text, in_tok=100, out_tok=200):
    return SimpleNamespace(
        custom_id=cid,
        result=SimpleNamespace(
            type="succeeded",
            message=SimpleNamespace(
                content=[SimpleNamespace(text=text)],
                usage=SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok),
            ),
        ),
    )


def _claude_service_with(entries, status="ended"):
    svc = ClaudeService(LLMModel.CLAUDE_SONNET_5, **KEY)
    batch = SimpleNamespace(id="batch_test", processing_status=status)
    svc.client = SimpleNamespace(messages=SimpleNamespace(batches=SimpleNamespace(
        results=lambda bid: iter(entries),
        retrieve=lambda bid: batch,
    )))
    return svc, batch


class TestClaudeCollect:
    def test_success_records_batch_price(self):
        svc, batch = _claude_service_with([_claude_ok_entry("a", "ALPHA")])
        results = svc._collect_results(batch, is_test=False)
        assert results == {"a": "ALPHA"}
        expected = (
            100 * svc.model.input_price + 200 * svc.model.output_price
        ) / 1_000_000 * svc.BATCH_COST_DISCOUNT
        assert svc.total_usage.cost == pytest.approx(expected)

    def test_errored_entry_carries_detail(self):
        entry = SimpleNamespace(
            custom_id="b",
            result=SimpleNamespace(type="errored",
                                   error={"type": "invalid_request"}),
        )
        svc, batch = _claude_service_with([entry])
        results = svc._collect_results(batch, is_test=False)
        assert is_mechanism_error(results["b"])
        assert "invalid_request" in results["b"]

    def test_multi_text_blocks_are_joined(self):
        entry = _claude_ok_entry("a", "part1")
        entry.result.message.content.append(SimpleNamespace(text=" part2"))
        # a thinking block (no .text attr) must be skipped
        entry.result.message.content.insert(
            0, SimpleNamespace(thinking="hmm"))
        svc, batch = _claude_service_with([entry])
        results = svc._collect_results(batch, is_test=False)
        assert results["a"] == "part1 part2"

    def test_harvest_is_usage_idempotent_in_process(self):
        svc, batch = _claude_service_with([_claude_ok_entry("a", "ALPHA")])
        first = svc.harvest_batch_chat("batch_test")
        cost_after_first = svc.total_usage.cost
        assert first == [("a", "ALPHA")]
        assert cost_after_first > 0
        second = svc.harvest_batch_chat("batch_test")
        assert second == [("a", "ALPHA")]
        assert svc.total_usage.cost == cost_after_first  # no double-billing

    def test_harvest_returns_none_while_running(self):
        svc, _ = _claude_service_with([], status="in_progress")
        assert svc.harvest_batch_chat("batch_test") is None


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------

def _g_resp(text, in_tok=100, out_tok=200, thoughts=0):
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=in_tok,
            candidates_token_count=out_tok,
            thoughts_token_count=thoughts,
        ),
    )


def _g_job(responses, state="JOB_STATE_SUCCEEDED"):
    return SimpleNamespace(
        name="batches/test",
        state=SimpleNamespace(name=state),
        dest=SimpleNamespace(inlined_responses=responses),
    )


class TestGoogleCollect:
    def test_success_records_batch_price_including_thoughts(self):
        svc = GoogleService(LLMModel.GEMINI_2_5_FLASH, **KEY)
        job = _g_job([SimpleNamespace(
            response=_g_resp("ALPHA", 100, 200, thoughts=50), error=None)])
        results = svc._collect_results(job, ["a"], is_test=False)
        assert results == [("a", "ALPHA")]
        expected = (
            100 * svc.model.input_price + 250 * svc.model.output_price
        ) / 1_000_000 * svc.BATCH_COST_DISCOUNT
        assert svc.total_usage.cost == pytest.approx(expected)

    def test_missing_tail_becomes_mechanism_error(self):
        svc = GoogleService(LLMModel.GEMINI_2_5_FLASH, **KEY)
        job = _g_job([SimpleNamespace(response=_g_resp("ALPHA"), error=None)])
        results = svc._collect_results(job, ["a", "b"], is_test=False)
        assert results[0] == ("a", "ALPHA")
        assert results[1][0] == "b"
        assert is_mechanism_error(results[1][1])

    def test_failed_job_with_no_dest_yields_all_errors(self):
        svc = GoogleService(LLMModel.GEMINI_2_5_FLASH, **KEY)
        job = SimpleNamespace(
            name="batches/test",
            state=SimpleNamespace(name="JOB_STATE_FAILED"),
            dest=None,
        )
        results = svc._collect_results(job, ["a", "b"], is_test=False)
        assert all(is_mechanism_error(text) for _, text in results)
        assert "JOB_STATE_FAILED" in results[0][1]

    def test_item_error_detail_included(self):
        svc = GoogleService(LLMModel.GEMINI_2_5_FLASH, **KEY)
        job = _g_job([SimpleNamespace(response=None,
                                      error={"code": 13, "message": "boom"})])
        results = svc._collect_results(job, ["a"], is_test=False)
        assert is_mechanism_error(results[0][1])
        assert "boom" in results[0][1]

    def test_realtime_usage_includes_thoughts(self):
        from llm_utils.llm_services.google_service import _usage_tokens
        in_tok, out_tok = _usage_tokens(SimpleNamespace(
            prompt_token_count=10, candidates_token_count=20,
            thoughts_token_count=30))
        assert (in_tok, out_tok) == (10, 50)
