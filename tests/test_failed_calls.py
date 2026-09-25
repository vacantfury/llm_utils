"""Failed calls leave one call-ledger row (status error | timeout, an error
class, zero tokens and cost), recorded once per final outcome: a retry that
later succeeds records only its success. Provider clients are faked and a fake
agent_manager ledger is injected — no network, no keys, no real ledger."""

import asyncio
import sys
import types
from types import SimpleNamespace

import pytest

from llm_utils import LLMModel, call_ledger, is_mechanism_error
from llm_utils.base_llm_service import BaseLLMService
from llm_utils.exceptions import CreditsExhaustedError
from llm_utils.llm_services import (
    ClaudeService, GoogleService, OpenAIService, OpenRouterService,
    SlurmClusterService,
)
from llm_utils.llm_services.bedrock_service import BedrockService

KEY = {"api_key": "test-key-not-real"}


class APIConnectionError(Exception):
    """Stands in for an SDK transport error class."""


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    for mod in ("openai_service", "bedrock_service", "slurm_cluster_service"):
        monkeypatch.setattr(
            f"llm_utils.llm_services.{mod}._backoff_seconds", lambda *a, **k: 0)
    monkeypatch.setattr(
        "llm_utils.base_llm_service._backoff_seconds", lambda *a, **k: 0)
    monkeypatch.setattr("llm_utils.base_llm_service.time.sleep", lambda s: None)


@pytest.fixture(autouse=True)
def reset_hook_state():
    """Class-level hook state never leaks between tests."""
    yield
    BaseLLMService.clear_usage_hook()
    BaseLLMService._usage_hook_warned = False
    BaseLLMService._failure_hook_warned = False
    BaseLLMService._failure_hook_shape = None


@pytest.fixture()
def rows(monkeypatch):
    """The fake ledger's rows; no usage hook installed (default recorder)."""
    out = []
    pkg = types.ModuleType("agent_manager")
    ledger = types.ModuleType("agent_manager.ledger")
    ledger.record = lambda **fields: out.append(fields) or fields
    pkg.ledger = ledger
    monkeypatch.setitem(sys.modules, "agent_manager", pkg)
    monkeypatch.setitem(sys.modules, "agent_manager.ledger", ledger)
    call_ledger.census_id.cache_clear()
    call_ledger.repo_of.cache_clear()
    BaseLLMService.clear_usage_hook()
    yield out
    BaseLLMService.clear_usage_hook()
    call_ledger.census_id.cache_clear()
    call_ledger.repo_of.cache_clear()


def _failure(row, status, error_class):
    assert row["status"] == status
    assert row["error_class"] == error_class
    assert (row["input_tokens"], row["output_tokens"], row["cost_usd"]) == (0, 0, 0.0)


def _openai_response(text="ok", in_tok=10, out_tok=5):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text),
                                 finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=in_tok, completion_tokens=out_tok),
    )


def _openai_scripted(svc, *outcomes):
    """Replace create with a script: exceptions raise, anything else returns."""
    seq = list(outcomes)

    async def create(**_params):
        item = seq.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    svc.async_client.chat.completions.create = create


# ---------------------------------------------------------------------------
# OpenAI family (realtime; the compatible endpoints share this path)
# ---------------------------------------------------------------------------

class TestOpenAI:
    def test_transport_error_records_one_error_row(self, rows):
        svc = OpenAIService(LLMModel.GPT_5_NANO, max_retries=2, **KEY)
        _openai_scripted(svc, APIConnectionError("connection reset"))
        assert is_mechanism_error(svc.chat("hi"))
        assert len(rows) == 1
        _failure(rows[0], "error", "APIConnectionError")
        assert rows[0]["model"] == LLMModel.GPT_5_NANO.model_id
        assert svc.total_usage.inference_count == 0  # UsageStats count completions

    def test_call_timeout_records_timeout(self, rows):
        svc = OpenRouterService(LLMModel.OR_DEEPSEEK_V4_FLASH, call_timeout=0.05, **KEY)

        async def never(**_params):
            await asyncio.sleep(3600)

        svc.async_client.chat.completions.create = never
        assert is_mechanism_error(svc.chat("hi"))
        assert len(rows) == 1
        _failure(rows[0], "timeout", "TimeoutError")

    def test_retry_that_succeeds_records_one_ok_row(self, rows):
        svc = OpenAIService(LLMModel.GPT_5_NANO, max_retries=3, **KEY)
        _openai_scripted(svc, RuntimeError("429 rate limit"), _openai_response("fine"))
        assert svc.chat("hi") == "fine"
        assert [r["status"] for r in rows] == ["ok"]
        assert "error_class" not in rows[0]  # null in the real ledger

    def test_retries_exhausted_records_one_row(self, rows):
        svc = OpenAIService(LLMModel.GPT_5_NANO, max_retries=2, **KEY)
        _openai_scripted(svc, *[RuntimeError("429 rate limit")] * 3)
        assert is_mechanism_error(svc.chat("hi"))
        assert len(rows) == 1
        _failure(rows[0], "error", "RuntimeError")

    def test_account_fatal_records_then_aborts_without_retry(self, rows):
        svc = OpenAIService(LLMModel.GPT_5_NANO, max_retries=3, **KEY)
        _openai_scripted(svc, RuntimeError("429 insufficient_quota"),
                         _openai_response())
        with pytest.raises(CreditsExhaustedError):
            svc.chat("hi")
        assert len(rows) == 1
        _failure(rows[0], "error", "RuntimeError")

    def test_structured_failure_records_then_raises(self, rows):
        svc = OpenAIService(LLMModel.GPT_5_NANO, max_retries=0, **KEY)

        async def parse(**_params):
            raise APIConnectionError("boom")

        svc.async_client.chat.completions.parse = parse
        with pytest.raises(APIConnectionError):
            svc.chat_structured("hi", output_schema=dict)
        assert len(rows) == 1
        _failure(rows[0], "error", "APIConnectionError")

    def test_batch_items_errored_and_expired(self, rows):
        svc = OpenAIService(LLMModel.GPT_5_NANO, **KEY)
        out = [
            {"custom_id": "a", "response": {"status_code": 500, "body": {}}},
            {"custom_id": "b", "response": {"status_code": 200, "body": {
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                "choices": [{"message": {"content": "B"}}]}}},
        ]
        err = [{"custom_id": "c", "error": {"code": "batch_expired", "message": "x"}}]
        svc._download_jsonl = lambda fid: out if fid == "out" else err
        svc._cleanup_batch_files = lambda batch: None
        batch = SimpleNamespace(id="b1", status="expired", output_file_id="out",
                                error_file_id="err")
        results = svc._collect_results(batch, is_test=False)
        assert results["b"] == "B"
        by_status = sorted((r["status"], r.get("error_class")) for r in rows)
        assert by_status == [("error", "http_500"), ("ok", None),
                             ("timeout", "batch_expired")]


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class TestClaude:
    def test_realtime_error_records_one_row(self, rows):
        svc = ClaudeService(LLMModel.CLAUDE_SONNET_5, max_retries=1, **KEY)

        def create(**_params):
            raise APIConnectionError("reset")

        svc.client = SimpleNamespace(messages=SimpleNamespace(create=create))
        assert is_mechanism_error(svc.chat("hi"))
        assert len(rows) == 1
        _failure(rows[0], "error", "APIConnectionError")

    def test_sdk_timeout_class_records_timeout(self, rows):
        svc = ClaudeService(LLMModel.CLAUDE_SONNET_5, max_retries=0, **KEY)

        class APITimeoutError(Exception):
            pass

        def create(**_params):
            raise APITimeoutError("Request timed out.")

        svc.client = SimpleNamespace(messages=SimpleNamespace(create=create))
        assert is_mechanism_error(svc.chat("hi"))
        _failure(rows[0], "timeout", "APITimeoutError")

    def test_batch_items_and_harvest_idempotence(self, rows):
        entries = [
            SimpleNamespace(custom_id="a", result=SimpleNamespace(
                type="errored", error={"type": "invalid_request"})),
            SimpleNamespace(custom_id="b", result=SimpleNamespace(type="expired")),
        ]
        svc = ClaudeService(LLMModel.CLAUDE_SONNET_5, **KEY)
        batch = SimpleNamespace(id="bt", processing_status="ended")
        svc.client = SimpleNamespace(messages=SimpleNamespace(batches=SimpleNamespace(
            results=lambda bid: iter(entries), retrieve=lambda bid: batch)))
        svc.harvest_batch_chat("bt")
        svc.harvest_batch_chat("bt")  # same process: no second set of rows
        assert sorted((r["status"], r["error_class"]) for r in rows) == [
            ("error", "batch_errored"), ("timeout", "batch_expired")]


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------

class TestGoogle:
    def test_realtime_error_records_one_row(self, rows):
        svc = GoogleService(LLMModel.GEMINI_2_5_FLASH, max_retries=1, **KEY)

        def generate_content(**_kw):
            raise APIConnectionError("500 internal")

        svc.client = SimpleNamespace(models=SimpleNamespace(
            generate_content=generate_content))
        assert is_mechanism_error(svc.chat("hi"))
        assert len(rows) == 1
        _failure(rows[0], "error", "APIConnectionError")

    def test_expired_job_missing_items_record_timeouts(self, rows):
        svc = GoogleService(LLMModel.GEMINI_2_5_FLASH, **KEY)
        job = SimpleNamespace(name="j", state=SimpleNamespace(name="JOB_STATE_EXPIRED"),
                              dest=None)
        results = svc._collect_results(job, ["a", "b"], is_test=False)
        assert all(is_mechanism_error(t) for _, t in results)
        assert [(r["status"], r["error_class"]) for r in rows] == [
            ("timeout", "batch_expired")] * 2


# ---------------------------------------------------------------------------
# Bedrock and the SLURM cluster route (async per-call services)
# ---------------------------------------------------------------------------

def _bare_bedrock(max_retries):
    # Built without __init__: boto3 is an optional extra, absent in the test env.
    svc = BedrockService.__new__(BedrockService)
    BaseLLMService.__init__(svc, max_retries=max_retries)
    svc.model = LLMModel.BEDROCK_CLAUDE_HAIKU_4_5
    return svc


async def _bedrock_one(svc):
    return await svc._one_call(asyncio.Semaphore(1), [], None, 0.0, 16, False)


class TestBedrock:
    def test_throttle_then_success_records_one_ok_row(self, rows):
        svc = _bare_bedrock(max_retries=2)
        seq = [RuntimeError("ThrottlingException: slow down"),
               {"usage": {"inputTokens": 3, "outputTokens": 2},
                "output": {"message": {"content": [{"text": "hey"}]}},
                "stopReason": "end_turn"}]

        def converse(*_a):
            item = seq.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item

        svc._converse = converse
        assert asyncio.run(_bedrock_one(svc)) == "hey"
        assert [r["status"] for r in rows] == ["ok"]

    def test_error_records_one_row(self, rows):
        svc = _bare_bedrock(max_retries=2)

        def converse(*_a):
            raise APIConnectionError("ValidationException: bad input")

        svc._converse = converse
        assert is_mechanism_error(asyncio.run(_bedrock_one(svc)))
        assert len(rows) == 1
        _failure(rows[0], "error", "APIConnectionError")


class TestSlurm:
    def test_error_records_one_row(self, rows):
        manager = SimpleNamespace(acquire_endpoint=lambda m: "http://fake",
                                  release_endpoint=lambda m, e: None)
        svc = SlurmClusterService(LLMModel.LLAMA3_1_8B_CLUSTER,
                                  server_manager=manager, max_retries=1)

        async def create(**_params):
            raise APIConnectionError("connection refused")

        async def close():
            return None

        svc._make_async_client = lambda url: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            close=close)
        assert is_mechanism_error(svc.chat("hi"))
        assert len(rows) == 1
        _failure(rows[0], "error", "APIConnectionError")


# ---------------------------------------------------------------------------
# Usage hooks: old signatures are never called for a failure
# ---------------------------------------------------------------------------

class TestHooks:
    def test_old_signature_hook_skipped_and_recording_stays_on(self, rows):
        seen = []
        BaseLLMService.set_usage_hook(
            lambda model, i, o, c, *, is_test, label: seen.append((i, o, c)))
        BaseLLMService._usage_hook_warned = False
        svc = OpenAIService(LLMModel.GPT_5_NANO, max_retries=0, **KEY)
        _openai_scripted(svc, APIConnectionError("reset"), _openai_response(in_tok=7))
        assert is_mechanism_error(svc.chat("first"))
        assert seen == [] and rows == []  # the hook owns recording: no default row
        assert BaseLLMService._usage_hook_warned is False
        assert svc.chat("second") == "ok"
        assert seen == [(7, 5, pytest.approx(svc.total_usage.cost))]

    def test_new_signature_hook_receives_status(self, rows):
        seen = []

        def hook(model, i, o, c, *, is_test=False, label=None,
                 status="ok", error_class=None):
            seen.append((status, error_class, i, o, c, is_test))

        BaseLLMService.set_usage_hook(hook)
        svc = OpenAIService(LLMModel.GPT_5_NANO, max_retries=0, **KEY)
        _openai_scripted(svc, APIConnectionError("reset"), _openai_response())
        svc.chat("fails", is_test=True)
        svc.chat("works")
        assert seen[0] == ("error", "APIConnectionError", 0, 0, 0.0, True)
        assert seen[1][0] == "ok"  # completions keep the old call shape
        assert rows == []

    def test_kwargs_hook_receives_status(self, rows):
        seen = []
        BaseLLMService.set_usage_hook(lambda *a, **k: seen.append(k))
        svc = OpenAIService(LLMModel.GPT_5_NANO, max_retries=0, **KEY)
        svc._record_failure(TimeoutError())
        assert seen[0]["status"] == "timeout"
        assert seen[0]["error_class"] == "TimeoutError"

    def test_failing_new_hook_never_raises(self, rows):
        def hook(model, i, o, c, **kw):
            raise RuntimeError("ledger down")

        BaseLLMService.set_usage_hook(hook)
        BaseLLMService._usage_hook_warned = False
        svc = OpenAIService(LLMModel.GPT_5_NANO, max_retries=0, **KEY)
        _openai_scripted(svc, APIConnectionError("reset"))
        assert is_mechanism_error(svc.chat("hi"))  # no exception escaped
        assert BaseLLMService._usage_hook_warned is False

    def test_signature_read_once_per_hook(self, rows, monkeypatch):
        calls = []
        import llm_utils.base_llm_service as base

        real = base._hook_takes_outcome
        monkeypatch.setattr(base, "_hook_takes_outcome",
                            lambda h: calls.append(h) or real(h))
        hook = lambda *a, **k: None  # noqa: E731
        BaseLLMService.set_usage_hook(hook)
        svc = OpenAIService(LLMModel.GPT_5_NANO, **KEY)
        for _ in range(3):
            svc._record_failure("x")
        assert calls == [hook]


# ---------------------------------------------------------------------------
# Review fixes (2026-09-24): one row per outcome, whole-batch failures,
# repeat harvests, a recorder that never raises, timeout classification.
# ---------------------------------------------------------------------------

def _raises(exc):
    def fn(*_a, **_k):
        raise exc
    return fn


class TestOneRowPerOutcome:
    def test_openai_billed_response_without_choices_is_one_ok_row(self, rows):
        svc = OpenAIService(LLMModel.GPT_5_NANO, max_retries=0, **KEY)
        empty = SimpleNamespace(choices=[], usage=SimpleNamespace(
            prompt_tokens=10, completion_tokens=5))
        _openai_scripted(svc, empty)
        assert is_mechanism_error(svc.chat("hi"))
        assert [r["status"] for r in rows] == ["ok"]
        assert rows[0]["output_tokens"] == 5

    def test_bedrock_billed_response_with_bad_body_is_one_ok_row(self, rows):
        svc = _bare_bedrock(max_retries=0)
        svc._converse = lambda *_a: {
            "usage": {"inputTokens": 3, "outputTokens": 2},
            "output": {"message": {"content": None}}}
        assert is_mechanism_error(asyncio.run(_bedrock_one(svc)))
        assert [r["status"] for r in rows] == ["ok"]


class TestSubmitFailures:
    def test_openai_submit(self, rows):
        svc = OpenAIService(LLMModel.GPT_5_NANO, max_retries=0, **KEY)
        svc._sync_client = SimpleNamespace(files=SimpleNamespace(
            create=_raises(APIConnectionError("reset"))))
        with pytest.raises(APIConnectionError):
            svc.submit_batch_chat([("a", [("x", None)])])
        assert len(rows) == 1
        _failure(rows[0], "error", "APIConnectionError")

    def test_claude_submit(self, rows):
        svc = ClaudeService(LLMModel.CLAUDE_SONNET_5, max_retries=0, **KEY)
        svc.client = SimpleNamespace(messages=SimpleNamespace(batches=SimpleNamespace(
            create=_raises(APIConnectionError("reset")))))
        with pytest.raises(APIConnectionError):
            svc.submit_batch_chat([("a", [("x", None)])])
        assert len(rows) == 1
        _failure(rows[0], "error", "APIConnectionError")

    def test_google_submit(self, rows):
        svc = GoogleService(LLMModel.GEMINI_2_5_FLASH, max_retries=0, **KEY)
        svc.client = SimpleNamespace(batches=SimpleNamespace(
            create=_raises(APIConnectionError("reset"))))
        with pytest.raises(APIConnectionError):
            svc.submit_batch_chat([("a", [("x", None)])])
        assert len(rows) == 1
        _failure(rows[0], "error", "APIConnectionError")


class TestStructured:
    def test_claude_structured_failure_records_then_raises(self, rows):
        svc = ClaudeService(LLMModel.CLAUDE_SONNET_5, max_retries=0, **KEY)
        svc.client = SimpleNamespace(messages=SimpleNamespace(
            parse=_raises(APIConnectionError("reset"))))
        with pytest.raises(APIConnectionError):
            svc.chat_structured("hi", output_schema=dict)
        assert len(rows) == 1
        _failure(rows[0], "error", "APIConnectionError")


def _openai_failed_batch(total=3):
    return SimpleNamespace(
        id="bf", status="failed", output_file_id=None, error_file_id=None,
        input_file_id=None, errors=None,
        request_counts=SimpleNamespace(total=total, completed=0, failed=total))


class TestWholeBatchFailures:
    def test_openai_batch_chat_missing_ids(self, rows):
        svc = OpenAIService(LLMModel.GPT_5_NANO, use_batch_api=True, **KEY)
        batch = _openai_failed_batch(total=2)
        svc._submit_batch = lambda *a, **k: batch
        svc._poll_until_done = lambda b: b
        out = svc.batch_chat([("a", [("x", None)]), ("b", [("y", None)])])
        assert all(is_mechanism_error(t) for _, t in out)
        assert [(r["status"], r["error_class"]) for r in rows] == [
            ("error", "batch_failed")] * 2

    def test_openai_harvest_failed_batch_counts_requests_once(self, rows):
        svc = OpenAIService(LLMModel.GPT_5_NANO, **KEY)
        batch = _openai_failed_batch(total=3)
        svc._sync_client = SimpleNamespace(batches=SimpleNamespace(
            retrieve=lambda bid: batch))
        assert svc.harvest_batch_chat("bf") == []
        assert svc.harvest_batch_chat("bf") == []  # same process: no new rows
        assert [(r["status"], r["error_class"]) for r in rows] == [
            ("error", "batch_failed")] * 3

    def _google_job(self, svc, state, *, stats=None, inlined=None):
        job = SimpleNamespace(
            name="jobs/1", state=SimpleNamespace(name=state), src=None,
            completion_stats=stats,
            dest=SimpleNamespace(inlined_responses=inlined) if inlined else None)
        svc.client = SimpleNamespace(batches=SimpleNamespace(get=lambda name: job))
        return job

    def test_google_harvest_failed_job_uses_its_request_count(self, rows):
        svc = GoogleService(LLMModel.GEMINI_2_5_FLASH, **KEY)
        self._google_job(svc, "JOB_STATE_FAILED", stats=SimpleNamespace(
            successful_count=0, failed_count=4, incomplete_count=None))
        assert svc.harvest_batch_chat("jobs/1") == []
        assert [(r["status"], r["error_class"]) for r in rows] == [
            ("error", "batch_failed")] * 4

    def test_google_harvest_expired_job_without_count_records_one_timeout(self, rows):
        svc = GoogleService(LLMModel.GEMINI_2_5_FLASH, **KEY)
        self._google_job(svc, "JOB_STATE_EXPIRED")
        svc.harvest_batch_chat("jobs/1")
        assert [(r["status"], r["error_class"]) for r in rows] == [
            ("timeout", "batch_expired")]

    def test_google_repeat_harvest_records_once(self, rows):
        svc = GoogleService(LLMModel.GEMINI_2_5_FLASH, **KEY)
        ok = SimpleNamespace(response=SimpleNamespace(
            text="A", usage_metadata=SimpleNamespace(
                prompt_token_count=2, candidates_token_count=1,
                thoughts_token_count=0)))
        bad = SimpleNamespace(response=None, error={"code": 500})
        self._google_job(svc, "JOB_STATE_SUCCEEDED", inlined=[ok, bad])
        first = svc.harvest_batch_chat("jobs/1")
        second = svc.harvest_batch_chat("jobs/1")
        assert first == second and first[0] == ("0", "A")
        assert sorted((r["status"], r.get("error_class")) for r in rows) == [
            ("error", "batch_item_error"), ("ok", None)]


class TestRecorderNeverRaises:
    def test_shape_check_failure_is_the_old_shape(self, rows, monkeypatch):
        import llm_utils.base_llm_service as base

        seen = []
        monkeypatch.setattr(base, "_hook_takes_outcome", _raises(RuntimeError("odd")))
        BaseLLMService.set_usage_hook(lambda *a, **k: seen.append(k))
        svc = OpenAIService(LLMModel.GPT_5_NANO, **KEY)
        svc._record_failure(APIConnectionError("x"))  # must not raise
        assert seen == []


class TestTimeoutClassification:
    def test_google_deadline_exceeded(self, rows):
        class DeadlineExceeded(Exception):
            pass

        svc = OpenAIService(LLMModel.GPT_5_NANO, **KEY)
        svc._record_failure(DeadlineExceeded("504 Deadline Exceeded"))
        _failure(rows[0], "timeout", "DeadlineExceeded")

    @pytest.mark.parametrize("attr", ["status_code", "code"])
    def test_http_504_status(self, rows, attr):
        class ServerError(Exception):
            pass

        exc = ServerError("gateway timeout")
        setattr(exc, attr, 504)
        svc = OpenAIService(LLMModel.GPT_5_NANO, **KEY)
        svc._record_failure(exc)
        _failure(rows[0], "timeout", "ServerError")

    def test_other_status_codes_stay_error(self, rows):
        class ServerError(Exception):
            code = 503

        svc = OpenAIService(LLMModel.GPT_5_NANO, **KEY)
        svc._record_failure(ServerError("unavailable"))
        _failure(rows[0], "error", "ServerError")
