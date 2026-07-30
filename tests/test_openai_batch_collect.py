"""OpenAI batch result collection: per-item errors, missing-id fill,
batch-price cost recording, file cleanup. Provider calls are monkeypatched —
no network."""

from types import SimpleNamespace

import pytest

from llm_utils import LLMModel, is_mechanism_error
from llm_utils.llm_services import OpenAIService


@pytest.fixture()
def service():
    return OpenAIService(LLMModel.GPT_5_NANO, api_key="test-key-not-real")


def _batch(status="completed", output_file_id="out-1", error_file_id=None):
    return SimpleNamespace(
        id="batch_test",
        status=status,
        input_file_id="in-1",
        output_file_id=output_file_id,
        error_file_id=error_file_id,
        errors=None,
    )


def _ok_entry(cid, text, in_tok=10, out_tok=5):
    return {
        "custom_id": cid,
        "response": {
            "status_code": 200,
            "body": {
                "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok},
                "choices": [{"message": {"content": text},
                             "finish_reason": "stop"}],
            },
        },
    }


def _patch_files(monkeypatch, service, files: dict, deleted: list):
    monkeypatch.setattr(
        service, "_download_jsonl", lambda fid: files[fid])
    monkeypatch.setattr(
        service, "_cleanup_batch_files",
        lambda batch: deleted.append(
            (batch.input_file_id, batch.output_file_id, batch.error_file_id)))


class TestCollectResults:
    def test_success_records_batch_price(self, monkeypatch, service):
        deleted = []
        _patch_files(monkeypatch, service,
                     {"out-1": [_ok_entry("a", "ALPHA", 100, 200)]}, deleted)
        results = service._collect_results(_batch(), is_test=False)
        assert results == {"a": "ALPHA"}
        expected = (
            100 * service.model.input_price + 200 * service.model.output_price
        ) / 1_000_000 * service.BATCH_COST_DISCOUNT
        assert service.total_usage.cost == pytest.approx(expected)
        assert service.total_usage.inference_count == 1
        assert deleted  # cleanup ran

    def test_item_error_becomes_mechanism_error(self, monkeypatch, service):
        entry = {"custom_id": "b",
                 "response": {"status_code": 500, "body": {}},
                 "error": {"code": "server_error"}}
        _patch_files(monkeypatch, service, {"out-1": [entry]}, [])
        results = service._collect_results(_batch(), is_test=False)
        assert is_mechanism_error(results["b"])
        assert service.total_usage.inference_count == 0

    def test_error_file_entries_merge(self, monkeypatch, service):
        files = {
            "out-1": [_ok_entry("a", "ALPHA")],
            "err-1": [{"custom_id": "b", "error": {"code": "expired"}}],
        }
        _patch_files(monkeypatch, service, files, [])
        results = service._collect_results(
            _batch(error_file_id="err-1"), is_test=False)
        assert results["a"] == "ALPHA"
        assert is_mechanism_error(results["b"])

    def test_filtered_empty_content_gets_placeholder(self, monkeypatch, service):
        entry = _ok_entry("a", "")
        entry["response"]["body"]["choices"][0]["finish_reason"] = "content_filter"
        _patch_files(monkeypatch, service, {"out-1": [entry]}, [])
        results = service._collect_results(_batch(), is_test=False)
        assert "filtered out" in results["a"]

    def test_wholesale_failed_batch_returns_empty(self, monkeypatch, service):
        _patch_files(monkeypatch, service, {}, [])
        results = service._collect_results(
            _batch(status="failed", output_file_id=None), is_test=False)
        assert results == {}


class TestBatchChatMissingIdFill:
    def test_missing_ids_filled_with_mechanism_error(self, monkeypatch, service):
        service.use_batch_api = True
        monkeypatch.setattr(
            service, "_submit_batch", lambda *a, **k: _batch())
        monkeypatch.setattr(service, "_poll_until_done", lambda b: b)
        monkeypatch.setattr(
            service, "_collect_results", lambda b, is_test: {"a": "ALPHA"})
        out = service.batch_chat([("a", [("hi", None)]), ("b", [("yo", None)])])
        assert out[0] == ("a", "ALPHA")
        assert out[1][0] == "b"
        assert is_mechanism_error(out[1][1])
