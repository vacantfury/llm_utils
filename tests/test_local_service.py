"""Public factory/local-service contracts, without torch, weights, or network."""

import sys
from types import SimpleNamespace

import pytest

from llm_utils import LLMModel, LLMServiceFactory, LocalLMService, Provider


@pytest.fixture
def hf(monkeypatch):
    calls = SimpleNamespace(tokenizer=[], model=[], template=[], pipeline=[], generate=[])

    class Tokenizer:
        chat_template = "test template"
        pad_token = None
        eos_token = "<eos>"
        pad_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            calls.template.append((messages, kwargs))
            return f"thinking={kwargs.get('enable_thinking', True)}:{messages}"

        def encode(self, text):
            return text.split()

    tokenizer = Tokenizer()

    def load_tokenizer(model_id, **kwargs):
        calls.tokenizer.append((model_id, kwargs))
        return tokenizer

    def load_model(model_id, **kwargs):
        calls.model.append((model_id, kwargs))
        model = SimpleNamespace(generation_config=SimpleNamespace())
        model.to = lambda device: model
        if "device_map" in kwargs:
            model.hf_device_map = {"": 0}
        return model

    def generate(prompts, **kwargs):
        calls.generate.append((prompts, kwargs))
        if isinstance(prompts, str):
            return [{"generated_text": "ready"}]
        return [[{"generated_text": "ready"}] for _ in prompts]

    def pipeline(task, **kwargs):
        calls.pipeline.append((task, kwargs))
        return generate

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        float16="test-float16", bfloat16="test-bfloat16",
        cuda=SimpleNamespace(is_available=lambda: True),
    ))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
        pipeline=pipeline,
    ))
    monkeypatch.setattr("llm_utils.llm_services.local_lm_service.TQDM_AVAILABLE", False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    LLMServiceFactory.clear_config_loader()
    yield calls, tokenizer
    LLMServiceFactory.clear_config_loader()


@pytest.mark.parametrize("size,name", [
    ("4B", "QWEN3_4B"),
    ("14B", "QWEN3_14B"), ("32B", "QWEN3_32B"),
])
def test_dense_qwen3_normal_factory_and_chat(hf, size, name):
    model_id = f"Qwen/Qwen3-{size}"
    model = LLMModel.from_string(model_id)
    assert model is LLMModel.from_string(name)
    assert model.provider is Provider.LOCAL
    assert model.family == "qwen"
    service = LLMServiceFactory.create(model_id, device="cpu", max_tokens=8)
    assert isinstance(service, LocalLMService)
    assert service.chat("Reply with ready.") == "ready"
    assert service.batch_chat([
        ("second", [("Reply with ready.", None)]),
        ("first", [("Reply with ready.", None)]),
    ], batch_size=2) == [("second", "ready"), ("first", "ready")]
    assert service.get_usage()["total"]["inference_count"] == 3


@pytest.mark.parametrize("size,name", [
    ("0.6B", "QWEN3_0_6B"), ("1.7B", "QWEN3_1_7B"),
    ("8B", "QWEN3_8B"),
])
def test_removed_dense_qwen3_models_reject_before_loading(hf, size, name):
    calls, _ = hf
    assert not hasattr(LLMModel, name)
    for model_string in (f"Qwen/Qwen3-{size}", name):
        with pytest.raises(ValueError, match="Unknown model"):
            LLMModel.from_string(model_string)
        with pytest.raises(ValueError, match="Unknown model"):
            LLMServiceFactory.create(model_string, device="cpu")
    assert not calls.tokenizer
    assert not calls.model


def test_explicit_load_options_reach_both_hf_loaders(hf):
    calls, _ = hf
    LLMServiceFactory.create(
        LLMModel.LLAMA3_2_1B, device="cuda", revision="fixed-revision",
        cache_dir="/model-cache", local_files_only=True, torch_dtype="bfloat16",
    )
    expected = {"token": None, "revision": "fixed-revision",
                "cache_dir": "/model-cache", "local_files_only": True}
    assert calls.tokenizer[0][1] == expected
    assert calls.model[0][1] == {
        **expected, "torch_dtype": "bfloat16", "device_map": "auto",
    }
    assert "device" not in calls.pipeline[0][1]


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_existing_loading_defaults_are_preserved(hf, device):
    calls, tokenizer = hf
    LLMServiceFactory.create(LLMModel.LLAMA3_2_1B, device=device)
    assert calls.tokenizer[0][1] == {"token": None}
    expected = {"token": None}
    if device in ("cuda", "mps"):
        expected["torch_dtype"] = "test-float16"
    if device == "cuda":
        expected["device_map"] = "auto"
        assert "device" not in calls.pipeline[0][1]
    else:
        assert calls.pipeline[0][1]["device"] == device
    assert calls.model[0][1] == expected
    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token == tokenizer.eos_token


def test_template_options_reach_chat_batch_and_do_not_leak_between_calls(hf):
    calls, _ = hf
    options = {"enable_thinking": False}
    service = LLMServiceFactory.create(
        LLMModel.LLAMA3_2_1B, device="cpu", chat_template_kwargs=options,
    )
    options["enable_thinking"] = True
    service.chat("one", system_message="system")
    service.batch_chat([("two", [("two", None)])],
                       chat_template_kwargs={"enable_thinking": True})
    service.chat("three")
    assert [kw["enable_thinking"] for _, kw in calls.template] == [False, True, False]
    assert calls.template[0][0][0] == {"role": "system", "content": "system"}
    for _, kwargs in calls.template:
        assert kwargs["tokenize"] is False
        assert kwargs["add_generation_prompt"] is True
    assert "thinking=False" in calls.generate[0][0][0]
    assert "thinking=True" in calls.generate[1][0][0]


@pytest.mark.parametrize("option", ["tokenize", "return_tensors", "return_dict", "conversation"])
def test_template_options_cannot_replace_pipeline_text_contract(hf, option):
    with pytest.raises(ValueError, match="chat_template_kwargs"):
        LLMServiceFactory.create(LLMModel.LLAMA3_2_1B, device="cpu",
                                 chat_template_kwargs={option: True})


@pytest.mark.parametrize("missing", [False, True])
def test_explicit_template_options_never_silently_fall_back(hf, missing):
    calls, tokenizer = hf
    service = LLMServiceFactory.create(LLMModel.LLAMA3_2_1B, device="cpu")
    if missing:
        tokenizer.chat_template = None
    else:
        def fail(*args, **kwargs):
            raise ValueError("broken template")
        tokenizer.apply_chat_template = fail
    with pytest.raises(ValueError, match="template"):
        service.chat("one", chat_template_kwargs={"enable_thinking": False})
    assert not calls.generate
