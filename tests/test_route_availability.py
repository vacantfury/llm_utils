"""Route availability seam (t10): logical model -> routes, and which routes this
process can use. Offline: probes run against a fake HTTP layer."""

import httpx2
import pytest

import llm_utils
from llm_utils import (
    LLMModel, LLMServiceFactory, Provider, RegistryRouteResolver, config,
    route_models, route_status, routes_for, usable_routes, route_statuses,
    logical_models,
)
from llm_utils import routes as routes_mod

_KEY_ENVS = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "DEEPSEEK_API_KEY",
             "ZAI_API_KEY", "OPENROUTER_API_KEY", "XAI_API_KEY", "MOONSHOT_API_KEY"]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in _KEY_ENVS:
        monkeypatch.delenv(k, raising=False)
    yield
    LLMServiceFactory.clear_server_manager()


class TestRegistryLookup:
    def test_logical_name_is_weights_else_model_id(self):
        assert LLMModel.GLM_5_2.logical_name == "glm-5.2"
        assert LLMModel.GPT_4O.logical_name == "gpt-4o"

    def test_every_route_group_is_the_route_twins_closure(self):
        for name, rows in logical_models().items():
            for m in rows:
                assert set(m.routes()) == set(rows), name

    def test_same_model_id_rows_share_a_logical_model(self):
        # Two rows serving the same id with no shared logical name would make
        # the id ambiguous here and split one model into two.
        by_id = {}
        for m in LLMModel:
            by_id.setdefault(m.model_id, set()).add(m.logical_name)
        assert {k: v for k, v in by_id.items() if len(v) > 1} == {}

    def test_routes_for_glm5_lists_every_host(self):
        assert routes_for("glm-5") == {"zai", "openrouter", "bedrock"}

    def test_lookup_forms_agree(self):
        by_logical = route_models("glm-5")
        assert route_models(LLMModel.OR_GLM_5) == by_logical
        assert route_models("OR_GLM_5") == by_logical
        assert route_models(LLMModel.OR_GLM_5.model_id) == by_logical

    def test_single_route_model(self):
        assert route_models("gpt-4o") == (LLMModel.GPT_4O,)
        assert routes_for(LLMModel.GPT_4O) == {"openai"}

    def test_model_id_shared_by_two_routes_resolves(self):
        # from_string refuses this id (two rows); the route lookup groups them.
        assert routes_for("meta-llama/Meta-Llama-3-8B-Instruct") == {"local", "slurm_cluster"}

    def test_unknown_model_raises_instead_of_empty(self):
        with pytest.raises(ValueError):
            routes_for("no-such-model")

    def test_resolver_object_form(self):
        r = RegistryRouteResolver()
        assert r.routes_for("glm-5") == routes_for("glm-5")

    def test_seam_exports(self):
        for name in ("routes_for", "route_status", "usable_routes", "RouteStatus",
                     "RegistryRouteResolver", "config"):
            assert name in llm_utils.__all__


class TestConfigCheck:
    def test_api_key_present_or_absent(self, monkeypatch):
        s = route_status(LLMModel.OR_GLM_5)
        assert not s.usable and "OPENROUTER_API_KEY" in s.reason and not s.probed
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        s = route_status(LLMModel.OR_GLM_5)
        assert s.usable and s.route == "openrouter" and s.jurisdiction == "us"

    def test_usable_routes_filters(self, monkeypatch):
        monkeypatch.setenv("ZAI_API_KEY", "k")
        monkeypatch.setattr(routes_mod, "_bedrock_credentials", lambda: (False, "none"))
        assert usable_routes("glm-5") == (LLMModel.GLM_5,)
        assert len(route_statuses("glm-5")) == 3

    def test_claude_api_opt_in_gates_the_anthropic_route(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        model = next(m for m in LLMModel if m.provider is Provider.ANTHROPIC)
        assert route_status(model).usable
        monkeypatch.delenv("LLM_UTILS_ALLOW_CLAUDE_API")
        s = route_status(model)
        assert not s.usable and "LLM_UTILS_ALLOW_CLAUDE_API" in s.reason

    def test_broker_mode_makes_api_routes_usable_without_keys(self, monkeypatch):
        monkeypatch.setenv("LLM_UTILS_BROKER_URL", "http://127.0.0.1:9")
        s = route_status(LLMModel.OR_GLM_5, probe=True)
        assert s.usable and "broker" in s.reason and not s.probed

    def test_slurm_needs_a_registered_endpoint_manager(self):
        assert not route_status(LLMModel.LLAMA3_8B_CLUSTER).usable
        LLMServiceFactory.set_server_manager(object())
        assert route_status(LLMModel.LLAMA3_8B_CLUSTER).usable

    def test_local_needs_the_local_extra(self, monkeypatch):
        monkeypatch.setattr(routes_mod.importlib.util, "find_spec", lambda name: None)
        s = route_status(LLMModel.LLAMA3_8B)
        assert not s.usable and "torch" in s.reason

    def test_broker_mode_with_an_invalid_url(self, monkeypatch):
        monkeypatch.setenv("LLM_UTILS_BROKER_URL", "")
        s = route_status(LLMModel.OR_GLM_5)
        assert not s.usable and "valid loopback URL" in s.reason

    def test_route_status_refuses_a_multi_route_logical_name(self):
        with pytest.raises(ValueError, match="route_statuses"):
            route_status("glm-5")
        assert route_status("OR_GLM_5").model is LLMModel.OR_GLM_5

    @pytest.fixture
    def no_aws(self, monkeypatch, tmp_path):
        # A stand-in boto3 (the bedrock extra is not installed in the suite).
        import sys
        import types
        boto3 = types.ModuleType("boto3")
        boto3.Session = lambda *a, **k: pytest.fail("resolved the credential chain")
        botocore = types.ModuleType("botocore")
        botocore_config = types.ModuleType("botocore.config")
        botocore_config.Config = lambda **k: k
        for name, mod in (("boto3", boto3), ("botocore", botocore),
                          ("botocore.config", botocore_config)):
            monkeypatch.setitem(sys.modules, name, mod)
        real = routes_mod.importlib.util.find_spec
        monkeypatch.setattr(routes_mod.importlib.util, "find_spec",
                            lambda name: object() if name == "boto3" else real(name))
        for env in routes_mod._AWS_CONFIG_ENVS:
            monkeypatch.delenv(env, raising=False)
        monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "none"))
        monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "none2"))
        return boto3

    def test_bedrock_config_only_never_resolves_credentials(self, no_aws, monkeypatch):
        s = route_status(LLMModel.BEDROCK_GLM_5)
        assert not s.usable and "probe=True" in s.reason
        monkeypatch.setenv("AWS_PROFILE", "research")
        s = route_status(LLMModel.BEDROCK_GLM_5)
        assert s.usable and "AWS_PROFILE" in s.reason

    def test_bedrock_credentials_file_counts(self, no_aws, monkeypatch, tmp_path):
        f = tmp_path / "credentials"
        f.write_text("[default]\n")
        monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(f))
        assert route_status(LLMModel.BEDROCK_GLM_5).usable

    def test_bedrock_probe_is_an_identity_call(self, no_aws, monkeypatch):
        boto3 = no_aws
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x")
        called = []

        class FakeSts:
            def get_caller_identity(self):
                called.append(True)
                return {"Account": "0"}

        class FakeSession:
            def __init__(self, *a, **k):
                pass

            def client(self, name, **k):
                assert name == "sts"
                return FakeSts()

        monkeypatch.setattr(boto3, "Session", FakeSession)
        s = route_status(LLMModel.BEDROCK_GLM_5, probe=True)
        assert s.usable and s.probed and called

    def test_bedrock_probe_failure_is_unusable(self, no_aws, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x")

        class Boom:
            def __init__(self, *a, **k):
                pass

            def client(self, *a, **k):
                raise RuntimeError("no route to STS")

        monkeypatch.setattr(no_aws, "Session", Boom)
        s = route_status(LLMModel.BEDROCK_GLM_5, probe=True)
        assert not s.usable and s.probed and "RuntimeError" in s.reason

    def test_bedrock_without_boto3(self, monkeypatch):
        real = routes_mod.importlib.util.find_spec
        monkeypatch.setattr(routes_mod.importlib.util, "find_spec",
                            lambda name: None if name == "boto3" else real(name))
        s = route_status(LLMModel.BEDROCK_GLM_5)
        assert not s.usable and "boto3" in s.reason


class TestProbe:
    @pytest.fixture
    def http(self, monkeypatch):
        calls = []

        def fake_get(url, headers=None, timeout=None):
            calls.append((url, headers, timeout))
            if isinstance(fake_get.answer, Exception):
                raise fake_get.answer
            return httpx2.Response(fake_get.answer, text=fake_get.body,
                                   request=httpx2.Request("GET", url))

        fake_get.answer = 200
        fake_get.body = ""
        monkeypatch.setattr(routes_mod.httpx2, "get", fake_get)
        fake_get.calls = calls
        return fake_get

    def test_no_probe_when_config_fails(self, http):
        s = route_status(LLMModel.OR_GLM_5, probe=True)
        assert not s.usable and http.calls == []

    def test_probe_hits_the_model_listing_with_the_key(self, http, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
        s = route_status(LLMModel.DEEPSEEK_V4_FLASH, probe=True)
        assert s.usable and s.probed
        url, headers, timeout = http.calls[0]
        assert url == "https://api.deepseek.com/models"
        assert headers["Authorization"] == "Bearer k"
        assert timeout == config.get("routes.probe_timeout_s")

    def test_openai_default_base(self, http, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        route_status(LLMModel.GPT_4O, probe=True)
        assert http.calls[0][0] == "https://api.openai.com/v1/models"

    @pytest.mark.parametrize("code,usable,word", [
        (401, False, "rejected"), (403, False, "rejected"), (402, False, "cannot pay"),
        (503, False, "endpoint error"), (404, True, "not confirmed"),
        (429, True, "rate-limited"), (400, True, "not confirmed")])
    def test_status_codes(self, http, monkeypatch, code, usable, word):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        http.answer = code
        s = route_status(LLMModel.OR_GLM_5, probe=True)
        assert s.usable is usable and word in s.reason

    def test_quota_429_is_unusable(self, http, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        http.answer = 429
        http.body = '{"error": {"code": "insufficient_quota"}}'
        s = route_status(LLMModel.GPT_4O, probe=True)
        assert not s.usable and "quota" in s.reason

    def test_google_400_is_an_invalid_key(self, http, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "bad")
        http.answer = 400
        gemini = next(m for m in LLMModel if m.provider is Provider.GOOGLE)
        s = route_status(gemini, probe=True)
        assert not s.usable and "rejected" in s.reason

    def test_openrouter_probes_the_per_key_endpoint(self, http, monkeypatch):
        # Its model listing is public, so it would not test the key.
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        route_status(LLMModel.OR_GLM_5, probe=True)
        assert http.calls[0][0] == "https://openrouter.ai/api/v1/key"

    def test_any_request_failure_is_unreachable(self, http, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        http.answer = UnicodeEncodeError("ascii", "é", 0, 1, "bad header")
        s = route_status(LLMModel.OR_GLM_5, probe=True)
        assert not s.usable and "unreachable" in s.reason

    def test_network_failure_is_unreachable(self, http, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        http.answer = httpx2.ConnectError("down")
        s = route_status(LLMModel.OR_GLM_5, probe=True, timeout=1.0)
        assert not s.usable and "unreachable" in s.reason

    def test_anthropic_and_google_probe_headers(self, http, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
        monkeypatch.setenv("GOOGLE_API_KEY", "g")
        claude = next(m for m in LLMModel if m.provider is Provider.ANTHROPIC)
        gemini = next(m for m in LLMModel if m.provider is Provider.GOOGLE)
        route_status(claude, probe=True)
        route_status(gemini, probe=True)
        assert http.calls[0][1]["x-api-key"] == "a"
        assert http.calls[1][1]["x-goog-api-key"] == "g"

    def test_resolver_usable_form(self, http, monkeypatch):
        monkeypatch.setenv("ZAI_API_KEY", "k")
        monkeypatch.setattr(routes_mod, "_bedrock_credentials", lambda: (False, "none"))
        assert RegistryRouteResolver().usable_routes_for("glm-5", probe=True) == {"zai"}


class TestConfig:
    @pytest.fixture(autouse=True)
    def _reset(self, monkeypatch, tmp_path):
        monkeypatch.delenv(config.CONFIG_ENV, raising=False)
        monkeypatch.chdir(tmp_path)
        config.reload_config()
        yield
        config.reload_config()

    def test_packaged_default(self):
        assert config.get("routes.probe_timeout_s") == 10.0

    def test_project_file_found_upward(self, tmp_path, monkeypatch):
        (tmp_path / "llm_utils.yaml").write_text("routes:\n  probe_timeout_s: 3\n")
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        monkeypatch.chdir(sub)
        config.reload_config()
        assert config.get("routes.probe_timeout_s") == 3

    def test_env_path_beats_the_project_file(self, tmp_path, monkeypatch):
        (tmp_path / "llm_utils.yaml").write_text("routes:\n  probe_timeout_s: 3\n")
        f = tmp_path / "x.yaml"
        f.write_text("routes:\n  probe_timeout_s: 7\n")
        monkeypatch.setenv(config.CONFIG_ENV, str(f))
        config.reload_config()
        assert config.get("routes.probe_timeout_s") == 7

    def test_unknown_key_fails(self, tmp_path, monkeypatch):
        f = tmp_path / "x.yaml"
        f.write_text("routes:\n  probe_timout_s: 3\n")
        monkeypatch.setenv(config.CONFIG_ENV, str(f))
        config.reload_config()
        with pytest.raises(ValueError, match="probe_timout_s"):
            config.get_config()

    def test_wrong_type_fails(self, tmp_path):
        (tmp_path / "llm_utils.yaml").write_text("routes:\n  probe_timeout_s: '3'\n")
        config.reload_config()
        with pytest.raises(ValueError, match="must be a float"):
            config.get_config()
        with pytest.raises(ValueError, match="may not be null"):
            config.configure(routes={"probe_timeout_s": None})

    def test_configure_beats_the_project_file(self, tmp_path):
        (tmp_path / "llm_utils.yaml").write_text("routes:\n  probe_timeout_s: 3\n")
        config.reload_config()
        config.configure(routes={"probe_timeout_s": 2.5})
        assert config.get("routes.probe_timeout_s") == 2.5
        with pytest.raises(ValueError):
            config.configure(routes={"nope": 1})

    def test_a_fifo_config_file_is_accepted(self, tmp_path, monkeypatch):
        import os
        import threading
        fifo = tmp_path / "cfg.fifo"
        os.mkfifo(fifo)
        writer = threading.Thread(target=lambda: fifo.write_text("routes:\n  probe_timeout_s: 4\n"))
        writer.start()
        monkeypatch.setenv(config.CONFIG_ENV, str(fifo))
        config.reload_config()
        assert config.get("routes.probe_timeout_s") == 4
        writer.join(timeout=5)
