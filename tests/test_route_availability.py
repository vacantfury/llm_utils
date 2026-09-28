"""Route availability seam (t10): logical model -> routes, and which routes this
process can use. Offline: probes run against a fake HTTP layer."""

import httpx2
import pytest

import llm_utils
from llm_utils import (
    LLMModel, LLMServiceFactory, Provider, RegistryRouteResolver, config,
    route_models, route_status, routes_for, usable_routes, available_routes,
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
        assert len(available_routes("glm-5")) == 3

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
            return httpx2.Response(fake_get.answer, request=httpx2.Request("GET", url))

        fake_get.answer = 200
        monkeypatch.setattr(routes_mod.httpx2, "get", fake_get)
        fake_get.calls = calls
        return fake_get

    def test_no_probe_when_config_fails(self, http):
        s = route_status(LLMModel.OR_GLM_5, probe=True)
        assert not s.usable and http.calls == []

    def test_probe_hits_the_model_listing_with_the_key(self, http, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        s = route_status(LLMModel.OR_GLM_5, probe=True)
        assert s.usable and s.probed
        url, headers, timeout = http.calls[0]
        assert url == "https://openrouter.ai/api/v1/models"
        assert headers["Authorization"] == "Bearer k"
        assert timeout == config.get("routes.probe_timeout_s")

    def test_openai_default_base(self, http, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        route_status(LLMModel.GPT_4O, probe=True)
        assert http.calls[0][0] == "https://api.openai.com/v1/models"

    @pytest.mark.parametrize("code,usable,word", [
        (401, False, "rejected"), (403, False, "rejected"), (503, False, "endpoint error"),
        (404, True, "not confirmed"), (429, True, "not confirmed")])
    def test_status_codes(self, http, monkeypatch, code, usable, word):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        http.answer = code
        s = route_status(LLMModel.OR_GLM_5, probe=True)
        assert s.usable is usable and word in s.reason

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

    def test_env_path_wins_and_unknown_key_fails(self, tmp_path, monkeypatch):
        f = tmp_path / "x.yaml"
        f.write_text("routes:\n  probe_timout_s: 3\n")
        monkeypatch.setenv(config.CONFIG_ENV, str(f))
        config.reload_config()
        with pytest.raises(ValueError, match="probe_timout_s"):
            config.get_config()

    def test_configure_overrides_and_validates(self):
        config.configure(routes={"probe_timeout_s": 2.5})
        assert config.get("routes.probe_timeout_s") == 2.5
        with pytest.raises(ValueError):
            config.configure(routes={"nope": 1})
