"""Multi-route lookup: jurisdiction facts and same-model route twins.
Registry-only — no network."""

from llm_utils import LLMModel, Provider


class TestJurisdiction:
    def test_every_provider_declares_a_jurisdiction(self):
        for p in Provider:
            assert p.jurisdiction in {"us", "prc", "self"}, p

    def test_mainland_endpoints_marked_prc(self):
        for p in (Provider.DEEPSEEK, Provider.ZAI, Provider.MOONSHOT):
            assert p.jurisdiction == "prc", p

    def test_us_hosts_of_open_weights_marked_us(self):
        # The whole point of the dual-route registry: same weights, US host.
        for p in (Provider.OPENROUTER, Provider.BEDROCK):
            assert p.jurisdiction == "us", p

    def test_self_served_routes_marked_self(self):
        for p in (Provider.LOCAL, Provider.SLURM_CLUSTER):
            assert p.jurisdiction == "self", p

    def test_model_jurisdiction_follows_its_provider(self):
        assert LLMModel.GLM_5_2.jurisdiction == "prc"
        assert LLMModel.OR_GLM_5_2.jurisdiction == "us"
        assert LLMModel.KIMI_K2_INSTRUCT.jurisdiction == "self"


class TestRouteTwins:
    def test_twins_are_symmetric(self):
        for m in LLMModel:
            for twin in m.route_twins():
                assert m in twin.route_twins(), (m.name, twin.name)

    def test_twins_share_weights_and_differ_in_route(self):
        for m in LLMModel:
            for twin in m.route_twins():
                assert twin.weights == m.weights
                assert twin.provider is not m.provider, (m.name, twin.name)

    def test_single_route_models_have_no_twins(self):
        assert LLMModel.GPT_5.route_twins() == ()
        assert LLMModel.GPT_5.routes() == (LLMModel.GPT_5,)

    def test_routes_includes_self(self):
        for m in LLMModel:
            assert m in m.routes()
            assert len(m.routes()) == len(m.route_twins()) + 1

    def test_weights_never_spans_two_rows_on_one_provider(self):
        """A weights value must identify ONE row per provider — otherwise
        'the OpenRouter route to X' would be ambiguous."""
        seen = {}
        for m in LLMModel:
            if not m.weights:
                continue
            key = (m.weights, m.provider)
            assert key not in seen, f"{m.name} collides with {seen[key]}"
            seen[key] = m.name


class TestRouteSelection:
    def test_prc_model_resolves_to_its_us_twin(self):
        assert LLMModel.GLM_5_2.us_route() is LLMModel.OR_GLM_5_2
        assert LLMModel.DEEPSEEK_V4_FLASH.us_route() is LLMModel.OR_DEEPSEEK_V4_FLASH
        assert LLMModel.KIMI_K3.us_route() is LLMModel.OR_KIMI_K3
        # Bedrock counts as the US route where that's what's registered.
        assert LLMModel.GLM_5.us_route() is LLMModel.BEDROCK_GLM_5

    def test_us_route_of_a_us_model_is_itself(self):
        assert LLMModel.GPT_5.us_route() is LLMModel.GPT_5
        assert LLMModel.OR_GLM_5_2.us_route() is LLMModel.OR_GLM_5_2

    def test_us_route_is_none_when_unregistered(self):
        # A PRC-only row with no US twin must report None, not a wrong guess.
        prc_only = [m for m in LLMModel
                    if m.jurisdiction == "prc" and m.us_route() is None]
        for m in prc_only:
            assert m.us_route() is None

    def test_self_route_found_where_registered(self):
        assert (LLMModel.BEDROCK_DEEPSEEK_V3_2.self_route()
                is LLMModel.DEEPSEEK_V3_2_EXP)
        assert LLMModel.KIMI_K2_INSTRUCT.self_route() is LLMModel.KIMI_K2_INSTRUCT
        assert LLMModel.GPT_5.self_route() is None

    def test_claude_direct_and_bedrock_are_twins(self):
        assert (LLMModel.CLAUDE_SONNET_5.route_twins()
                == (LLMModel.BEDROCK_CLAUDE_SONNET_5,))
