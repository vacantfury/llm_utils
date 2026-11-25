"""Guards consumer selection of equivalent models across serving jurisdictions."""

from llm_utils import LLMModel, Provider


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


class TestRouteSelection:
    def test_prc_model_resolves_to_its_us_twin(self):
        assert LLMModel.GLM_5_2.us_route() is LLMModel.OR_GLM_5_2
        assert LLMModel.DEEPSEEK_V4_FLASH.us_route() is LLMModel.OR_DEEPSEEK_V4_FLASH
        assert LLMModel.KIMI_K3.us_route() is LLMModel.OR_KIMI_K3
        assert LLMModel.KIMI_K2_7_CODE.us_route() is LLMModel.OR_KIMI_K2_7_CODE
        assert LLMModel.GLM_4_7.us_route() is LLMModel.OR_GLM_4_7

    def test_multiple_us_routes_tie_break_is_explicit(self):
        """GLM-5 is on BOTH OpenRouter and Bedrock; the winner must come from
        the documented preference order, not enum definition order."""
        us_twins = {m.provider for m in LLMModel.GLM_5.route_twins()}
        assert {Provider.OPENROUTER, Provider.BEDROCK} <= us_twins
        assert LLMModel.GLM_5.us_route() is LLMModel.OR_GLM_5
        # Bedrock is still reachable — just not the default pick.
        assert LLMModel.BEDROCK_GLM_5 in LLMModel.GLM_5.routes()

    def test_models_absent_from_openrouter_have_no_us_route(self):
        """Verified 2026-07-30: these two are NOT on OpenRouter, so the
        registry must report no US route rather than inventing one."""
        assert LLMModel.GLM_4_7_FLASHX.us_route() is None
        assert LLMModel.KIMI_K2_7_CODE_HIGHSPEED.us_route() is None

    def test_us_route_of_a_us_model_is_itself(self):
        assert LLMModel.GPT_5.us_route() is LLMModel.GPT_5
        assert LLMModel.OR_GLM_5_2.us_route() is LLMModel.OR_GLM_5_2

    def test_self_route_found_where_registered(self):
        assert (LLMModel.BEDROCK_DEEPSEEK_V3_2.self_route()
                is LLMModel.DEEPSEEK_V3_2_EXP)
        assert LLMModel.KIMI_K2_INSTRUCT.self_route() is LLMModel.KIMI_K2_INSTRUCT
        assert LLMModel.GPT_5.self_route() is None
