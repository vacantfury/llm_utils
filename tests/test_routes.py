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

    def test_gemma_3_12b_managed_and_self_served_are_twins(self):
        """The registry's only managed-vs-self-served pair on an OPEN
        multimodal checkpoint.

        A consumer holding weights fixed while varying the serving stack has
        to be able to reach both routes from one handle; this pair is what
        makes that a registry fact rather than a naming coincidence. The two
        ids differ only by `.` vs `/`, so pairing them by string similarity
        would be a coin flip — `weights` is the join key.
        """
        bedrock, cluster = (LLMModel.BEDROCK_GEMMA_3_12B,
                            LLMModel.GEMMA_3_12B_IT)
        assert bedrock.route_twins() == (cluster,)
        assert bedrock.self_route() is cluster
        assert cluster.self_route() is cluster
        assert bedrock.jurisdiction == "us"
        assert cluster.jurisdiction == "self"
        # Same checkpoint, deliberately near-identical but DISTINCT ids.
        assert bedrock.model_id == "google.gemma-3-12b-it"
        assert cluster.model_id == "google/gemma-3-12b-it"
        assert bedrock.model_id != cluster.model_id
        # Neither id is ambiguous, so from_string resolves each to one route.
        assert LLMModel.from_string("google.gemma-3-12b-it") is bedrock
        assert LLMModel.from_string("google/gemma-3-12b-it") is cluster

    def test_qwen_vl_generational_ladder_is_complete(self):
        """Three qwen-VL generations, same family and size class.

        A consumer using this as a natural experiment on post-training needs
        all three rungs to exist, to share `family`, and to be self-served —
        otherwise "the difference is the post-training" smuggles in a vendor
        or a serving route as well. Assert the invariants that claim rests on
        rather than trusting the enum to stay tidy.
        """
        ladder = (LLMModel.QWEN2_VL_7B,
                  LLMModel.QWEN2_5_VL_7B,
                  LLMModel.QWEN3_VL_8B_INSTRUCT)
        assert [m.family for m in ladder] == ["qwen"] * 3
        assert {m.provider for m in ladder} == {Provider.SLURM_CLUSTER}
        assert {m.jurisdiction for m in ladder} == {"self"}
        # Distinct checkpoints: none of them is another's twin.
        assert len({m.model_id for m in ladder}) == 3
        for m in ladder:
            assert m.weights is None, (
                f"{m.name} gained a `weights` twin — the ladder assumes each "
                "rung is a single checkpoint on a single route")
        assert LLMModel.from_string("Qwen/Qwen2-VL-7B-Instruct") is ladder[0]
