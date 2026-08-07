"""Model-registry invariants: every row constructible, priced sanely, provider
dispatchable, string lookup unambiguous within a provider. No network."""

import collections

import pytest

from llm_utils import LLMModel, LLMServiceFactory, Provider

# Serving routes we run ourselves — zero price is the convention there.
_SELF_SERVED = {Provider.LOCAL, Provider.SLURM_CLUSTER}


def test_every_row_has_id_provider_prices():
    for m in LLMModel:
        assert isinstance(m.model_id, str) and m.model_id, m.name
        assert isinstance(m.provider, Provider), m.name
        assert m.input_price is not None and m.input_price >= 0, m.name
        assert m.output_price is not None and m.output_price >= 0, m.name


def test_metered_api_rows_are_priced():
    """Metered API providers must carry real list prices — a zero price makes
    cost recording silently log $0. Bedrock is exempted row-by-row: the
    open-weight serverless rows deliberately carry 0 until verified (see
    registry comments)."""
    for m in LLMModel:
        if m.provider in _SELF_SERVED or m.provider is Provider.BEDROCK:
            continue
        if m is LLMModel.GLM_4_7_FLASH:  # provider's free tier — genuine $0
            continue
        assert m.input_price > 0 and m.output_price > 0, (
            f"{m.name}: zero price on metered provider {m.provider.value}")


def test_self_served_rows_are_free():
    for m in LLMModel:
        if m.provider in _SELF_SERVED:
            assert m.input_price == 0 and m.output_price == 0, m.name


def test_every_provider_dispatches():
    for provider in Provider:
        assert LLMServiceFactory.is_provider_supported(provider), provider


def test_model_id_unique_within_provider():
    seen = collections.Counter((m.provider, m.model_id) for m in LLMModel)
    dups = [k for k, v in seen.items() if v > 1]
    assert not dups, f"duplicate (provider, model_id): {dups}"


def test_from_string_resolves_unique_ids_and_rejects_ambiguous():
    """Every uniquely-registered id must resolve; an id shared across serving
    routes (same weights local AND cluster) must raise instead of silently
    picking a provider."""
    by_id = collections.Counter(m.model_id for m in LLMModel)
    for m in LLMModel:
        if by_id[m.model_id] == 1:
            assert LLMModel.from_string(m.model_id) is m
        else:
            with pytest.raises(ValueError, match="Ambiguous"):
                LLMModel.from_string(m.model_id)
            # The enum NAME still resolves each route unambiguously.
            assert LLMModel.from_string(m.name) is m


def test_quirks_are_quirk_enum_members():
    from llm_utils import ModelQuirk
    for m in LLMModel:
        for q in getattr(m, "quirks", ()) or ():
            assert isinstance(q, ModelQuirk), (m.name, q)


def test_bedrock_claude_twins_share_temperature_quirk():
    """The same model rejects a custom temperature regardless of serving
    route — a Bedrock row must carry the quirk its direct-API twin has
    (the gap that 400'd every Bedrock Claude 5.x batch row)."""
    from llm_utils import ModelQuirk
    pairs = [
        (LLMModel.CLAUDE_SONNET_5, LLMModel.BEDROCK_CLAUDE_SONNET_5),
        (LLMModel.CLAUDE_OPUS_4_8, LLMModel.BEDROCK_CLAUDE_OPUS_4_8),
        (LLMModel.CLAUDE_FABLE_5, LLMModel.BEDROCK_CLAUDE_FABLE_5),
    ]
    for direct, bedrock in pairs:
        assert direct.has_quirk(ModelQuirk.NO_CUSTOM_TEMPERATURE), direct.name
        assert bedrock.has_quirk(ModelQuirk.NO_CUSTOM_TEMPERATURE), bedrock.name


class TestVersionProvenance:
    """`__version__` must equal the INSTALLED distribution version.

    A hardcoded literal here is a second source of truth beside
    pyproject.toml and it drifted unnoticed across two releases (v6.0.0 and
    v6.1.0 both shipped reading "5.2.0"). Consumers print this string into
    experiment provenance records, so a stale value silently misattributes
    which code produced a result.
    """

    def test_version_matches_installed_distribution(self):
        import importlib.metadata as md
        import llm_utils
        assert llm_utils.__version__ == md.version("llm_utils")

    def test_version_is_not_a_placeholder(self):
        import llm_utils
        assert not llm_utils.__version__.endswith("+unknown"), (
            "package metadata unreadable — is it installed?")
