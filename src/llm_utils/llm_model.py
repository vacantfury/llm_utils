"""
LLM model registry — every static fact about a model lives here.

The shape: each LLMModel enum value is a frozen `ModelSpec` dataclass.

  LLMModel.GPT_5.value
      → ModelSpec(model_id="gpt-5", provider=OPENAI, input_price=2.5,
                  output_price=10.0, max_context_len=None,
                  quirks={USES_MAX_COMPLETION_TOKENS, NO_CUSTOM_TEMPERATURE})

What lives here (static, code-time facts about the model):
  - model_id, provider, pricing
  - max_context_len: architectural ceiling from upstream config.json (e.g.,
    Llama-2's max_position_embeddings=2048). Used by LLMConfig validator
    to reject `cluster.max_model_len` overruns before vLLM rejects them.
  - quirks: API-side behavior flags (e.g., GPT-5 only accepts temperature=1).

What does NOT live here (deployment choices, not model facts):
  - num_gpus, mem_gb, dtype, the actually-served max_model_len,
    gpu_types_excluded — those live in the CONSUMER's cluster config.

Adding a new model: append one `LLMModel.<NAME> = ModelSpec(...)` row.
Adding a new per-model fact: add a field to `ModelSpec`, populate selected
rows. No new scattered dicts.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Provider(str, Enum):
    """LLM service provider — drives factory dispatch."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    # ── OpenAI-compatible third-party endpoints. DeepSeek + Z.AI + Moonshot
    # are DIRECT MAINLAND endpoints — data is processed under PRC jurisdiction
    # per the providers' own policies. Consumers must route ONLY
    # zero-personal-data bulk work through them (LLM-judge calls / evals over
    # public-benchmark responses); anything personal stays on US-jurisdiction
    # providers or a non-retaining US host of the same weights (see OPENROUTER).
    # xAI (Grok) is US jurisdiction. This package is transport only — each
    # consumer enforces its own routing policy on top.
    DEEPSEEK = "deepseek"
    ZAI = "zai"
    # US aggregator serving open weights via configurable host routing. With a
    # zero-data-retention routing policy enabled on the account, it is the
    # US-jurisdiction route to Chinese open-weight models.
    OPENROUTER = "openrouter"
    XAI = "xai"
    MOONSHOT = "moonshot"
    # AWS Bedrock (bedrock-runtime.converse) — US-hosted managed API (us-east-1)
    # fronting Claude / Kimi / DeepSeek / GLM / Qwen / Nova / …; boto3 auth via
    # the standard AWS credential chain (AWS_PROFILE). US-jurisdiction even for
    # Chinese-origin weights (Bedrock hosts them in us-east-1). NOT
    # OpenAI-wire-compatible → its own BedrockService (converse), not an
    # OpenAIService subclass.
    BEDROCK = "bedrock"
    LOCAL = "local"
    SLURM_CLUSTER = "slurm_cluster"

    @property
    def jurisdiction(self) -> str:
        """Where this route's endpoint processes data: ``"us"`` ·
        ``"prc"`` (mainland China) · ``"self"`` (our own hardware — no
        third party sees the data).

        A transport FACT about the endpoint, not a policy: this package never
        decides what may be sent where. Consumers with a data-routing policy
        read this (with ``LLMModel.route_twins`` / ``us_route`` below) to pick
        a route for a given payload.
        """
        return _PROVIDER_JURISDICTION[self]


# Endpoint jurisdiction per provider. PRC entries are the DIRECT mainland
# endpoints; the same open weights served by OPENROUTER / BEDROCK are
# US-jurisdiction, which is what makes the dual-route registry rows useful.
_PROVIDER_JURISDICTION = {
    Provider.OPENAI: "us",
    Provider.ANTHROPIC: "us",
    Provider.GOOGLE: "us",
    Provider.XAI: "us",
    Provider.OPENROUTER: "us",
    Provider.BEDROCK: "us",
    Provider.DEEPSEEK: "prc",
    Provider.ZAI: "prc",
    Provider.MOONSHOT: "prc",
    Provider.LOCAL: "self",
    Provider.SLURM_CLUSTER: "self",
}

# Tie-break when a model has SEVERAL US-jurisdiction routes (e.g. GLM-5 is on
# both OpenRouter and Bedrock): `us_route()` returns the first match in this
# order. Ranked by how universally usable the route is, NOT by price —
# OpenRouter needs one API key, while Bedrock needs an AWS account with the
# model enabled, so an OpenRouter row is the safer default. A consumer that
# wants a specific host filters `routes()` itself.
_US_ROUTE_PREFERENCE = (
    Provider.OPENROUTER,
    Provider.OPENAI,
    Provider.ANTHROPIC,
    Provider.GOOGLE,
    Provider.XAI,
    Provider.BEDROCK,
)


class ModelQuirk(str, Enum):
    """API-side behavior flags. A model carries a frozenset of these.

    Each quirk is observable in code via `model.has_quirk(...)`. Adding a
    new quirk is just: define the enum value, mark the relevant rows on
    LLMModel. No new sets-by-model scattered through constants.py.
    """
    # OpenAI newer models renamed `max_tokens` → `max_completion_tokens`
    USES_MAX_COMPLETION_TOKENS = "uses_max_completion_tokens"
    # OpenAI reasoning models reject any temperature != 1.0
    NO_CUSTOM_TEMPERATURE = "no_custom_temperature"
    # Gemini thinking models count THOUGHT tokens against max_output_tokens,
    # so a caller's small cap starves the visible text (a max_tokens=1200 call
    # can truncate mid-sentence with the budget spent on thought). The Google
    # service grants thinking headroom on top of the caller's budget.
    THINKING_SHARES_OUTPUT_BUDGET = "thinking_shares_output_budget"


@dataclass(frozen=True)
class ModelSpec:
    """Static facts about a model. Frozen so each enum value is hashable."""
    model_id: str
    provider: Provider
    input_price: float = 0.0          # $/1M input tokens (0 for self-hosted)
    output_price: float = 0.0         # $/1M output tokens
    max_context_len: Optional[int] = None   # arch ceiling; None when unknown
    # Provider-enforced OUTPUT ceiling (maxTokens). None = no known cap (use the
    # global default). Bedrock rejects a maxTokens above the model's hard limit
    # with a ValidationException (e.g. Amazon Nova caps at 10000 while the project
    # default max_tokens is 16384) — BedrockService clamps its request to this.
    max_output_tokens: Optional[int] = None
    quirks: frozenset = field(default_factory=frozenset)
    # Research-facing labels, stamped as first-class fields into results.json so
    # readers never recover them by archaeology. `family` is a static fact;
    # `alignment_tier` is a coarse safety-alignment label (strong/mid/weak).
    family: Optional[str] = None
    alignment_tier: Optional[str] = None
    # Identity of the underlying MODEL, for models reachable on more than one
    # serving route. Rows sharing a `weights` value are the same model served
    # by different endpoints — GLM-5.2 direct from Z.AI (mainland China), via
    # OpenRouter (US), or self-served on the cluster; Claude Sonnet 5 direct
    # from Anthropic or via Bedrock. Set it to the vendor's canonical model
    # name. None = exactly one registered route. Consumed by `route_twins()` /
    # `us_route()` / `self_route()`.
    weights: Optional[str] = None
    # Chat template NAME for self-served (vLLM) deployment — a model fact the
    # consumer passes into its serving layer's cluster config, which resolves
    # it to a .jinja file it ships (this package stopped owning the serving
    # lifecycle in v6.0.0). None means "use the tokenizer's baked-in template"
    # (correct for modern chat-tuned checkpoints; their tokenizer.json ships one).
    # Set explicitly for: (a) classifiers / non-chat fine-tunes whose training
    # prompts are pre-formatted (HarmBench-Llama-2-13b-cls → "passthrough"),
    # (b) legacy chat models whose tokenizer doesn't ship a template.
    chat_template: Optional[str] = None


# ────────────────────────────────────────────────────────────────────
# Quirk presets (used heavily by GPT-5 / O-series rows below)
# ────────────────────────────────────────────────────────────────────
_GPT5_QUIRKS = frozenset({
    ModelQuirk.USES_MAX_COMPLETION_TOKENS,
    ModelQuirk.NO_CUSTOM_TEMPERATURE,
})
_GPT41_QUIRKS = frozenset({
    ModelQuirk.USES_MAX_COMPLETION_TOKENS,
})
# Newer Claude models reject `temperature` (400). They use `max_tokens` (not
# `max_completion_tokens`), so only the no-temperature quirk applies.
_NO_TEMP = frozenset({ModelQuirk.NO_CUSTOM_TEMPERATURE})
# Rows that think by default (Gemini 2.5/3.x flash+pro, Claude Opus 5 /
# Fable 5): thought tokens bill and cap as OUTPUT tokens, so services grant
# extra output budget and count thought tokens in recorded usage.
_THINKING_BUDGET = frozenset({ModelQuirk.THINKING_SHARES_OUTPUT_BUDGET})


class LLMModel(Enum):
    """Per-model static registry. Each value is a frozen ModelSpec."""

    # ──────── OpenAI ────────
    # Provider shutdown schedule (verified 2026-07-24): gpt-3.5-turbo, gpt-4,
    # gpt-4-turbo, gpt-4.1-nano, o1, o3-mini, o4-mini shut down 2026-10-23;
    # gpt-5 / gpt-5-mini / gpt-5-nano and o3 shut down 2026-12-11.
    GPT_3_5_TURBO = ModelSpec("gpt-3.5-turbo",  Provider.OPENAI, 0.50,  1.50)
    GPT_4         = ModelSpec("gpt-4",          Provider.OPENAI, 30.00, 60.00)
    GPT_4_TURBO   = ModelSpec("gpt-4-turbo",    Provider.OPENAI, 10.00, 30.00)
    GPT_4O        = ModelSpec("gpt-4o",         Provider.OPENAI, 2.50,  10.00)
    GPT_4O_MINI   = ModelSpec("gpt-4o-mini",    Provider.OPENAI, 0.15,  0.60)

    # GPT-4.1 series — `max_completion_tokens` rename starts here
    GPT_4_1       = ModelSpec("gpt-4.1",        Provider.OPENAI, 2.00,  8.00,  quirks=_GPT41_QUIRKS)
    GPT_4_1_MINI  = ModelSpec("gpt-4.1-mini",   Provider.OPENAI, 0.40,  1.60,  quirks=_GPT41_QUIRKS)
    GPT_4_1_NANO  = ModelSpec("gpt-4.1-nano",   Provider.OPENAI, 0.10,  0.40,  quirks=_GPT41_QUIRKS)

    # GPT-5 series — also drops custom temperature
    # (gpt-5 / gpt-5-nano prices corrected 2026-07-24 against the official pricing page)
    GPT_5         = ModelSpec("gpt-5",          Provider.OPENAI, 1.25,  10.00, quirks=_GPT5_QUIRKS)
    GPT_5_MINI    = ModelSpec("gpt-5-mini",     Provider.OPENAI, 0.25,  2.00,  quirks=_GPT5_QUIRKS)
    GPT_5_NANO    = ModelSpec("gpt-5-nano",     Provider.OPENAI, 0.05,  0.40,  quirks=_GPT5_QUIRKS)
    GPT_5_PRO     = ModelSpec("gpt-5-pro",      Provider.OPENAI, 15.00, 120.00, quirks=_GPT5_QUIRKS)
    # gpt-5.1 / gpt-5.2: live with no scheduled retirement (verified 2026-07-24;
    # an earlier "retires 2026-07-23" note here was wrong — only the separate
    # gpt-5.2-chat-latest alias retires, 2026-08-10).
    GPT_5_1       = ModelSpec("gpt-5.1",        Provider.OPENAI, 1.25,  10.00, quirks=_GPT5_QUIRKS)
    GPT_5_2       = ModelSpec("gpt-5.2",        Provider.OPENAI, 1.75,  14.00, quirks=_GPT5_QUIRKS)
    GPT_5_2_PRO   = ModelSpec("gpt-5.2-pro",    Provider.OPENAI, 21.00, 168.00, quirks=_GPT5_QUIRKS)
    GPT_5_4       = ModelSpec("gpt-5.4",        Provider.OPENAI, 2.50,  15.00, quirks=_GPT5_QUIRKS)
    GPT_5_4_MINI  = ModelSpec("gpt-5.4-mini",   Provider.OPENAI, 0.75,  4.50,  quirks=_GPT5_QUIRKS)
    GPT_5_4_NANO  = ModelSpec("gpt-5.4-nano",   Provider.OPENAI, 0.20,  1.25,  quirks=_GPT5_QUIRKS)
    GPT_5_4_PRO   = ModelSpec("gpt-5.4-pro",    Provider.OPENAI, 30.00, 180.00, quirks=_GPT5_QUIRKS)
    GPT_5_5       = ModelSpec("gpt-5.5",        Provider.OPENAI, 5.00,  30.00, quirks=_GPT5_QUIRKS)
    GPT_5_5_PRO   = ModelSpec("gpt-5.5-pro",    Provider.OPENAI, 30.00, 180.00, quirks=_GPT5_QUIRKS)
    # GPT-5.6 (launched 2026-07-09): sol = flagship, terra = mid-tier (JUDGE pick),
    # luna = cost tier (added 2026-07-24 — the launch's third tier, initially missed).
    # No 5.6 "-pro" SKUs exist: pro is a `reasoning.mode` request param, not a model id.
    GPT_5_6_SOL   = ModelSpec("gpt-5.6-sol",    Provider.OPENAI, 5.00,  30.00, quirks=_GPT5_QUIRKS)
    GPT_5_6_TERRA = ModelSpec("gpt-5.6-terra",  Provider.OPENAI, 2.50,  15.00, quirks=_GPT5_QUIRKS)
    GPT_5_6_LUNA  = ModelSpec("gpt-5.6-luna",   Provider.OPENAI, 1.00,  6.00,  quirks=_GPT5_QUIRKS)

    # O-series reasoning models — same shape as GPT-5
    O1            = ModelSpec("o1",             Provider.OPENAI, 15.00, 60.00, quirks=_GPT5_QUIRKS)
    O3            = ModelSpec("o3",             Provider.OPENAI, 2.00,  8.00,  quirks=_GPT5_QUIRKS)
    O3_MINI       = ModelSpec("o3-mini",        Provider.OPENAI, 1.10,  4.40,  quirks=_GPT5_QUIRKS)
    O4_MINI       = ModelSpec("o4-mini",        Provider.OPENAI, 1.10,  4.40,  quirks=_GPT5_QUIRKS)

    # ──────── Anthropic ────────
    # Claude 5 family added 2026-07-24. All three reject a custom temperature
    # (sampling params removed on 5.x, as on Opus 4.7/4.8) and run a 1M context.
    # Sonnet
    CLAUDE_SONNET_4   = ModelSpec("claude-sonnet-4-20250514",   Provider.ANTHROPIC, 3.00, 15.00)
    CLAUDE_SONNET_4_5 = ModelSpec("claude-sonnet-4-5-20250929", Provider.ANTHROPIC, 3.00, 15.00)
    CLAUDE_SONNET_4_6 = ModelSpec("claude-sonnet-4-6",          Provider.ANTHROPIC, 3.00, 15.00)
    CLAUDE_SONNET_5   = ModelSpec("claude-sonnet-5",            Provider.ANTHROPIC, 3.00, 15.00,  # intro $2/$10 through 2026-08-31
        max_context_len=1_000_000, quirks=_NO_TEMP, weights="claude-sonnet-5")
    # NOTE (2026-07-30, per the Anthropic pricing page): Opus 4, Opus 4.1 and
    # Sonnet 4 are deprecated/retired on the DIRECT API (still served via
    # Bedrock / Google Cloud). Rows kept for reference; expect direct calls
    # to start failing — prefer the 4.5+ rows.
    # Opus
    CLAUDE_OPUS_4   = ModelSpec("claude-opus-4-20250514",   Provider.ANTHROPIC, 15.00, 75.00)
    CLAUDE_OPUS_4_1 = ModelSpec("claude-opus-4-1-20250805", Provider.ANTHROPIC, 15.00, 75.00)
    CLAUDE_OPUS_4_5 = ModelSpec("claude-opus-4-5-20251101", Provider.ANTHROPIC, 5.00,  25.00)
    CLAUDE_OPUS_4_6 = ModelSpec("claude-opus-4-6",          Provider.ANTHROPIC, 5.00,  25.00)
    CLAUDE_OPUS_4_7 = ModelSpec("claude-opus-4-7",          Provider.ANTHROPIC, 5.00,  25.00, quirks=_NO_TEMP)
    CLAUDE_OPUS_4_8 = ModelSpec("claude-opus-4-8",          Provider.ANTHROPIC, 5.00,  25.00, quirks=_NO_TEMP, weights="claude-opus-4-8")
    CLAUDE_OPUS_5   = ModelSpec("claude-opus-5",            Provider.ANTHROPIC, 5.00,  25.00,
        max_context_len=1_000_000, quirks=_NO_TEMP | _THINKING_BUDGET)  # thinking ON by default (unlike 4.7/4.8)
    # Frontier tier. Always-on thinking; requires 30-day data retention (a
    # zero-data-retention org gets a 400 on every request).
    CLAUDE_FABLE_5  = ModelSpec("claude-fable-5",           Provider.ANTHROPIC, 10.00, 50.00,
        max_context_len=1_000_000, quirks=_NO_TEMP | _THINKING_BUDGET, weights="claude-fable-5")
    # Haiku
    CLAUDE_HAIKU_4_5 = ModelSpec("claude-haiku-4-5-20251001", Provider.ANTHROPIC, 1.00, 5.00, weights="claude-haiku-4-5")

    # ──────── Google ────────
    # gemini-2.0-flash + gemini-2.0-flash-lite REMOVED 2026-07-24 (shut down
    # 2026-06-01 — API calls error). The 2.5 trio below shuts down 2026-10-16.
    GEMINI_2_5_FLASH               = ModelSpec("gemini-2.5-flash",                Provider.GOOGLE, 0.30,  2.50, quirks=_THINKING_BUDGET)
    GEMINI_2_5_FLASH_LITE          = ModelSpec("gemini-2.5-flash-lite",           Provider.GOOGLE, 0.10,  0.40)   # price corrected 2026-07-24
    GEMINI_2_5_PRO                 = ModelSpec("gemini-2.5-pro",                  Provider.GOOGLE, 1.25,  10.00, quirks=_THINKING_BUDGET)
    GEMINI_3_FLASH_PREVIEW         = ModelSpec("gemini-3-flash-preview",          Provider.GOOGLE, 0.50,  3.00, quirks=_THINKING_BUDGET)  # superseded; migrate to 3.6-flash
    # gemini-3-pro-preview + gemini-3.1-flash-lite-preview REMOVED 2026-07-12 (dead — 404, verified 2026-07-12); use the GA IDs below.
    GEMINI_3_1_PRO_PREVIEW         = ModelSpec("gemini-3.1-pro-preview",          Provider.GOOGLE, 2.00,  12.00, quirks=_THINKING_BUDGET)  # still the newest pro (no GA pro as of 2026-07-24)
    GEMINI_3_1_FLASH_LITE          = ModelSpec("gemini-3.1-flash-lite",           Provider.GOOGLE, 0.25,  1.50)   # GA (replaced dead preview)
    GEMINI_3_5_FLASH               = ModelSpec("gemini-3.5-flash",                Provider.GOOGLE, 1.50,  9.00, quirks=_THINKING_BUDGET)   # GA ~May 2026 — JUDGE pick
    # GA 2026-07-21 pair (added 2026-07-24):
    GEMINI_3_5_FLASH_LITE          = ModelSpec("gemini-3.5-flash-lite",           Provider.GOOGLE, 0.30,  2.50)
    GEMINI_3_6_FLASH               = ModelSpec("gemini-3.6-flash",                Provider.GOOGLE, 1.50,  7.50, quirks=_THINKING_BUDGET)   # current flash flagship

    # ──────── DeepSeek (direct mainland, OpenAI-compatible) — judge/eval only, no personal data ────────
    DEEPSEEK_V4_FLASH = ModelSpec("deepseek-v4-flash", Provider.DEEPSEEK, 0.14,  0.28, family="deepseek", weights="deepseek-v4-flash")
    DEEPSEEK_V4_PRO   = ModelSpec("deepseek-v4-pro",   Provider.DEEPSEEK, 0.435, 0.87, family="deepseek", weights="deepseek-v4-pro")

    # ──────── Z.AI / GLM (direct mainland, OpenAI-compatible) — judge/eval only, no personal data ────────
    GLM_5_2        = ModelSpec("glm-5.2",        Provider.ZAI, 1.40, 4.40, family="glm", weights="glm-5.2")   # current flagship (added 2026-07-12)
    GLM_5          = ModelSpec("glm-5",          Provider.ZAI, 1.00, 3.20, family="glm", weights="glm-5")
    GLM_5_TURBO    = ModelSpec("glm-5-turbo",    Provider.ZAI, 1.20, 4.00, family="glm", weights="glm-5-turbo")
    GLM_5V_TURBO   = ModelSpec("glm-5v-turbo",   Provider.ZAI, 1.20, 4.00, family="glm", weights="glm-5v-turbo")   # vision
    GLM_4_7        = ModelSpec("glm-4.7",        Provider.ZAI, 0.60, 2.20, family="glm", weights="glm-4.7")
    GLM_4_7_FLASHX = ModelSpec("glm-4.7-flashx", Provider.ZAI, 0.07, 0.40, family="glm")   # JUDGE pick — full 4.7 reasoning, ~20x cheaper
    GLM_4_7_FLASH  = ModelSpec("glm-4.7-flash",  Provider.ZAI, 0.00, 0.00, family="glm", weights="glm-4.7-flash")   # free tier
    GLM_4_6V       = ModelSpec("glm-4.6v",       Provider.ZAI, 0.30, 0.90, family="glm", weights="glm-4.6v")   # vision variant

    # ──────── xAI / Grok (US jurisdiction, OpenAI-compatible) ────────
    # Registry per docs.x.ai 2026-07-08. Retired names (grok-4, grok-4-fast,
    # grok-4-1-fast, grok-3, grok-3-mini) are server-side ALIASES routing to
    # grok-4.3 — register only the real current ids.
    GROK_4_5      = ModelSpec("grok-4.5",      Provider.XAI, 2.00, 6.00, family="grok")   # newest; xAI-recommended for code+general (500k ctx)
    GROK_4_3      = ModelSpec("grok-4.3",      Provider.XAI, 1.25, 2.50, family="grok")   # flagship value tier (1M ctx)
    GROK_4_20_REASONING     = ModelSpec("grok-4.20-0309-reasoning",     Provider.XAI, 1.25, 2.50, family="grok")
    GROK_4_20_NON_REASONING = ModelSpec("grok-4.20-0309-non-reasoning", Provider.XAI, 1.25, 2.50, family="grok")
    GROK_4_20_MULTI_AGENT   = ModelSpec("grok-4.20-multi-agent-0309",   Provider.XAI, 1.25, 2.50, family="grok")
    GROK_BUILD_0_1          = ModelSpec("grok-build-0.1",               Provider.XAI, 1.00, 2.00, family="grok")   # code-specialist (256k ctx)

    # ──────── Moonshot / Kimi (direct mainland, OpenAI-compatible) — judge/eval
    # /attack-target only, no personal data (added 2026-07-16). input_price is the
    # cache-MISS $/1M; Moonshot's cache-hit is ~10x cheaper and unmodeled here, as
    # with DeepSeek. Deprecated ids (kimi-k2-*-preview, kimi-k2-thinking,
    # kimi-latest) and kimi-k2.5 (sunsets 2026-08-31) are deliberately omitted.
    # The self-served open-weight KIMI_K2_INSTRUCT on Provider.SLURM_CLUSTER is
    # a separate row below and stays as-is. ────────
    KIMI_K3 = ModelSpec(
        # K3 (released 2026-07-16): 2.8T-param open-weight MoE, thinking always on.
        # NO_CUSTOM_TEMPERATURE: always-on thinking rejects a custom temperature.
        # ⚠️ its always-on thinking counts against max_tokens, so a small cap
        # starves the visible reply (verified empty at max_tokens=64) — the
        # default.yaml budget (16384) is a normal budget and is correct here.
        "kimi-k3", Provider.MOONSHOT, 3.00, 15.00,
        max_context_len=1_048_576, family="kimi", weights="kimi-k3",
        quirks=frozenset({ModelQuirk.NO_CUSTOM_TEMPERATURE}))
    KIMI_K2_7_CODE = ModelSpec(
        "kimi-k2.7-code", Provider.MOONSHOT, 0.95, 4.00,
        max_context_len=262_144, family="kimi", weights="kimi-k2.7-code")
    KIMI_K2_7_CODE_HIGHSPEED = ModelSpec(
        "kimi-k2.7-code-highspeed", Provider.MOONSHOT, 1.90, 8.00,
        max_context_len=262_144, family="kimi")
    KIMI_K2_6 = ModelSpec(
        "kimi-k2.6", Provider.MOONSHOT, 0.95, 4.00,
        max_context_len=262_144, family="kimi", weights="kimi-k2.6")

    # ──────── OpenRouter (US aggregator over hosted open weights) ────────
    # US-jurisdiction route to open-weight models, incl. Chinese-origin
    # families — with a zero-data-retention routing policy enabled on the
    # account, requests reach only non-retaining hosts. Prices = cheapest-host
    # list at registration; ZDR routing may pick a pricier host.
    #
    # `weights=` pairs each row with its DIRECT (mainland) twin above, so
    # `LLMModel.<NATIVE>.us_route()` returns the row here — see route_twins().
    # Prices re-verified against openrouter.ai model pages 2026-07-30.
    # Rows marked PROMO are launch discounts that WILL expire — re-verify.
    OR_DEEPSEEK_V4_FLASH = ModelSpec("deepseek/deepseek-v4-flash", Provider.OPENROUTER, 0.09,  0.18, family="deepseek", weights="deepseek-v4-flash")
    OR_DEEPSEEK_V4_PRO   = ModelSpec("deepseek/deepseek-v4-pro",   Provider.OPENROUTER, 0.435, 0.87, family="deepseek", weights="deepseek-v4-pro")
    OR_GLM_5_2           = ModelSpec("z-ai/glm-5.2",               Provider.OPENROUTER, 0.36,  0.75, family="glm", weights="glm-5.2")      # PROMO (-70%)
    OR_GLM_5             = ModelSpec("z-ai/glm-5",                 Provider.OPENROUTER, 0.60,  1.92, family="glm", weights="glm-5")
    OR_GLM_5_TURBO       = ModelSpec("z-ai/glm-5-turbo",           Provider.OPENROUTER, 1.20,  4.00, family="glm", weights="glm-5-turbo")
    OR_GLM_5V_TURBO      = ModelSpec("z-ai/glm-5v-turbo",          Provider.OPENROUTER, 1.20,  4.00, family="glm", weights="glm-5v-turbo")  # vision
    OR_GLM_4_7           = ModelSpec("z-ai/glm-4.7",               Provider.OPENROUTER, 0.40,  1.75, family="glm", weights="glm-4.7")
    OR_GLM_4_7_FLASH     = ModelSpec("z-ai/glm-4.7-flash",         Provider.OPENROUTER, 0.06,  0.40, family="glm", weights="glm-4.7-flash")  # free on the direct endpoint, paid here
    OR_GLM_4_6V          = ModelSpec("z-ai/glm-4.6v",              Provider.OPENROUTER, 0.30,  0.90, family="glm", weights="glm-4.6v")     # vision
    OR_QWEN_3_7_MAX      = ModelSpec("qwen/qwen3.7-max",           Provider.OPENROUTER, 1.475, 4.425, family="qwen")   # no direct-endpoint row registered
    OR_KIMI_K2_6         = ModelSpec("moonshotai/kimi-k2.6",       Provider.OPENROUTER, 0.60,  3.41, family="kimi", weights="kimi-k2.6")
    OR_KIMI_K2_7_CODE    = ModelSpec("moonshotai/kimi-k2.7-code",  Provider.OPENROUTER, 0.71,  3.50, family="kimi", weights="kimi-k2.7-code")
    OR_KIMI_K3           = ModelSpec("moonshotai/kimi-k3",         Provider.OPENROUTER, 2.90, 15.00, family="kimi", quirks=_NO_TEMP, weights="kimi-k3")
    OR_MINIMAX_M3        = ModelSpec("minimax/minimax-m3",         Provider.OPENROUTER, 0.24,  0.96, family="minimax")  # PROMO (-60%)
    # Verified ABSENT from OpenRouter 2026-07-30 (so their direct rows above
    # have no US twin — us_route() correctly returns None): glm-4.7-flashx
    # (only glm-4.7-flash is listed) and kimi-k2.7-code-highspeed (a serving
    # mode of kimi-k2.7-code upstream, not a separate model).

    # ──────── AWS Bedrock (US-hosted us-east-1; auth via the standard AWS
    # credential chain, AWS_PROFILE). model_id MUST be the exact Bedrock
    # invocable id (captured from `aws bedrock list-*` 2026-07-18):
    # Claude is INFERENCE_PROFILE-only → the `us.`-prefixed cross-region profile;
    # Qwen/DeepSeek/Nova/GLM/Kimi/Mistral/Gemma/GPT-OSS invoke on the bare
    # ON_DEMAND id. Prices are us-east-1 list ($/1M tokens, verified 2026-07-18 via
    # AWS Bedrock pricing) where known; 0.0 = UNTRACKED (new/exotic models whose
    # list price isn't published) — prices here are for cost-ESTIMATION/tracking
    # parity with the other providers. Chinese-origin weights here are
    # US-jurisdiction (Bedrock hosts in us-east-1), unlike the direct-mainland
    # Provider.MOONSHOT/ZAI/DEEPSEEK rows. ────────
    # -- Claude (inference-profile ids; prices = Anthropic list, match on Bedrock) --
    BEDROCK_CLAUDE_HAIKU_4_5 = ModelSpec(
        "us.anthropic.claude-haiku-4-5-20251001-v1:0", Provider.BEDROCK, 1.00, 5.00,
        max_context_len=200_000, family="claude", alignment_tier="strong",
        weights="claude-haiku-4-5")
    # Same NO_CUSTOM_TEMPERATURE quirk as the direct-API twins (Sonnet 5 /
    # Opus 4.8 / Fable 5 reject a custom temperature on Bedrock too).
    BEDROCK_CLAUDE_SONNET_5 = ModelSpec(
        "us.anthropic.claude-sonnet-5", Provider.BEDROCK, 3.00, 15.00,  # promo $2/$10 through 2026-08-31
        max_context_len=200_000, family="claude", alignment_tier="strong",
        quirks=_NO_TEMP, weights="claude-sonnet-5")
    BEDROCK_CLAUDE_OPUS_4_8 = ModelSpec(        # flagship-tier target/judge
        "us.anthropic.claude-opus-4-8", Provider.BEDROCK, 5.00, 25.00,
        max_context_len=200_000, family="claude", alignment_tier="strong",
        quirks=_NO_TEMP, weights="claude-opus-4-8")
    BEDROCK_CLAUDE_FABLE_5 = ModelSpec(         # newest flagship; list price unpublished → UNTRACKED
        "us.anthropic.claude-fable-5", Provider.BEDROCK,
        max_context_len=200_000, family="claude", alignment_tier="strong",
        quirks=_NO_TEMP | _THINKING_BUDGET, weights="claude-fable-5")
    # -- Amazon Nova (inference-profile ids; Nova family caps output at 10000) --
    BEDROCK_NOVA_MICRO = ModelSpec(             # cheapest text (fast judge candidate)
        "us.amazon.nova-micro-v1:0", Provider.BEDROCK, 0.035, 0.14,
        max_context_len=128_000, family="nova", max_output_tokens=5_000)
    BEDROCK_NOVA_LITE = ModelSpec(              # cheap multimodal (ON_DEMAND)
        "us.amazon.nova-lite-v1:0", Provider.BEDROCK, 0.06, 0.24,
        family="nova", max_output_tokens=5_000)  # Nova hard cap 10000; clamp well under
    BEDROCK_NOVA_PRO = ModelSpec(               # stronger multimodal
        "us.amazon.nova-pro-v1:0", Provider.BEDROCK, 0.80, 3.20,
        max_context_len=300_000, family="nova", max_output_tokens=5_000)
    # -- Open-weight families on Bedrock (US-hosted → NOT PRC-jurisdiction). Prices
    #    UNTRACKED (per-model list not published for these newer ids). --
    BEDROCK_QWEN3_VL_235B = ModelSpec(          # ON_DEMAND VLM target (bare id)
        "qwen.qwen3-vl-235b-a22b", Provider.BEDROCK,
        family="qwen", alignment_tier="mid")
    BEDROCK_DEEPSEEK_V3_2 = ModelSpec(          # ON_DEMAND permissive capability/judge
        "deepseek.v3.2", Provider.BEDROCK,
        family="deepseek", alignment_tier="weak", weights="deepseek-v3.2")
    BEDROCK_GPT_OSS_120B = ModelSpec(           # OpenAI open-weight (120B)
        "openai.gpt-oss-120b-1:0", Provider.BEDROCK,
        max_context_len=128_000, family="gpt-oss")
    BEDROCK_GLM_5 = ModelSpec(                  # Z.AI GLM-5 (US-hosted here)
        "zai.glm-5", Provider.BEDROCK,
        family="glm", alignment_tier="mid", weights="glm-5")
    BEDROCK_KIMI_K2_5 = ModelSpec(              # Moonshot Kimi K2.5 (US-hosted here)
        "moonshotai.kimi-k2.5", Provider.BEDROCK,
        max_context_len=262_144, family="kimi")
    BEDROCK_MISTRAL_LARGE_3 = ModelSpec(        # Mistral Large 3 (675B)
        "mistral.mistral-large-3-675b-instruct", Provider.BEDROCK,
        family="mistral", alignment_tier="mid")
    BEDROCK_GEMMA_3_27B = ModelSpec(            # Google Gemma-3 27B (open weight)
        "google.gemma-3-27b-it", Provider.BEDROCK,
        max_context_len=128_000, family="gemma", alignment_tier="mid")
    # ── Expansion 2026-07-19 (ids verified against `aws bedrock
    #    list-inference-profiles` / `list-foundation-models`). Rule: a model WITH a us.-prefixed
    #    inference profile invokes on that profile id (Claude/Nova/Llama/Pixtral);
    #    models WITHOUT one invoke on the bare ON_DEMAND id (Gemma/GPT-OSS/Qwen/
    #    GLM/Kimi/DeepSeek/Mistral-Large). max_output_tokens: Nova caps at 10000
    #    (clamp 5000); Llama/Pixtral/Mistral cap output at 8192 on Bedrock.
    #    Prices in this block: UNTRACKED (0.0 — cost records as $0) except
    #    Nova Premier; fill from the AWS pricing page when cost tracking on
    #    these rows starts to matter. --
    # -- MULTIMODAL (vision) targets across MORE vendors — the target diversity a
    #    VLM-jailbreak paper wants (image-rendered attacks reach Claude/Nova/Qwen +
    #    now Mistral-Pixtral / Meta-Llama-Vision / Meta-Llama-4 / Gemma). --
    BEDROCK_PIXTRAL_LARGE = ModelSpec(          # Mistral Pixtral Large (multimodal; inference-profile)
        "us.mistral.pixtral-large-2502-v1:0", Provider.BEDROCK,
        max_context_len=128_000, max_output_tokens=8_192,
        family="pixtral", alignment_tier="mid")
    BEDROCK_LLAMA_3_2_90B_VISION = ModelSpec(   # Meta Llama 3.2 90B Vision (inference-profile)
        "us.meta.llama3-2-90b-instruct-v1:0", Provider.BEDROCK,
        max_context_len=128_000, max_output_tokens=8_192,
        family="llama", alignment_tier="mid")
    BEDROCK_LLAMA_3_2_11B_VISION = ModelSpec(   # Meta Llama 3.2 11B Vision (inference-profile)
        "us.meta.llama3-2-11b-instruct-v1:0", Provider.BEDROCK,
        max_context_len=128_000, max_output_tokens=8_192,
        family="llama", alignment_tier="mid")
    BEDROCK_LLAMA_4_MAVERICK = ModelSpec(       # Meta Llama 4 Maverick (multimodal MoE; inference-profile)
        "us.meta.llama4-maverick-17b-instruct-v1:0", Provider.BEDROCK,
        max_output_tokens=8_192, family="llama4", alignment_tier="mid")
    BEDROCK_LLAMA_4_SCOUT = ModelSpec(          # Meta Llama 4 Scout (multimodal MoE; inference-profile)
        "us.meta.llama4-scout-17b-instruct-v1:0", Provider.BEDROCK,
        max_output_tokens=8_192, family="llama4", alignment_tier="mid")
    BEDROCK_NOVA_PREMIER = ModelSpec(           # strongest Nova multimodal (inference-profile)
        "us.amazon.nova-premier-v1:0", Provider.BEDROCK, 0.80, 3.20,
        max_context_len=1_000_000, family="nova", max_output_tokens=5_000)
    BEDROCK_GEMMA_3_12B = ModelSpec(            # Google Gemma-3 12B (open weight, multimodal)
        "google.gemma-3-12b-it", Provider.BEDROCK,
        max_context_len=128_000, family="gemma", alignment_tier="mid")
    # -- Additional strong open-weight TEXT target (text-encoding attacks) --
    BEDROCK_LLAMA_3_3_70B = ModelSpec(          # Meta Llama 3.3 70B (open text; inference-profile)
        "us.meta.llama3-3-70b-instruct-v1:0", Provider.BEDROCK,
        max_context_len=128_000, max_output_tokens=8_192,
        family="llama", alignment_tier="mid")
    # -- Safety-classifier (guard/judge) models on Bedrock — OpenAI open "safeguard"
    #    reasoning models; candidates for a Bedrock-served input guard or a
    #    judge-robustness lens (bare ON_DEMAND id). NOT attack targets. --
    BEDROCK_GPT_OSS_SAFEGUARD_20B = ModelSpec(
        "openai.gpt-oss-safeguard-20b", Provider.BEDROCK,
        max_context_len=128_000, family="gpt-oss-safeguard")
    BEDROCK_GPT_OSS_SAFEGUARD_120B = ModelSpec(
        "openai.gpt-oss-safeguard-120b", Provider.BEDROCK,
        max_context_len=128_000, family="gpt-oss-safeguard")

    # ──────── Local (HF transformers in-process) ────────
    LLAMA3       = ModelSpec("llama3",                                   Provider.LOCAL)
    LLAMA3_1     = ModelSpec("llama3.1",                                 Provider.LOCAL)
    LLAMA3_8B    = ModelSpec("meta-llama/Meta-Llama-3-8B-Instruct",      Provider.LOCAL)
    LLAMA3_2_1B  = ModelSpec("meta-llama/Llama-3.2-1B-Instruct",         Provider.LOCAL)
    LLAMA3_2_3B  = ModelSpec("meta-llama/Llama-3.2-3B-Instruct",         Provider.LOCAL)
    MISTRAL_7B   = ModelSpec("mistralai/Mistral-7B-Instruct-v0.2",       Provider.LOCAL)
    PHI_3_MINI   = ModelSpec("microsoft/Phi-3-mini-4k-instruct",         Provider.LOCAL)

    # ──────── SLURM cluster (vLLM-served) — max_context_len populated ────────
    LLAMA3_1_8B_CLUSTER = ModelSpec(
        "meta-llama/Llama-3.1-8B-Instruct", Provider.SLURM_CLUSTER,
        max_context_len=131_072)               # Llama-3.1 long-context
    LLAMA3_8B_CLUSTER = ModelSpec(
        "meta-llama/Meta-Llama-3-8B-Instruct", Provider.SLURM_CLUSTER,
        max_context_len=8_192)                 # Llama-3 (8K)
    VICUNA_13B_CLUSTER = ModelSpec(
        "lmsys/vicuna-13b-v1.5", Provider.SLURM_CLUSTER,
        max_context_len=4_096)                 # Vicuna v1.5 (4K)
    # ── Safety-tuned open text-LLM targets, cross-family: Meta Llama-3.1-8B
    #    (above) + Alibaba Qwen2.5-7B + Google Gemma-2-9B.
    QWEN2_5_7B_INSTRUCT = ModelSpec(
        "Qwen/Qwen2.5-7B-Instruct", Provider.SLURM_CLUSTER,
        max_context_len=32_768, family="qwen", alignment_tier="mid")
    GEMMA2_9B_IT = ModelSpec(
        "google/gemma-2-9b-it", Provider.SLURM_CLUSTER,
        max_context_len=8_192, family="gemma", alignment_tier="mid")
    PIXTRAL_12B  = ModelSpec("mistralai/Pixtral-12B-2409",      Provider.SLURM_CLUSTER,
        family="mistral", alignment_tier="weak")
    LLAVA_7B     = ModelSpec("llava-hf/llava-1.5-7b-hf",        Provider.SLURM_CLUSTER)
    QWEN2_5_VL_7B = ModelSpec("Qwen/Qwen2.5-VL-7B-Instruct",    Provider.SLURM_CLUSTER,
        family="qwen", alignment_tier="mid")
    # Cluster-served VLMs: safety-aligned (Meta) + cross-family (OpenGVLab).
    LLAMA3_2_11B_VISION = ModelSpec(
        "meta-llama/Llama-3.2-11B-Vision-Instruct", Provider.SLURM_CLUSTER,
        family="llama", alignment_tier="strong")
    INTERNVL3_8B = ModelSpec(   # needs trust_remote_code (set in cluster config)
        "OpenGVLab/InternVL3-8B", Provider.SLURM_CLUSTER,
        family="internvl", alignment_tier="mid")
    QWEN3_VL_8B_INSTRUCT = ModelSpec(
        # Recency control, sibling of QWEN2_5_VL_7B (same family, newer
        # generation). Native 256K context (expandable to 1M) — no arch
        # ceiling concern at our max_model_len default (20480).
        "Qwen/Qwen3-VL-8B-Instruct", Provider.SLURM_CLUSTER,
        family="qwen", alignment_tier="mid")

    # ──────── Guard model baselines (safety classifiers — inspect/judge, not
    # attack targets). `alignment_tier` deliberately left None
    # for all four: that field means "how aligned is this model against being
    # attacked as a target" (task.py stamps it only from `target_llm`), which
    # doesn't apply to a guard's role — same precedent as the judge models
    # below (HARMBENCH_LLAMA_2_13B_CLS / LLAMA_3_3_70B_INSTRUCT), which also
    # leave it unset. `family` is still set (base-architecture provenance).
    GUARDREASONER_VL_7B = ModelSpec(
        # Fine-tune of Qwen2.5-VL-7B-Instruct (R-SFT + online RL) — mirrors
        # QWEN2_5_VL_7B's serving profile exactly. Chat/reasoning model:
        # emits <think>...</think> + <result>...</result>, NOT a passthrough
        # classifier — uses the tokenizer's own (Qwen2.5-VL) baked-in chat
        # template like its base model.
        "yueliu1999/GuardReasoner-VL-7B", Provider.SLURM_CLUSTER,
        family="qwen")
    LLAMA_GUARD_4_12B = ModelSpec(
        # Llama4ForConditionalGeneration — dense 12B pruned from Llama 4
        # Scout (shared expert retained, routed experts densified; NOT MoE
        # at inference). Natively multimodal, no trust_remote_code. Tokenizer
        # ships its own safety-taxonomy chat template (categories baked in),
        # so NOT passthrough — see conf/llm/llama_guard_4_12b.yaml for the
        # TODO on how that template surfaces via vLLM's OpenAI endpoint.
        "meta-llama/Llama-Guard-4-12B", Provider.SLURM_CLUSTER,
        family="llama")
    LLAMA_GUARD_3_8B = ModelSpec(
        # Llama-3.1-8B base, fine-tuned for content-safety classification.
        # Text-only. max_context_len mirrors LLAMA3_1_8B_CLUSTER's known
        # Llama-3.1 ceiling (131072) — same base architecture. Tokenizer
        # ships its own safety-taxonomy chat template (like Llama Guard 4),
        # so NOT passthrough.
        "meta-llama/Llama-Guard-3-8B", Provider.SLURM_CLUSTER,
        max_context_len=131_072, family="llama")
    WILDGUARD = ModelSpec(
        # Mistral-7B-v0.3 base. NOT a chat-tuned classifier — expects one
        # exact pre-formatted instruction string (system + user rules +
        # "Human user:"/"AI assistant:" fields + "Answers: [/INST]"), quoted
        # in conf/llm/wildguard.yaml. Same situation as
        # HARMBENCH_LLAMA_2_13B_CLS: passthrough, caller supplies the raw
        # formatted prompt as a single message. max_context_len left
        # unset — Mistral-7B-v0.3's config.json max_position_embeddings is
        # disputed between 32768 (commonly cited practical ceiling) and
        # 131072 (some sources); repo is gated so config.json couldn't be
        # fetched to confirm. Our default max_model_len (20480) is under
        # either number, so this is safe either way — but don't rely on
        # this field to catch an overrun for this model.
        "allenai/wildguard", Provider.SLURM_CLUSTER,
        family="mistral", chat_template="passthrough")

    # Open-source judge candidates (added 2026-07-13) — response-harm classifiers.
    QWEN3GUARD_GEN_8B = ModelSpec(
        # Generative guard; safety taxonomy baked into the tokenizer's own chat
        # template. Two-turn [user, assistant] → "Safety: Safe/Unsafe/Controversial".
        # 119 languages. No trust_remote_code (Gen variant).
        "Qwen/Qwen3Guard-Gen-8B", Provider.SLURM_CLUSTER, family="qwen")
    SHIELDLM_7B = ModelSpec(
        # InternLM2-7B base — needs trust_remote_code (conf). Passthrough: caller
        # supplies the [Answer]/[Analysis] instruction template as one message.
        "thu-coai/ShieldLM-7B-internlm2", Provider.SLURM_CLUSTER,
        family="internlm", chat_template="passthrough")
    MD_JUDGE_V0_1 = ModelSpec(
        # Mistral-7B-v0.1 classifier (SALAD-Bench). Passthrough: caller supplies
        # the [INST] task template; outputs first-line safe/unsafe + O-category.
        "OpenSafetyLab/MD-Judge-v0.1", Provider.SLURM_CLUSTER,
        family="mistral", chat_template="passthrough")
    THINKGUARD = ModelSpec(
        # Llama-Guard-3-8B + critique fine-tune; native Llama-Guard-3 chat
        # template. Two-turn [user, assistant] → safe/unsafe (+ critique).
        "Rakancorle1/ThinkGuard", Provider.SLURM_CLUSTER,
        max_context_len=131_072, family="llama")

    # Judge models for canonical benchmark evaluators
    HARMBENCH_LLAMA_2_13B_CLS = ModelSpec(
        "cais/HarmBench-Llama-2-13b-cls", Provider.SLURM_CLUSTER,
        max_context_len=2_048,                 # Llama-2 architecture (HARD)
        # Fine-tuned classifier — training prompts are pre-wrapped in
        # Llama-2's [INST] <<SYS>>...[/INST] syntax verbatim. Llama-2's
        # tokenizer doesn't ship a chat template, so we emit content as-is.
        chat_template="passthrough")
    LLAMA_3_3_70B_INSTRUCT = ModelSpec(
        "meta-llama/Llama-3.3-70B-Instruct", Provider.SLURM_CLUSTER,
        max_context_len=131_072)               # Llama-3.3 long-context

    # Open general-LLM RUBRIC judges (added 2026-07-13) — capable general models
    # applying the HarmBench classifier rubric (faithful rule-appliers, vs. the
    # purpose-built guards' own taxonomies). Chat-mode; no passthrough — native
    # chat templates.
    HERMES_4_70B = ModelSpec(
        # Llama-3.1-70B base; "neutral alignment" (steerable, low refusal by design).
        "NousResearch/Hermes-4-70B", Provider.SLURM_CLUSTER,
        family="llama", max_context_len=131_072)
    GLM_4_5_AIR = ModelSpec(
        # 106B MoE (12B active); capable + strong zh. trust_remote_code (conf).
        "zai-org/GLM-4.5-Air", Provider.SLURM_CLUSTER,
        family="glm", max_context_len=131_072)
    COMMAND_A = ModelSpec(
        # Cohere 111B dense; CONTEXTUAL/STRICT safety dial, steerable general judge.
        "CohereLabs/c4ai-command-a-03-2025", Provider.SLURM_CLUSTER,
        family="cohere", max_context_len=131_072)

    # Large-MoE RUBRIC judges (added 2026-07-15) — the capability ceiling of the
    # open judge pool; need multi-GPU tensor-parallel serving (won't fit clusters
    # whose QOS caps one GPU per job). Same HarmBench-rubric chat-judge path as
    # the 70B tier above; capability drives recall of buried encoded harm. Each
    # needs a serve-smoke before real use.
    QWEN3_235B_A22B_INSTRUCT = ModelSpec(
        # 235B/22B-active MoE, bf16; strongest open Chinese-language model.
        "Qwen/Qwen3-235B-A22B-Instruct-2507", Provider.SLURM_CLUSTER,
        family="qwen", max_context_len=262_144)
    DEEPSEEK_V3_2_EXP = ModelSpec(
        # 671B MoE, NATIVE fp8; permissive + published ASR-judge precedent (2603.17368).
        # Sparse attention (DSA) -> vLLM serve-smoke required first.
        "deepseek-ai/DeepSeek-V3.2-Exp", Provider.SLURM_CLUSTER,
        family="deepseek", max_context_len=163_840, weights="deepseek-v3.2")
    KIMI_K2_INSTRUCT = ModelSpec(
        # 1T/32B-active MoE, block-fp8; lowest documented general-harm refusal ->
        # permissive-ceiling judge. Needs a full high-memory GPU node (fp8 ~1 TB).
        "moonshotai/Kimi-K2-Instruct", Provider.SLURM_CLUSTER,
        family="kimi", max_context_len=131_072)
    # (Mistral-Large-3 considered + dropped 2026-07-15: 675B on-the-fly fp8 OOM
    #  risk + slow download; its minimal-alignment/permissive axis is already
    #  covered by Kimi-K2 + DeepSeek-V3.2.)

    # ──────── Accessors ────────────────────────────────────────────────
    # Same API surface as before: model.model_id, model.provider, etc.
    # Just reading from the dataclass instead of a positional tuple.

    @property
    def model_id(self) -> str:
        return self.value.model_id

    @property
    def provider(self) -> Provider:
        return self.value.provider

    @property
    def input_price(self) -> float:
        return self.value.input_price

    @property
    def output_price(self) -> float:
        return self.value.output_price

    @property
    def max_context_len(self) -> Optional[int]:
        """Architectural max context (from upstream config.json), None if unknown.

        Used by `LLMConfig` validator to reject `cluster.max_model_len`
        values that would cause vLLM to refuse at server startup (e.g.,
        Llama-2's RoPE positional encoding is undefined beyond 2048).
        """
        return self.value.max_context_len

    @property
    def max_output_tokens(self) -> Optional[int]:
        """Provider-enforced output ceiling (maxTokens), None if uncapped.

        BedrockService clamps its requested maxTokens to this so a model with
        a hard output cap below the global default (e.g. Amazon Nova = 10000
        vs the default 16384) doesn't 400 with a ValidationException.
        """
        return self.value.max_output_tokens

    @property
    def chat_template(self) -> Optional[str]:
        """Chat-template NAME for self-served (vLLM) deployment.

        A model fact the consumer passes into its serving layer's cluster
        config (the serving layer resolves the name to a .jinja file it
        ships). None means "use the tokenizer's baked-in chat template"
        (vLLM derives it automatically — correct for modern chat-tuned
        models). Set explicitly for classifiers and legacy models whose
        tokenizer doesn't ship a template. YAML can override per-deployment
        via `cluster.chat_template` but the source of truth is this field.
        """
        return self.value.chat_template

    @property
    def family(self) -> Optional[str]:
        """Model family label (e.g. 'qwen', 'llama') — stamped into results.json."""
        return self.value.family

    @property
    def alignment_tier(self) -> Optional[str]:
        """Coarse safety-alignment label ('strong'/'mid'/'weak'), or None."""
        return self.value.alignment_tier

    @property
    def weights(self) -> Optional[str]:
        """Identity of the underlying weights when this model is reachable on
        more than one serving route; None for single-route (closed) models."""
        return self.value.weights

    @property
    def jurisdiction(self) -> str:
        """Where THIS route processes data: 'us' / 'prc' / 'self'."""
        return self.provider.jurisdiction

    def has_quirk(self, q: ModelQuirk) -> bool:
        """Whether this model has an API-side behavior quirk."""
        return q in self.value.quirks

    # ------------------------------------------------------------------
    # Multi-route lookup — the same open weights on different endpoints.
    # Transport facts only; the DECISION of which route a payload may use is
    # the consumer's policy, never this package's.
    # ------------------------------------------------------------------

    def route_twins(self) -> tuple:
        """Every OTHER registry row serving the same weights, in registry
        order. Empty for single-route models.

        Example: ``LLMModel.GLM_5_2.route_twins()`` → the OpenRouter row
        (US-jurisdiction host of the same weights).
        """
        if not self.weights:
            return ()
        return tuple(
            m for m in type(self)
            if m is not self and m.weights == self.weights
        )

    def routes(self) -> tuple:
        """This row plus all its route twins — every way to reach these
        weights, in registry order."""
        if not self.weights:
            return (self,)
        return tuple(m for m in type(self) if m.weights == self.weights)

    def us_route(self) -> Optional["LLMModel"]:
        """The US-jurisdiction route to these weights, or None if there is
        none registered. Returns self when this route is already US.

        With several US routes the tie-break is ``_US_ROUTE_PREFERENCE``
        (OpenRouter first — one API key, versus Bedrock's account setup);
        filter ``routes()`` yourself to pin a specific host. A self-served
        route is reported by ``self_route()`` instead, since "no third party
        at all" is a different guarantee than "a US third party".
        """
        if self.jurisdiction == "us":
            return self
        candidates = [m for m in self.route_twins() if m.jurisdiction == "us"]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda m: (
                _US_ROUTE_PREFERENCE.index(m.provider)
                if m.provider in _US_ROUTE_PREFERENCE
                else len(_US_ROUTE_PREFERENCE)
            ),
        )

    def self_route(self) -> Optional["LLMModel"]:
        """The self-hosted route to these weights (local / SLURM cluster), or
        None if not registered. Returns self when already self-hosted."""
        if self.jurisdiction == "self":
            return self
        for m in self.route_twins():
            if m.jurisdiction == "self":
                return m
        return None

    @classmethod
    def from_string(cls, model_str: str) -> "LLMModel":
        """Resolve a string (model_id or enum name) to LLMModel.

        Tries model_id first, then enum-name (case-insensitive, normalizing
        '-' and '.' to '_'). Raises ValueError with a sample of available
        models on no match.
        """
        # Try by model_id first. The same id may be registered on several
        # serving routes (same weights local AND cluster-served) — an
        # ambiguous string cannot pick a provider, so fail loudly instead of
        # silently returning whichever row is defined first.
        matches = [m for m in cls if m.model_id == model_str]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            names = ", ".join(
                f"LLMModel.{m.name} ({m.provider.value})" for m in matches)
            raise ValueError(
                f"Ambiguous model id '{model_str}' — registered on multiple "
                f"serving routes: {names}. Pass the enum member (or enum "
                f"name) instead.")

        # Try by enum name (case insensitive, '-'/'.' → '_')
        model_str_upper = model_str.upper().replace("-", "_").replace(".", "_")
        for model in cls:
            if model.name == model_str_upper:
                return model

        available = [m.model_id for m in cls]
        raise ValueError(
            f"Unknown model: '{model_str}'. "
            f"Available: {available[:10]}... ({len(available)} total)")
