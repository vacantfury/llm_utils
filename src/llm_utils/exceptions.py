"""Package-local exceptions.

Previously lived in the host research repos as ``src.utils.exceptions``; defined here so
the package stands alone. Hierarchy verified against the original in the
imaging_text_attacks host at v2.2.0 reconciliation.
"""


class FatalModelError(Exception):
    """
    Raised when a MODEL-SPECIFIC error occurs that should terminate work on that
    model immediately (e.g. Model ID not found / 404). Per-model, not
    per-account: other providers/models in the same run can still proceed, so the
    experiment runner records the affected task as failed rather than aborting the
    whole run. For account-global failures use ``AccountFatalError`` below.
    """
    pass


class AccountFatalError(Exception):
    """A provider-account / API-key-global failure that dooms EVERY remaining
    task that uses this key — an invalid key, or an exhausted credit balance.

    Unlike a per-model ``FatalModelError``, retrying other cells is pointless:
    the same key fails identically everywhere. So the right response is to abort
    the whole run FAST with an actionable message, instead of grinding each prompt
    into a mechanism-error and "completing" with a degenerate all-excluded result
    (the failure mode the mechanism-error sentinel protects correctness from, but
    at the cost of a useless full-length run).

    Raised from a service's error handler (``BaseLLMService._raise_if_account_fatal``)
    and RE-RAISED — not swallowed — by the experiment runner so it aborts the run.
    """
    pass


class InvalidCredentialError(AccountFatalError):
    """The provider rejected the API key: invalid, revoked, or unauthorized
    (HTTP 401 / ``invalid_api_key`` / ``authentication_error`` / Google's
    "API key not valid"). Will not recover mid-run — fix the key and rerun."""
    pass


class CreditsExhaustedError(AccountFatalError):
    """The provider accepted the key but the account is out of credits / over its
    billing quota: OpenAI ``insufficient_quota`` (surfaced as a 429),
    Anthropic "credit balance is too low", DeepSeek "Insufficient Balance",
    or a generic HTTP 402 payment-required. Top up the account and rerun."""
    pass
