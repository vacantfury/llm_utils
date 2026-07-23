"""Package-local exceptions.

Previously lived in the host research repos as ``src.utils.exceptions``; defined here so
the package stands alone. Reconciliation note (TODO task 1): verify class hierarchy
matches the host repos' originals before migrating consumers.
"""


class FatalModelError(Exception):
    """A model is unusable for the whole run (e.g. 404 / model not found)."""


class CreditsExhaustedError(Exception):
    """The provider account is out of credits / over its billing quota."""


class InvalidCredentialError(Exception):
    """The provider API key was rejected (invalid or revoked)."""
