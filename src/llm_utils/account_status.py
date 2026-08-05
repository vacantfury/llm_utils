"""Account-status seam — provider credit balances + pure spend math.

Transport only, stateless: services fetch a point-in-time ``AccountStatus``
from the provider's own billing/balance endpoint (where one exists); the pure
helpers below turn a caller-supplied series of such snapshots into burn rate
and days-to-empty. Persistence of snapshots, alerting thresholds, and any
notification channel are the CONSUMER's job — this module never stores state
and never decides when a balance is "low".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence, Tuple


@dataclass
class AccountStatus:
    """Point-in-time snapshot of one provider account's credit state.

    ``supported=False`` means the provider exposes no balance endpoint usable
    with a normal API key (postpaid cloud billing, or no such API); every
    other field is then None. Amounts are in the account's own currency —
    DeepSeek can report CNY, most others USD — so compare like with like.
    """
    provider: str
    supported: bool
    balance: Optional[float] = None        # remaining prepaid credits
    currency: Optional[str] = None
    total_granted: Optional[float] = None  # lifetime credits purchased/granted
    total_used: Optional[float] = None     # lifetime spend
    raw: dict = field(default_factory=dict)  # untouched provider payload

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def burn_rate(
    snapshots: Sequence[Tuple[datetime, float]],
) -> Optional[float]:
    """Average spend per day from a series of (time, balance) snapshots.

    Sums only the balance DECREASES between consecutive snapshots, so a
    top-up mid-series raises the balance without corrupting the rate.
    Returns None when fewer than two snapshots or zero elapsed time.
    """
    if len(snapshots) < 2:
        return None
    ordered = sorted(snapshots, key=lambda s: s[0])
    spent = sum(
        max(0.0, prev_bal - cur_bal)
        for (_, prev_bal), (_, cur_bal) in zip(ordered, ordered[1:])
    )
    elapsed_days = (ordered[-1][0] - ordered[0][0]).total_seconds() / 86400.0
    if elapsed_days <= 0:
        return None
    return spent / elapsed_days


def days_to_empty(
    balance: Optional[float],
    rate_per_day: Optional[float],
) -> Optional[float]:
    """Projected days until the balance hits zero at the given burn rate.

    None when either input is unknown or the rate is zero/negative (no
    measurable burn → no finite projection).
    """
    if balance is None or rate_per_day is None or rate_per_day <= 0:
        return None
    return balance / rate_per_day
