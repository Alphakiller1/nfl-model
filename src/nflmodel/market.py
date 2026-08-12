"""Odds math and paired de-vigging.

Kept dependency-free and duplicated deliberately rather than importing from
nfl-genesis: this is the deployable layer, and it must be able to price a quote
without the research stack installed.
"""

from __future__ import annotations

from dataclasses import dataclass


def american_to_decimal(american: float) -> float:
    a = float(american)
    if a == 0:
        raise ValueError("American odds of 0 are not a price")
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / abs(a))


def american_to_implied(american: float) -> float:
    """Implied probability WITH vig — the cost per $1 of payout."""
    return 1.0 / american_to_decimal(american)


def prob_to_american(p: float) -> int:
    if not 0.0 < p < 1.0:
        raise ValueError("Probability must be strictly within (0, 1)")
    return round(-100.0 * p / (1.0 - p)) if p >= 0.5 else round(100.0 * (1.0 - p) / p)


def devig_two_way(home_implied: float, away_implied: float) -> tuple[float, float]:
    """Proportional de-vig of a paired two-way market.

    Both sides are required. De-vigging one side against an assumed overround is
    guesswork, and a one-sided 'fair' price is the most common way a phantom edge
    is manufactured.
    """
    total = home_implied + away_implied
    if total <= 0:
        raise ValueError("Implied probabilities must be positive")
    if not 1.0 <= total <= 1.30:
        # Below 1 is an arbitrage or a bad parse; far above 1 is not a real
        # two-way market. Either way, refuse rather than emit a fair price.
        raise ValueError(f"Implausible overround {total:.4f}; refusing to de-vig")
    return home_implied / total, away_implied / total


@dataclass(frozen=True)
class PairedQuote:
    """A two-sided price with both American odds retained for display."""

    home_american: float
    away_american: float

    @property
    def home_fair(self) -> float:
        return devig_two_way(
            american_to_implied(self.home_american),
            american_to_implied(self.away_american),
        )[0]

    @property
    def away_fair(self) -> float:
        return 1.0 - self.home_fair

    @property
    def overround(self) -> float:
        return american_to_implied(self.home_american) + american_to_implied(
            self.away_american
        )
