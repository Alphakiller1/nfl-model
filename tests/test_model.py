"""Tests pin the two properties that matter: honest pricing, and no laundering."""

from __future__ import annotations

import math

import pytest

from nflmodel import authority as auth
from nflmodel.forecast import anchor, forecast_game, forecast_slate
from nflmodel.market import (
    PairedQuote,
    american_to_implied,
    devig_two_way,
    prob_to_american,
)
from nflmodel.teams import canonical

# ── market math ───────────────────────────────────────────────────────────────

def test_american_probability_round_trip() -> None:
    for american in (-350, -220, -110, 145, 260, 900):
        p = american_to_implied(american)
        assert prob_to_american(p) == pytest.approx(american, abs=1)


def test_even_money_is_ambiguous_and_resolves_to_minus_100() -> None:
    """+100 and -100 are the SAME probability, so the round trip cannot be 1:1.

    Both map to exactly 0.5. `prob_to_american` breaks the tie toward the
    favourite branch (p >= 0.5) and returns -100. Worth pinning: silently
    round-tripping +100 to -100 inside a larger pipeline would look like a sign
    bug rather than the tie-break it is.
    """
    assert american_to_implied(100) == pytest.approx(0.5)
    assert american_to_implied(-100) == pytest.approx(0.5)
    assert prob_to_american(0.5) == -100


def test_minus_110_is_the_standard_vigged_line() -> None:
    assert american_to_implied(-110) == pytest.approx(0.5238, abs=1e-4)


def test_paired_devig_sums_to_one_and_removes_the_overround() -> None:
    quote = PairedQuote(home_american=-150, away_american=130)
    assert quote.overround > 1.0
    assert quote.home_fair + quote.away_fair == pytest.approx(1.0)
    assert quote.home_fair < american_to_implied(-150)   # vig removed


def test_refuses_to_devig_an_implausible_overround() -> None:
    """A sub-1 total is an arb or a bad parse; emitting a 'fair' price hides it."""
    with pytest.raises(ValueError, match="Implausible overround"):
        devig_two_way(0.40, 0.40)


# ── anchoring ─────────────────────────────────────────────────────────────────

def test_lambda_zero_reproduces_the_market_exactly() -> None:
    """The published selection is lam = 0, so the forecast must BE the market."""
    for market in (0.10, 0.35, 0.5, 0.72, 0.94):
        assert anchor(market, structural=0.99, lam=0.0) == market


def test_lambda_moves_toward_but_not_past_the_structural_view() -> None:
    market, structural = 0.50, 0.70
    half = anchor(market, structural, lam=0.5)
    assert market < half < structural
    assert anchor(market, structural, lam=1.0) == pytest.approx(structural, abs=1e-9)


def test_missing_structural_view_leaves_the_market_untouched() -> None:
    assert anchor(0.61, structural=None, lam=0.5) == 0.61


# ── authority: the part that stops an unpromoted model becoming a bet ─────────

def test_current_authority_is_research_only_and_may_not_bet() -> None:
    a = auth.current()
    assert a.level is auth.Level.RESEARCH_ONLY
    assert a.may_bet is False
    assert a.unmet_gates            # and it can say exactly which


def test_unpromoted_model_can_never_emit_bet() -> None:
    a = auth.current()
    for edge in (-0.05, 0.0, 0.02, 0.10):
        assert a.action_for(edge, has_price=True) is auth.Action.MONITOR


def test_implausible_edge_is_reviewed_not_traded() -> None:
    a = auth.promote(set(auth.REQUIRED_GATES))
    assert a.may_bet is True
    assert a.action_for(0.40, has_price=True) is auth.Action.REVIEW


def test_no_price_means_avoid_even_when_promoted() -> None:
    a = auth.promote(set(auth.REQUIRED_GATES))
    assert a.action_for(None, has_price=False) is auth.Action.AVOID


def test_promotion_requires_every_gate() -> None:
    partial = set(auth.REQUIRED_GATES) - {"probability_space_clv_above_zero"}
    a = auth.promote(partial)
    assert a.level is auth.Level.RESEARCH_ONLY
    assert a.may_bet is False
    assert "probability_space_clv_above_zero" in a.unmet_gates


# ── slate contract ────────────────────────────────────────────────────────────

def test_slate_reports_market_price_and_monitor_action() -> None:
    out = forecast_slate([
        {"home_team": "KC", "away_team": "LAC", "home_american": -175, "away_american": 150},
    ])
    assert out["may_bet"] is False
    assert out["lam"] == 0.0
    game = out["games"][0]
    assert game["action"] == "MONITOR"
    assert game["edge_vs_market"] == 0.0        # lam=0 => no edge, by construction
    assert game["home_fair"] + game["away_fair"] == pytest.approx(1.0, abs=1e-5)


def test_unpriced_game_is_reported_not_silently_dropped() -> None:
    """A shorter list is indistinguishable from an empty slate; report the gap."""
    out = forecast_slate([
        {"home_team": "KC", "away_team": "LAC", "home_american": -175, "away_american": 150},
        {"home_team": "BUF", "away_team": "NYJ"},          # no price
    ])
    assert len(out["games"]) == 1
    assert len(out["skipped"]) == 1
    assert out["skipped"][0]["action"] == "AVOID"


def test_forecast_game_emits_american_odds_both_ways() -> None:
    f = forecast_game(game="LAC@KC", home_team="KC", away_team="LAC",
                      home_american=-175, away_american=150)
    assert f.home_american < 0 < f.away_american
    assert math.isclose(f.home_fair + f.away_fair, 1.0, abs_tol=1e-5)


def test_schedule_abbreviation_resolves_to_rating_abbreviation() -> None:
    """Kept from 067d9b0 and rewritten for the current API.

    That commit asserted it through `ratings.rating_for`, which read a vendored
    JSON; ratings are now solved live and the aliasing moved to `teams.canonical`.
    The property being defended is the same one and still matters: nflverse keeps
    the abbreviation a team carried at the time, so LAR and LA must never be rated
    as two franchises.
    """
    assert canonical("LAR") == canonical("LA") == "LA"
