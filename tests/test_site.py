"""The NFL board and the static dashboard.

The load-bearing property here is not layout, it is honesty: a research-only authority must
never produce a card that reads as an actionable edge.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nflmodel import authority as auth
from nflmodel import projections
from nflmodel.board import board_html
from nflmodel.board_nfl import build_board, build_card
from nflmodel.forecast import forecast_slate
from nflmodel.projections import OutlookError, division_winners, outlook, week_one_projections
from nflmodel.site import build_site

SLATE = [
    {"game": "LAC@KC", "home_team": "KC", "away_team": "LAC",
     "home_american": -175, "away_american": 150},
    {"game": "DAL@PHI", "home_team": "PHI", "away_team": "DAL",
     "home_american": -130, "away_american": 110},
]


def test_research_only_slate_produces_no_gems():
    """A gem asserts an actionable edge; this authority permits none, ever."""
    board = build_board(forecast_slate(SLATE))
    assert board.gems == 0
    assert all(card.gems == 0 for card in board.cards)


def test_every_card_carries_the_permitted_action():
    board = build_board(forecast_slate(SLATE))
    for card in board.cards:
        assert card.status_label in {a.value for a in auth.Action}
        assert card.status_label != auth.Action.BET.value


def test_board_meta_states_authority_and_unmet_gates():
    board = build_board(forecast_slate(SLATE))
    joined = " · ".join(board.meta)
    assert auth.Level.RESEARCH_ONLY.value in joined
    unmet = len(auth.REQUIRED_GATES) - len(auth.SATISFIED_GATES)
    assert f"{unmet} of {len(auth.REQUIRED_GATES)}" in joined


def test_zero_lambda_card_says_it_matches_the_market():
    card = build_card(forecast_slate(SLATE)["games"][0])
    assert "Matches the market" in card.headline
    assert card.headline_tone == "mut"


def test_unpairable_game_is_reported_not_dropped():
    payload = forecast_slate(SLATE + [{"game": "X@Y", "home_team": "Y", "away_team": "X"}])
    board = build_board(payload)
    assert len(board.cards) == 3
    assert any(card.status_label == "AVOID" for card in board.cards)


def test_avoid_card_is_not_counted_as_a_pick():
    payload = forecast_slate([{"game": "X@Y", "home_team": "Y", "away_team": "X"}])
    assert build_board(payload).picks == 0


def test_empty_slate_renders_an_honest_empty_state():
    html = board_html(build_board(forecast_slate([])))
    assert "No slate loaded" in html


# ── document ────────────────────────────────────────────────────────────────


def _render(tmp_path: Path, games=SLATE) -> str:
    slate = tmp_path / "slate.json"
    slate.write_text(json.dumps(games), encoding="utf-8")
    out = build_site(tmp_path / "index.html", slate)
    return out.read_text(encoding="utf-8")


def test_site_leads_with_the_authority_gate(tmp_path):
    html = _render(tmp_path)
    assert html.index('id="authority"') < html.index('id="board"')
    assert "may never emit a bet" in html


def test_site_lists_every_production_gate(tmp_path):
    html = _render(tmp_path)
    for gate in auth.REQUIRED_GATES:
        assert gate in html


def test_site_carries_the_shared_brand_contract(tmp_path):
    html = _render(tmp_path)
    assert "#08090F" in html          # canonical deep-navy ground
    assert "#9A6BFF" in html          # canonical violet brand
    assert "DM Sans" in html and "Roboto Condensed" in html
    assert "bd-card" in html          # shared board kernel
    assert html.count("fonts.googleapis.com") == 1


def test_site_carries_the_responsible_gambling_notice(tmp_path):
    html = _render(tmp_path)
    assert "1-800-GAMBLER" in html
    assert "not provide betting advice" in html


def test_site_builds_without_a_slate(tmp_path):
    out = build_site(tmp_path / "index.html", None)
    assert "No slate loaded" in out.read_text(encoding="utf-8")


def test_week_one_projection_feed_covers_every_team_once():
    assert outlook()["schema"] == "genesis/season-outlook/1"
    assert outlook()["authority"] == "RESEARCH_ONLY"
    games = week_one_projections()
    assert len(games) == 16
    teams = [game[side] for game in games for side in ("away_team", "home_team")]
    assert len(set(teams)) == 32
    assert all(game["home_win_probability"] + game["away_win_probability"] == 1.0 for game in games)
    rams_game = next(game for game in games if game["home_team"] == "LAR")
    assert rams_game["neutral_site"] is True


def test_division_projection_returns_one_winner_per_division():
    winners = division_winners()
    assert len(winners) == 8
    assert len({winner["team"] for winner in winners}) == 8


def test_site_renders_week_one_logos_and_division_winners(tmp_path):
    html = _render(tmp_path)
    assert 'id="projections"' in html
    assert 'id="divisions"' in html
    assert html.count('a.espncdn.com/i/teamlogos/nfl/500/') >= 40
    assert "Projected division winners" in html
    assert "Preseason command center" in html
    assert 'class="division-grid"' in html


def test_site_fails_closed_when_genesis_handoff_is_invalid(tmp_path, monkeypatch):
    artifact = tmp_path / "outlook.json"
    artifact.write_text('{"schema": "invalid"}', encoding="utf-8")
    monkeypatch.setattr(projections, "_OUTLOOK_PATH", artifact)
    projections.outlook.cache_clear()
    with pytest.raises(OutlookError, match="schema"):
        projections.outlook()
    projections.outlook.cache_clear()
