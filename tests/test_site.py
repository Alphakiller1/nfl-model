"""The NFL board, the dashboard document, and the JSON contract.

The load-bearing property here is not layout, it is honesty: a research-only
authority must never produce a page that reads as an actionable edge, and the
authority must stay above the numbers no matter how much nicer the board looks
at the top.
"""
from __future__ import annotations

import json

import pytest

from nflmodel import authority as auth
from nflmodel import divisions, export, matrix, site, teams
from nflmodel.board import board_html
from nflmodel.board_nfl import build_board, build_card


# ── board ────────────────────────────────────────────────────────────────────
def test_research_only_slate_produces_no_gems(slate):
    """A gem asserts an actionable edge; this authority permits none, ever."""
    board = build_board(slate)
    assert board.gems == 0
    assert all(card.gems == 0 for card in board.cards)


def test_every_card_carries_a_permitted_action_and_never_bet(slate):
    board = build_board(slate)
    assert board.cards
    for card in board.cards:
        assert card.status_label in {a.value for a in auth.Action}
        assert card.status_label != auth.Action.BET.value


def test_board_meta_states_the_authority_and_the_unmet_gates(slate):
    joined = " · ".join(build_board(slate).meta)
    assert auth.Level.RESEARCH_ONLY.value in joined
    unmet = len(auth.REQUIRED_GATES) - len(auth.SATISFIED_GATES)
    assert f"{unmet} of {len(auth.REQUIRED_GATES)}" in joined


def test_a_card_shows_projected_points_not_a_probability(slate):
    """The most prominent number on a card is read as a score, so it must be one."""
    card = build_card(slate.projections[0])
    assert "%" not in card.away.score
    assert float(card.away.score) > 0
    assert float(card.home.score) > 0


def test_the_favoured_side_is_the_one_projected_to_score_more(slate):
    card = build_card(slate.projections[0])
    assert card.away.favored != card.home.favored
    if card.home.favored:
        assert float(card.home.score) > float(card.away.score)
    else:
        assert float(card.away.score) > float(card.home.score)


def test_a_card_headline_calls_the_difference_a_gap_not_an_edge(slate):
    headlines = " ".join(build_card(p).headline for p in slate.projections)
    assert "edge" not in headlines.lower() or "not an edge" in headlines.lower()


def test_the_matchup_shelf_is_explanatory_not_a_market(slate):
    """An unpriced explanatory shelf labelled 'no price' reads as a missing feed."""
    card = build_card(slate.projections[0])
    matchup = next(g for g in card.groups if g.tag == "matchup")
    assert matchup.market is False
    assert all(not tile.is_priced for tile in matchup.tiles)


def test_a_priced_full_game_shelf_counts_as_priced(slate):
    card = build_card(slate.projections[0])
    full_game = next(g for g in card.groups if g.tag == "fullgame")
    assert full_game.priced > 0


def test_an_empty_slate_renders_an_honest_empty_state(slate):
    slate.projections = []
    html = board_html(build_board(slate))
    assert "No regular-season games found" in html


# ── document ─────────────────────────────────────────────────────────────────
def _render(slate) -> str:
    table = {t: 0.0 for t in teams.all_abbrs()}
    table.update(slate.table)
    outlooks = divisions.simulate(
        divisions.build_games(slate.schedule, table), table, simulations=60)
    return site.render(slate, outlooks)


def test_site_leads_with_the_authority_gate(slate):
    html = _render(slate)
    assert html.index('id="authority"') < html.index('id="board"')
    assert "may never emit a bet" in html


def test_site_sections_appear_in_the_documented_order(slate):
    html = _render(slate)
    order = ['id="authority"', 'id="board"', 'id="ratings"', 'id="units"',
             'id="divisions"', 'id="methodology"']
    positions = [html.index(marker) for marker in order]
    assert positions == sorted(positions)


def test_site_lists_every_production_gate(slate):
    html = _render(slate)
    for gate in auth.REQUIRED_GATES:
        assert gate.replace("_", " ") in html


def test_site_carries_the_shared_brand_contract(slate):
    html = _render(slate)
    assert "#08090F" in html          # canonical deep-navy ground
    assert "#9A6BFF" in html          # canonical violet brand
    assert "DM Sans" in html and "Roboto Condensed" in html
    assert "bd-card" in html          # shared board kernel
    assert html.count("fonts.googleapis.com") == 1


def test_page_css_introduces_no_colour_literal_of_its_own():
    """Every colour must come from chase_tokens.css, or the three products drift."""
    import re
    literals = re.findall(r"#[0-9A-Fa-f]{3,8}\b", site._PAGE_CSS)
    assert literals == []


def test_site_states_the_measured_gap_to_the_market(slate):
    html = _render(slate)
    assert "10.3134" in html and "9.8708" in html
    assert "49.85%" in html
    assert "52.38%" in html            # the breakeven it is being compared against


def test_site_carries_the_responsible_gambling_notice(slate):
    html = _render(slate)
    assert "1-800-GAMBLER" in html
    assert "not betting advice" in html


def test_site_never_renders_a_bet_action(slate):
    assert 'bd-status is-pos">BET<' not in _render(slate)


def test_site_publishes_the_unit_rankings_and_the_division_race(slate):
    html = _render(slate)
    assert "Offensive &amp; defensive power rankings" in html
    assert "Projected division winners" in html
    for division in teams.DIVISIONS:
        assert division in html


def test_site_documents_what_the_simulation_does_not_model(slate):
    # Whitespace-normalised: the copy is wrapped in the source, so a phrase that
    # straddles a line break is present on the page and absent from the string.
    html = " ".join(_render(slate).split())
    assert "nobody gets injured" in html
    assert "coin flip" in html
    assert "tails are thinner than reality" in html


# ── export contract ──────────────────────────────────────────────────────────
def test_export_carries_the_authority_with_the_numbers(slate):
    payload = export.payload(slate)
    assert payload["schema"] == export.SCHEMA
    assert payload["authority"] == auth.Level.RESEARCH_ONLY.value
    assert payload["may_bet"] is False
    assert payload["unmet_gates"]
    assert payload["spread_lambda"] == 0.0


def test_export_publishes_a_null_edge_rather_than_omitting_the_key(slate):
    """A consumer must be able to check for the edge, not infer it from absence."""
    game = export.payload(slate)["games"][0]
    assert "edge_points" in game
    assert game["edge_points"] is None
    assert game["edge_withheld_reason"]


def test_export_game_carries_scores_totals_and_both_margins(slate):
    game = export.payload(slate)["games"][0]
    for key in ("projected_home_score", "projected_away_score", "projected_total",
                "model_margin", "market_margin", "published_margin", "market_gap"):
        assert game[key] is not None


def test_export_team_row_reconciles_offence_defence_and_efficiency(slate):
    row = next(r for r in export.payload(slate)["teams"] if r["form"])
    form = row["form"]
    # Compared with a tolerance rather than for equality: all three are rounded
    # to three places on the way out, so the sum of two of them can differ from
    # the third in the last digit without anything being wrong.
    assert (form["offense_index"] + form["defense_index"]
            == pytest.approx(form["efficiency_rating"], abs=1e-3))


def test_export_is_json_serialisable(slate, tmp_path):
    path = export.write(slate, tmp_path / "board.json")
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["season"] == slate.season
    assert len(reloaded["games"]) == len(slate.projections)


def test_export_names_the_matrix_lineage_and_its_promotion_status(slate):
    constants = export.payload(slate)["constants"]
    assert constants["matrix_lineage"] == matrix.LINEAGE_VERSION
    assert constants["matrix_status"] == "CHALLENGER/UNPROMOTED"
