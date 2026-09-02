"""The NFL board, the dashboard document, and the JSON contract.

The load-bearing property here is not layout, it is honesty: a research-only
authority must never produce a page that reads as an actionable edge, and the
authority must stay above the numbers no matter how much nicer the board looks
at the top.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from nflmodel import authority as auth
from nflmodel import divisions, export, matrix, site, teams
from nflmodel.board import board_html
from nflmodel.board_nfl import build_board, build_card
from nflmodel.sources.oddsapi import BookLine


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


def test_the_spread_and_moneyline_tiles_favour_the_same_team(slate):
    """Both are model numbers, so they must never name different favourites.
    Reading the moneyline off the published margin made "IND -1.5" sit beside
    "BAL 60%" on the same card — both correct, the card incoherent."""
    for projection in slate.projections:
        card = build_card(projection)
        tiles = {t.label: t.value for t in
                 next(g for g in card.groups if g.tag == "fullgame").tiles}
        spread, moneyline = tiles["Spread"], tiles["Moneyline"]
        if spread in {"PK", "—"} or moneyline == "—":
            continue
        assert spread.split()[0] == moneyline.split()[0], (spread, moneyline)


def test_filter_labels_carry_no_count_of_their_own(slate):
    """The kernel appends a match count, so "All 16" rendered as "ALL 16 16"."""
    for _, label in build_board(slate).filters:
        assert not any(character.isdigit() for character in label), label


def test_a_neutral_site_card_does_not_claim_a_home_field(slate):
    projection = slate.projections[0]
    neutral = replace(projection, neutral=True)
    card = build_card(neutral)
    rating = next(t for g in card.groups for t in g.tiles if t.label == "Rating gap")
    assert "home field" not in rating.state
    assert "neutral" in rating.state
    # ...and the breakdown drops the term rather than adding one that does not apply.
    breakdown = next(g for g in card.groups if g.tag == "breakdown")
    assert "Home field" not in {t.label for t in breakdown.tiles}
    assert len(breakdown.tiles) == len(matrix.STATS)


def test_a_card_headline_calls_the_difference_a_gap_not_an_edge(slate):
    headlines = " ".join(build_card(p).headline for p in slate.projections)
    assert "edge" not in headlines.lower() or "not an edge" in headlines.lower()


def test_the_matchup_shelf_is_explanatory_not_a_market(slate):
    """An unpriced explanatory shelf labelled 'no price' reads as a missing feed."""
    card = build_card(slate.projections[0])
    matchup = next(g for g in card.groups if g.tag == "matchup")
    assert matchup.market is False
    assert all(not tile.is_priced for tile in matchup.tiles)


def test_consensus_reference_is_not_counted_as_an_executable_price(slate):
    card = build_card(slate.projections[0])
    full_game = next(g for g in card.groups if g.tag == "fullgame")
    live_book = next(g for g in card.groups if g.tag == "livebook")
    assert full_game.priced == 0
    assert live_book.priced == 0


def test_verified_book_lines_get_their_own_first_class_shelf(slate):
    book = BookLine(
        "draftkings", "DraftKings", -4.5, 48.5, -190, 160,
        "2026-09-02T12:00:00Z", "2026-09-11T00:15:00Z",
    )
    projection = slate.projections[0]
    priced = replace(
        projection,
        book_name=book.book_title,
        book_key=book.book,
        book_margin=book.home_margin,
        book_total=book.total,
        book_last_update=book.last_update,
        home_moneyline=book.home_moneyline,
        away_moneyline=book.away_moneyline,
    )
    card = build_card(priced)
    assert card.groups[0].label == "DraftKings live lines"
    assert card.groups[0].priced == 3
    assert {tile.label for tile in card.groups[0].tiles} == {
        "Spread", "Total", "Moneyline",
    }


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
    order = ['id="authority"', 'id="board"', 'id="players"', 'id="ratings"', 'id="units"',
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
    assert "10.2274" in html and "9.7644" in html
    assert "49.56%" in html
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


def test_site_has_a_distinct_line_free_player_projection_layer(slate):
    html = _render(slate)
    assert "Offensive player &amp; kicker projections" in html
    assert "not sportsbook player lines" in html
    assert "a player who changed teams retains only 18%" in html


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
                "model_margin", "market_margin", "published_margin", "market_gap",
                "win_probability", "model_win_probability"):
        assert game[key] is not None


def test_export_carries_both_probabilities_and_they_are_distinguishable(slate):
    """One is the published price, one is the model's view. A consumer that
    cannot tell them apart will quote the market as the model."""
    game = next(g for g in export.payload(slate)["games"]
                if abs(g["market_gap"]) > 1.0)
    assert game["win_probability"] != game["model_win_probability"]


def test_every_matchup_tile_names_the_team_it_favours(slate):
    card = build_card(slate.projections[0])
    matchup = next(g for g in card.groups if g.tag == "matchup")
    for tile in matchup.tiles:
        assert tile.value.split()[0] in {slate.projections[0].home,
                                         slate.projections[0].away}, tile.value


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


def test_the_pages_evidence_matches_the_time_forward_audit():
    """`site.EVIDENCE` is hand-copied from `scripts/audit_regimes.py` output, so it
    is exactly the kind of thing that silently goes stale after a refit. Pinning
    it against the committed summary makes a drifted headline number a failing
    test rather than a wrong dashboard.

    This caught a real one: five surfaces truncated the ATS interval's lower
    bound to 47.81% when the value rounds to 47.82%.
    """
    import json
    import math
    from pathlib import Path

    fit = json.loads((Path(__file__).resolve().parents[1] / "reports"
                      / "regime_audit_2026.json").read_text(encoding="utf-8"))
    margin = fit["margin"]["baseline_50_50"]
    total = fit["total"]["baseline_shrunk"]
    wins, losses, pushes = margin["ats"]
    played = wins + losses
    rate = wins / played
    se = math.sqrt(rate * (1 - rate) / played)

    assert site.EVIDENCE["games"] == fit["protocol"]["games"]
    assert site.EVIDENCE["margin_model"] == pytest.approx(margin["mae"], abs=5e-5)
    assert site.EVIDENCE["margin_market"] == pytest.approx(
        fit["market"]["margin_mae"], abs=5e-5)
    assert site.EVIDENCE["margin_ratings_only"] == pytest.approx(
        fit["margin"]["rating_only"]["mae"], abs=5e-5)
    assert site.EVIDENCE["total_model"] == pytest.approx(total["mae"], abs=5e-5)
    assert site.EVIDENCE["total_market"] == pytest.approx(
        fit["market"]["total_mae"], abs=5e-5)
    assert site.EVIDENCE["total_league_mean"] == pytest.approx(
        fit["total"]["training_league_mean"]["mae"], abs=5e-5)
    assert site.EVIDENCE["ats"] == (wins, losses, pushes)
    assert site.EVIDENCE["ats_rate"] == pytest.approx(rate, abs=5e-5)
    low, high = site.EVIDENCE["ats_ci"]
    assert low == pytest.approx(rate - 1.96 * se, abs=5e-5)
    assert high == pytest.approx(rate + 1.96 * se, abs=5e-5)


def test_the_published_constants_match_the_measured_residuals():
    """MARGIN_SD is the forecast's residual SD, not the raw spread of NFL
    margins. Using the raw 14.32 would claim more uncertainty than the model has
    and push every win probability toward 50%."""
    import json
    from pathlib import Path

    from nflmodel import ratings as ratings_mod
    from nflmodel import totals as totals_mod

    fit = json.loads((Path(__file__).resolve().parents[1] / "reports"
                      / "regime_audit_2026.json").read_text(encoding="utf-8"))
    assert ratings_mod.MARGIN_SD == pytest.approx(
        fit["margin"]["baseline_50_50"]["residual_sd"], abs=0.005)
    assert ratings_mod.MARGIN_SD == matrix.MARGIN_SD
    assert totals_mod.TOTAL_SD == pytest.approx(
        fit["total"]["baseline_shrunk"]["residual_sd"], abs=0.005)
    # ...and NOT the raw margin spread, which is a full point higher.
    assert ratings_mod.MARGIN_SD < 14.32 - 0.5


# ── the sections added for information density ───────────────────────────────
def test_a_card_carries_the_factor_breakdown_that_explains_its_own_number(slate):
    """The board previously showed three market tiles and nothing about why.
    Every game with form now carries the five-family decomposition."""
    card = build_card(slate.projections[0])
    breakdown = next(g for g in card.groups if g.tag == "breakdown")
    # Five efficiency families plus the home-field term: every addend of the sum.
    assert len(breakdown.tiles) == len(matrix.STATS) + 1
    assert {t.label for t in breakdown.tiles} >= {"Home field"}
    assert breakdown.market is False
    # Ordered by size of effect: a reader scanning one card wants the reason.
    values = [float(t.value.split()[-1]) for t in breakdown.tiles]
    assert values == sorted(values, reverse=True)


def test_a_card_names_the_team_each_factor_favours(slate):
    projection = slate.projections[0]
    card = build_card(projection)
    breakdown = next(g for g in card.groups if g.tag == "breakdown")
    for tile in breakdown.tiles:
        assert tile.value.split()[0] in {projection.home, projection.away}


def test_a_side_shows_its_rating_and_record_not_a_bare_price(slate):
    """A "+150" beside a team name says nothing about whether the team is good;
    the price already has its own tile."""
    card = build_card(slate.projections[0], ratings_table=slate.table,
                      records={t: slate.record_for(t) for t in slate.forms})
    assert "+" in card.home.detail or "-" in card.home.detail


def test_the_disagreement_section_ranks_every_priced_game(slate):
    html = _render(slate)
    assert 'id="disagreements"' in html
    assert "Where the model differs from the market" in html
    # It must say what it is not, in the same breath as what it is.
    normalised = " ".join(html.split())
    assert "not</b> as a card" in normalised
    assert "49.56% of exactly these disagreements" in normalised


def test_the_disagreement_table_is_ordered_by_the_size_of_the_gap(slate):
    import re
    html = _render(slate)
    section = html[html.index('id="disagreements"'):html.index('id="ratings"')]
    gaps = [float(m) for m in re.findall(r'num score">[A-Z]{2,3} ([0-9.]+)</td>', section)]
    assert gaps
    assert gaps == sorted(gaps, reverse=True)


def test_the_playoff_section_publishes_both_conferences_and_the_cut_line(slate):
    html = _render(slate)
    assert 'id="seeds"' in html
    assert "AFC playoff field" in html and "NFC playoff field" in html
    assert "seed-cut" in html


def test_the_page_uses_the_shared_display_type_scale(slate):
    """The siblings all run the same clamp, weight and uppercase treatment. This
    page shipped a 36px sentence-case title and read like a different product."""
    css = site._PAGE_CSS
    assert "clamp(38px,6vw,64px)" in css      # hero, matching wnba-edge-model
    assert "clamp(26px,3.4vw,36px)" in css    # section titles
    assert "font-stretch:125%" in css
    assert "text-transform:uppercase" in css
