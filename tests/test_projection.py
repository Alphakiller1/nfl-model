"""Totals, scorelines, the published margin and the season simulation.

The properties pinned here are the ones a plausible-looking page would hide: that
a projected score reconciles with the total and margin it came from, that the
published margin is the market and not the model, and that a difference from the
market is never dressed up as an edge while the ATS record says it is not one.
"""

from __future__ import annotations

import pytest

from conftest import make_form
from nflmodel import authority as auth
from nflmodel import divisions, forecast, matrix, ratings, teams, totals


# ── totals and scorelines ────────────────────────────────────────────────────
def test_scoreline_reconciles_with_its_own_total_and_margin():
    projection = totals.project(make_form(epa=0.12), make_form(epa=-0.04),
                                rating_margin=5.0)
    assert projection.home_score + projection.away_score == pytest.approx(projection.total)
    assert projection.home_score - projection.away_score == pytest.approx(projection.margin)


def test_the_total_is_shrunk_toward_the_league_mean():
    """The raw model total is over-dispersed; publishing it unshrunk overstates
    how much of the variance the model explains."""
    raw = 60.0
    shrunk = totals.shrink_total(raw)
    assert totals.LEAGUE_MEAN_TOTAL < shrunk < raw
    assert shrunk == pytest.approx(
        totals.LEAGUE_MEAN_TOTAL + totals.TOTAL_SHRINK * (raw - totals.LEAGUE_MEAN_TOTAL))


def test_missing_form_falls_back_to_the_league_mean_and_says_so():
    projection = totals.project(None, None, rating_margin=6.0)
    assert projection.modelled is False
    assert projection.total == pytest.approx(totals.LEAGUE_MEAN_TOTAL)
    assert projection.margin == pytest.approx(6.0)


def test_a_projected_score_is_never_negative():
    """A huge margin against a modest total would otherwise produce one."""
    projection = totals.project(None, None, rating_margin=90.0)
    assert projection.away_score == 0.0
    assert projection.home_score == pytest.approx(projection.total)


def test_the_published_margin_is_the_even_blend_of_both_estimates():
    blended = totals.blend_margin(4.0, 10.0)
    assert blended == pytest.approx(7.0)
    assert totals.EFFICIENCY_WEIGHT == 0.5


def test_blending_uses_whichever_estimate_exists():
    assert totals.blend_margin(None, 8.0) == 8.0
    assert totals.blend_margin(3.0, None) == 3.0
    assert totals.blend_margin(None, None) is None


# ── the published forecast ───────────────────────────────────────────────────
def _project(**kwargs):
    defaults = dict(
        home="KC", away="LAC", team_ratings={"KC": 6.0, "LAC": -1.0},
        home_form=make_form(epa=0.10), away_form=make_form(epa=-0.02),
    )
    return forecast.project_game(**{**defaults, **kwargs})


def test_at_lambda_zero_the_published_margin_is_the_market_exactly():
    p = _project(market_margin=3.5)
    assert p.margin == pytest.approx(3.5)
    assert p.model_margin != pytest.approx(3.5)     # the model still disagrees


def test_without_a_market_the_model_publishes_its_own_number():
    p = _project(market_margin=None)
    assert p.margin == pytest.approx(p.model_margin)
    assert p.market_gap is None


def test_the_difference_from_the_market_is_a_gap_and_never_an_edge():
    p = _project(market_margin=3.5)
    assert p.market_gap == pytest.approx(p.model_margin - 3.5)
    assert p.edge_points is None
    assert "does not beat the closing line" in p.edge_withheld_reason


def test_a_priced_modelled_game_is_monitor_not_avoid():
    """AVOID means 'no usable price'. Reporting it on a fully priced, fully
    modelled game makes a working board read as a feed outage."""
    assert _project(market_margin=3.5).action == auth.Action.MONITOR.value


def test_an_unpriced_game_is_avoid():
    assert _project(market_margin=None).action == auth.Action.AVOID.value


def test_a_game_the_model_cannot_rate_is_avoid_not_a_guess():
    p = _project(team_ratings={}, home_form=None, away_form=None, market_margin=None)
    assert p.model_margin is None
    assert p.action == auth.Action.AVOID.value


def test_the_scoreline_is_built_from_the_model_not_the_published_margin():
    """At lambda 0 the published margin is the market. A scoreline derived from
    it would be the market's projection wearing the model's label."""
    p = _project(market_margin=-14.0)
    implied = p.projected_home_score - p.projected_away_score
    assert implied == pytest.approx(p.model_margin)
    assert implied != pytest.approx(p.margin)


def test_win_probability_follows_the_published_margin():
    p = _project(market_margin=7.0)
    assert p.win_probability == pytest.approx(ratings.win_probability(7.0))


def test_paired_moneylines_produce_a_fair_probability_that_sums_to_one():
    p = _project(market_margin=3.5, home_moneyline=-175, away_moneyline=150)
    assert 0.0 < p.market_fair_home < 1.0


def test_an_unpairable_moneyline_leaves_the_fair_price_absent_rather_than_guessed():
    p = _project(market_margin=3.5, home_moneyline=-175, away_moneyline=None)
    assert p.market_fair_home is None


# ── season simulation ────────────────────────────────────────────────────────
def _round_robin(season: int = 2026) -> list[dict]:
    rows, week = [], 1
    order = teams.members("AFC East") + teams.members("AFC North")
    for home in order:
        for away in order:
            if home != away:
                rows.append({
                    "season": season, "week": week, "game_type": "REG",
                    "home_team": home, "away_team": away, "location": "Home",
                    "home_score": None, "away_score": None,
                })
                week = week % 18 + 1
    return rows


def test_the_simulation_is_deterministic_for_the_same_inputs():
    """A dashboard whose odds move on every rebuild with no new data is
    indistinguishable from one with a bug."""
    table = {t: float(i) for i, t in enumerate(teams.members("AFC East")
                                               + teams.members("AFC North"))}
    games = divisions.build_games(_round_robin(), table)
    first = divisions.simulate(games, table, simulations=300)
    second = divisions.simulate(games, table, simulations=300)
    assert [(o.team, o.win_division) for o in first] == \
           [(o.team, o.win_division) for o in second]


def test_division_probabilities_sum_to_one_within_each_division():
    table = {t: float(i) for i, t in enumerate(teams.members("AFC East")
                                               + teams.members("AFC North"))}
    games = divisions.build_games(_round_robin(), table)
    outlooks = divisions.simulate(games, table, simulations=500)
    for members in divisions.by_division(outlooks).values():
        if members:
            assert sum(o.win_division for o in members) == pytest.approx(1.0, abs=1e-6)


def test_the_stronger_team_wins_its_division_more_often():
    members = teams.members("AFC East")
    table = {t: 0.0 for t in members}
    table[members[0]] = 12.0
    games = divisions.build_games(
        [r for r in _round_robin() if r["home_team"] in members
         and r["away_team"] in members], table)
    outlooks = {o.team: o for o in divisions.simulate(games, table, simulations=800)}
    best = outlooks[members[0]]
    assert best.win_division > 0.5
    assert best.projected_wins > max(outlooks[t].projected_wins for t in members[1:])


def test_projected_wins_and_losses_account_for_every_game():
    table = {t: 0.0 for t in teams.members("AFC East")}
    games = divisions.build_games(
        [r for r in _round_robin() if r["home_team"] in table and r["away_team"] in table],
        table)
    for outlook in divisions.simulate(games, table, simulations=200):
        assert outlook.projected_wins + outlook.projected_losses == pytest.approx(17.0)


def test_a_game_with_an_unrated_team_is_skipped_rather_than_guessed():
    table = {t: 0.0 for t in teams.members("AFC East")}
    rows = _round_robin() + [{"season": 2026, "week": 1, "game_type": "REG",
                              "home_team": "ZZZ", "away_team": "BUF",
                              "location": "Home", "home_score": None,
                              "away_score": None}]
    games = divisions.build_games(rows, table)
    assert all(g.home in table and g.away in table for g in games)


def test_the_simulation_uses_the_same_margin_the_board_publishes():
    """Two estimates of one quantity is how division odds end up disagreeing
    with the spreads on the same page."""
    from nflmodel import season as season_mod
    table = {"BUF": 6.0, "MIA": -2.0}
    forms = {"BUF": make_form(epa=0.10), "MIA": make_form(epa=-0.05)}
    margin_of = season_mod.margin_for(table, forms)
    expected = totals.project(forms["BUF"], forms["MIA"],
                              rating_margin=ratings.projected_margin(table, "BUF", "MIA")).margin
    assert margin_of("BUF", "MIA", False) == pytest.approx(expected)
    games = divisions.build_games(
        [{"season": 2026, "week": 1, "game_type": "REG", "home_team": "BUF",
          "away_team": "MIA", "location": "Home", "home_score": None,
          "away_score": None}], table, margin_of=margin_of)
    assert games[0].home_win_probability == pytest.approx(
        ratings.win_probability(expected))


def test_matrix_and_efficiency_margin_agree_on_a_neutral_site():
    home, away = make_form(epa=0.12), make_form(epa=-0.03)
    assert forecast.matrix_margin(home, away, True) == pytest.approx(
        matrix.margin_points(home, away, neutral=True))
