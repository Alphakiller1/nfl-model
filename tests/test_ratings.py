"""Power ratings, opponent adjustment and the relocation aliases.

The alias tests are not pedantry. nflverse keeps the abbreviation a team carried
at the time, so ten seasons of Raiders results arrive split across `OAK` and
`LV`. Rated as two franchises, both come out wrong and nothing in the output
looks broken -- the table is still 32 rows of plausible numbers.
"""

from __future__ import annotations

import pytest

from nflmodel import efficiency, preseason, ratings, teams


def game(home, away, home_points, away_points, week=1, season=2025, neutral=False):
    return ratings.Game(season=season, week=week, home=home, away=away,
                        home_points=home_points, away_points=away_points,
                        neutral=neutral)


# ── aliases ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("historical,current", [
    ("OAK", "LV"), ("SD", "LAC"), ("STL", "LA"), ("LAR", "LA"), ("WSH", "WAS"),
])
def test_relocations_fold_onto_todays_franchise(historical, current):
    assert teams.canonical(historical) == current


def test_from_rows_applies_the_alias_so_a_franchise_is_rated_once():
    rows = [
        {"season": 2019, "week": 1, "game_type": "REG", "home_team": "OAK",
         "away_team": "KC", "home_score": 24, "away_score": 20, "location": "Home"},
        {"season": 2025, "week": 1, "game_type": "REG", "home_team": "LV",
         "away_team": "KC", "home_score": 10, "away_score": 30, "location": "Home"},
    ]
    parsed = ratings.from_rows(rows)
    assert {g.home for g in parsed} == {"LV"}


def test_unknown_abbreviation_renders_rather_than_raising():
    assert teams.get("ZZZ").abbr == "?"


def test_every_division_has_exactly_four_teams():
    for division in teams.DIVISIONS:
        assert len(teams.members(division)) == 4
    assert len(teams.all_abbrs()) == 32


# ── the solve ────────────────────────────────────────────────────────────────
def test_an_unplayed_game_is_never_rated_as_nil_nil():
    """A blank score coerced to zero would poison every rating downstream."""
    rows = [{"season": 2026, "week": 1, "game_type": "REG", "home_team": "KC",
             "away_team": "LAC", "home_score": None, "away_score": None,
             "location": "Home"}]
    assert ratings.from_rows(rows) == []


def test_postseason_games_are_excluded():
    rows = [{"season": 2025, "week": 20, "game_type": "POST", "home_team": "KC",
             "away_team": "BUF", "home_score": 30, "away_score": 20,
             "location": "Home"}]
    assert ratings.from_rows(rows) == []


def test_ratings_are_centred_on_the_league_mean():
    games = [game("KC", "LAC", 30, 10), game("LAC", "KC", 17, 20),
             game("BUF", "NYJ", 28, 14), game("NYJ", "BUF", 13, 24)]
    table = ratings.build(games)
    assert sum(table.values()) == pytest.approx(0.0, abs=1e-6)


def test_the_better_team_rates_higher():
    games = [game("KC", "CAR", 35, 7, week=1), game("CAR", "KC", 10, 31, week=2)]
    table = ratings.build(games)
    assert table["KC"] > table["CAR"]


def test_home_field_is_removed_before_rating_so_hosting_earns_nothing():
    """Two teams that split a home-and-home by the home-field margin are equal."""
    hfa = ratings.HOME_FIELD_POINTS
    games = [game("KC", "BUF", 20 + hfa, 20, week=1),
             game("BUF", "KC", 20 + hfa, 20, week=2)]
    table = ratings.build(games, halflife=None)
    assert table["KC"] == pytest.approx(table["BUF"], abs=1e-6)


def test_capping_compresses_a_blowout_but_preserves_order():
    assert ratings.cap_margin(45.0) > ratings.cap_margin(30.0)
    assert ratings.cap_margin(45.0) < 45.0
    assert ratings.cap_margin(3.0) == pytest.approx(3.0, abs=0.02)


def test_cap_is_resolved_at_call_time_not_bound_at_import():
    """A sweep that reassigns the constant must actually change the result.

    Bound as a default argument this silently returned the import-time value and
    produced a perfectly flat parameter sweep -- a curve that looks like a
    finding and is a bug.
    """
    original = ratings.BLOWOUT_CAP
    try:
        ratings.BLOWOUT_CAP = 5.0
        assert ratings.cap_margin(40.0) == pytest.approx(5.0, abs=0.01)
    finally:
        ratings.BLOWOUT_CAP = original


def test_projected_margin_adds_home_field_and_neutral_removes_it():
    table = {"KC": 6.0, "CAR": -4.0}
    hosted = ratings.projected_margin(table, "KC", "CAR")
    neutral = ratings.projected_margin(table, "KC", "CAR", neutral=True)
    assert neutral == pytest.approx(10.0)
    assert hosted - neutral == pytest.approx(ratings.HOME_FIELD_POINTS)


def test_projected_margin_is_none_for_an_unrated_team():
    assert ratings.projected_margin({"KC": 6.0}, "KC", "ZZZ") is None


def test_win_probability_is_symmetric_and_bounded():
    assert ratings.win_probability(0.0) == pytest.approx(0.5)
    assert ratings.win_probability(7.0) + ratings.win_probability(-7.0) == pytest.approx(1.0)
    assert 0.0 < ratings.win_probability(-60.0) < ratings.win_probability(60.0) < 1.0


def test_rank_table_is_ordered_strongest_first():
    table = {"KC": 6.0, "CAR": -4.0, "BUF": 2.0}
    ranked = ratings.rank_table(table)
    assert [team for _, team, _ in ranked] == ["KC", "BUF", "CAR"]
    assert ranked[0][0] == 1


# ── blending the prior ───────────────────────────────────────────────────────
def test_live_weight_grows_with_games_played():
    assert preseason.live_weight(0) == 0.0
    assert preseason.live_weight(1) < preseason.live_weight(8)
    assert preseason.live_weight(6) == pytest.approx(0.5)      # K = 6


def test_live_weight_honours_the_module_constant_at_call_time():
    original = preseason.BLEND_K
    try:
        preseason.BLEND_K = 2.0
        assert preseason.live_weight(2) == pytest.approx(0.5)
    finally:
        preseason.BLEND_K = original


def test_blend_ratings_returns_the_prior_when_nothing_has_been_played():
    prior = {"KC": 5.0, "CAR": -5.0}
    blended = preseason.blend_ratings(prior, {"KC": -20.0, "CAR": 20.0}, {})
    assert blended == prior


def test_blend_ratings_moves_toward_the_live_estimate_as_games_accumulate():
    prior, live = {"KC": 5.0}, {"KC": 15.0}
    early = preseason.blend_ratings(prior, live, {"KC": 1.0})["KC"]
    late = preseason.blend_ratings(prior, live, {"KC": 16.0})["KC"]
    assert 5.0 < early < late < 15.0


# ── opponent adjustment ──────────────────────────────────────────────────────
def _line(game_id, team, opponent, epa, week=1):
    return efficiency.GameLine(season=2025, week=week, game_id=game_id, team=team,
                               opponent=opponent, plays=60.0, epa=epa,
                               first_down=0.30, explosive=0.06, sack=0.06,
                               turnover=0.02)


def test_adjustment_credits_an_offence_for_the_defence_it_faced():
    """Two offences post identical raw EPA; one did it against a better defence.

    The one that faced the tougher defence must adjust upward relative to the
    other. Without this, a team's schedule is silently part of its rating.
    """
    lines = [
        # KC posts 0.10 against CAR, whose defence is otherwise shredded.
        _line("g1", "KC", "CAR", 0.10), _line("g1", "CAR", "KC", -0.20),
        _line("g2", "BUF", "NYJ", 0.10), _line("g2", "NYJ", "BUF", 0.10),
        # NYJ's defence is good elsewhere; CAR's is bad elsewhere.
        _line("g3", "NYJ", "CAR", 0.05, week=2), _line("g3", "CAR", "NYJ", -0.25, week=2),
        _line("g4", "BUF", "CAR", 0.30, week=3), _line("g4", "CAR", "BUF", -0.15, week=3),
    ]
    offense, _ = efficiency.adjust(lines, "epa")
    assert offense["BUF"] > offense["KC"]


def test_a_team_game_with_no_plays_is_dropped_not_divided_by_zero():
    rows = [{"season": 2025, "week": 1, "team": "KC", "opponent_team": "LAC",
             "season_type": "REG", "game_id": "g", "attempts": 0, "carries": 0,
             "sacks_suffered": 0}]
    assert efficiency.game_lines(rows) == []


def test_game_lines_derive_per_play_rates_rather_than_totals():
    rows = [{"season": 2025, "week": 1, "team": "KC", "opponent_team": "LAC",
             "season_type": "REG", "game_id": "g", "attempts": 30, "carries": 25,
             "sacks_suffered": 5, "passing_epa": 6.0, "rushing_epa": 0.0,
             "passing_first_downs": 12, "rushing_first_downs": 6,
             "passing_20": 3, "rushing_20": 1, "passing_interceptions": 1,
             "fumbles_lost_total": 1}]
    line = efficiency.game_lines(rows)[0]
    assert line.plays == 60.0
    assert line.epa == pytest.approx(0.1)
    assert line.first_down == pytest.approx(18 / 60)
    assert line.explosive == pytest.approx(4 / 60)
    assert line.sack == pytest.approx(5 / 35)       # sacks per DROPBACK, not per play
    assert line.turnover == pytest.approx(2 / 60)
