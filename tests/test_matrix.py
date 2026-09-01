"""The logic matrix.

Two classes of test here, and the second is the one that would have caught the
bug this module was rewritten to fix.

* **Contract** -- weight groups are valid, features are all-or-nothing.
* **Sign and reconciliation** -- a better offence must raise the projection and a
  better defence must lower the opponent's, and the two indices must add to the
  efficiency rating exactly. The first version of this model fitted `def_epa`
  POSITIVE, which produced a perfectly plausible-looking ranking in which allowing
  more EPA per play made a defence look better. Nothing about the output shape
  reveals that; only asserting the direction does.
"""

from __future__ import annotations

import pytest

from conftest import average_form, make_form
from nflmodel import matrix


# ── contract ─────────────────────────────────────────────────────────────────
def test_weight_groups_are_valid():
    for name, group in matrix.GROUPS.items():
        matrix.validate_weight_group(group, name=name)


def test_weight_group_rejects_a_group_that_does_not_sum_to_one():
    with pytest.raises(matrix.WeightGroupError, match="sum to 1.0"):
        matrix.validate_weight_group({"a": 0.5, "b": 0.2})


def test_weight_group_rejects_negative_weights():
    with pytest.raises(matrix.WeightGroupError, match="negative"):
        matrix.validate_weight_group({"a": 1.4, "b": -0.4})


def test_incomplete_form_produces_no_projection():
    """A missing rate must never be silently treated as league average."""
    partial = matrix.TeamForm(off_epa=0.1)
    assert partial.complete() is False
    assert matrix.points(partial, average_form(), home=True) is None
    assert matrix.offense_index(partial) is None
    assert matrix.margin_points(partial, average_form()) is None


def test_pace_is_not_required_for_completeness():
    """Plays are carried for display but are deliberately not a model feature."""
    form = make_form(plays=None)
    assert form.complete() is True
    assert matrix.points(form, average_form(), home=True) is not None


# ── signs ────────────────────────────────────────────────────────────────────
def test_a_league_average_unit_indexes_at_zero():
    form = average_form()
    assert matrix.offense_index(form) == pytest.approx(0.0, abs=1e-9)
    assert matrix.defense_index(form) == pytest.approx(0.0, abs=1e-9)
    assert matrix.efficiency_rating(form) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("field,better,worse", [
    ("epa", 0.15, -0.15),
    ("first_down", 0.34, 0.24),
    ("explosive", 0.09, 0.03),
])
def test_a_better_offensive_rate_scores_more(field, better, worse):
    good = matrix.offense_index(make_form(**{field: better}))
    bad = matrix.offense_index(make_form(**{field: worse}))
    assert good > bad


@pytest.mark.parametrize("field,better,worse", [
    ("sack", 0.03, 0.11),        # being sacked is bad for an offence
    ("turnover", 0.008, 0.040),  # giving it away is bad for an offence
])
def test_an_offence_is_punished_for_sacks_and_giveaways(field, better, worse):
    good = matrix.offense_index(make_form(**{field: better}))
    bad = matrix.offense_index(make_form(**{field: worse}))
    assert good > bad


def test_allowing_less_is_better_defence():
    """`def_*` records what the OPPONENT did, so low is good for the first three."""
    stingy = matrix.defense_index(make_form(allowed_epa=-0.12, allowed_first_down=0.25,
                                            allowed_explosive=0.03))
    leaky = matrix.defense_index(make_form(allowed_epa=0.12, allowed_first_down=0.34,
                                           allowed_explosive=0.09))
    assert stingy > leaky


def test_generating_more_sacks_and_takeaways_is_better_defence():
    disruptive = matrix.defense_index(make_form(allowed_sack=0.11, allowed_turnover=0.040))
    passive = matrix.defense_index(make_form(allowed_sack=0.03, allowed_turnover=0.008))
    assert disruptive > passive


def test_every_prediction_coefficient_has_the_sign_football_says_it_should():
    c = matrix.COEFFICIENTS
    assert c["epa"] > 0
    assert c["first_down"] > 0
    assert c["explosive"] > 0
    assert c["sack"] < 0
    assert c["turnover"] < 0
    assert c["home_field"] > 0


# ── reconciliation ───────────────────────────────────────────────────────────
def test_offence_plus_defence_is_the_efficiency_rating():
    form = make_form(epa=0.11, allowed_epa=-0.04)
    assert (matrix.offense_index(form) + matrix.defense_index(form)
            == pytest.approx(matrix.efficiency_rating(form)))


def test_margin_is_the_difference_of_the_two_efficiency_ratings_plus_home_field():
    """The intercept and the league-mean terms must cancel out of a margin.

    This is what lets the dashboard show a unit ranking beside a game projection
    without the two quietly disagreeing.
    """
    home = make_form(epa=0.12, allowed_epa=-0.06)
    away = make_form(epa=-0.03, allowed_epa=0.04)
    expected = (matrix.efficiency_rating(home) - matrix.efficiency_rating(away)
                + matrix.COEFFICIENTS["home_field"])
    assert matrix.margin_points(home, away) == pytest.approx(expected)


def test_a_neutral_site_removes_the_home_field_term():
    home = make_form(epa=0.12)
    away = make_form(epa=-0.03)
    hosted = matrix.margin_points(home, away)
    neutral = matrix.margin_points(home, away, neutral=True)
    assert hosted - neutral == pytest.approx(matrix.COEFFICIENTS["home_field"])


def test_two_identical_teams_project_to_home_field_exactly():
    form = make_form(epa=0.05)
    assert matrix.margin_points(form, form) == pytest.approx(
        matrix.COEFFICIENTS["home_field"])


# ── the published breakdown ──────────────────────────────────────────────────
def test_the_breakdown_reconciles_with_the_margin_exactly():
    """A breakdown whose parts do not add up to the number above it is worse
    than none at all: it invites the reader to trust a wrong accounting."""
    home = make_form(epa=0.12, first_down=0.33, allowed_epa=-0.06)
    away = make_form(epa=-0.03, first_down=0.26, allowed_epa=0.05)
    contributions = matrix.margin_contributions(home, away)
    assert set(contributions) == set(matrix.STATS)
    total = sum(contributions.values()) + matrix.COEFFICIENTS["home_field"]
    assert total == pytest.approx(matrix.margin_points(home, away), abs=1e-9)


def test_the_breakdown_is_antisymmetric_in_the_two_teams():
    home = make_form(epa=0.12, allowed_epa=-0.06)
    away = make_form(epa=-0.03, allowed_epa=0.05)
    forward = matrix.margin_contributions(home, away)
    reverse = matrix.margin_contributions(away, home)
    for stat in matrix.STATS:
        assert forward[stat] == pytest.approx(-reverse[stat], abs=1e-12)


def test_a_contribution_follows_the_net_that_drives_it():
    """Each contribution is the coefficient times the difference of the two
    teams' nets, which is why the card shows nets rather than raw offence."""
    home = make_form(epa=0.12, allowed_epa=-0.06)
    away = make_form(epa=-0.03, allowed_epa=0.05)
    net_home = home.off_epa - home.def_epa
    net_away = away.off_epa - away.def_epa
    expected = matrix.COEFFICIENTS["epa"] * (net_home - net_away)
    assert matrix.margin_contributions(home, away)["epa"] == pytest.approx(expected)


def test_an_incomplete_form_yields_no_breakdown():
    assert matrix.margin_contributions(matrix.TeamForm(off_epa=0.1), average_form()) is None
