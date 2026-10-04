from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from math import exp, factorial
from types import SimpleNamespace

import pytest

from nflmodel import prop_matchup_paths, prop_report_view, prop_scouting, teams, weekly_props
from nflmodel.player_props import PlayerProjection
from nflmodel.sources import chase_context, espn_props
from nflmodel.sources.oddsapi import PlayerPropQuote

NOW = datetime(2026, 10, 4, 8, tzinfo=UTC)
KICKOFF = NOW + timedelta(days=1)


def player(position="RB", identity="p1", team="NE", opponent="SEA", **changes):
    metrics = {
        "QB": {
            "passing_yards": 240.0,
            "pass_attempts": 34.0,
            "completions": 22.0,
            "passing_tds": 1.5,
            "rush_attempts": 4.0,
            "rushing_yards": 20.0,
        },
        "RB": {
            "rushing_yards": 70.0,
            "carries": 16.0,
            "receiving_yards": 22.0,
            "targets": 4.0,
            "receptions": 3.0,
        },
        "WR": {"receiving_yards": 65.0, "targets": 8.0, "receptions": 5.0},
        "K": {"fg_attempts": 2.4, "fg_made": 2.0, "pat_made": 2.5, "kicking_points": 8.5},
    }[position]
    row = PlayerProjection(
        season=2026,
        week=4,
        game_id=f"2026_04_{opponent}_{team}",
        kickoff="Mon 1 PM ET",
        kickoff_utc=KICKOFF.isoformat(),
        team=team,
        opponent=opponent,
        home=True,
        player_id=identity,
        player_name=f"Player {identity}",
        position=position,
        depth_rank=1,
        depth_slot=f"{position}1",
        roster_status="ACT",
        injury_status=None,
        headshot_url="",
        history_games=20,
        last_team=team,
        role_continuity="same team",
        persistence_weight=0.72,
        role_reason="Established current starter",
        confidence="high",
        team_environment_source="independent model + scheme",
        implied_team_points=24.0,
        metrics=metrics,
        scheme_context={
            "pass_attempt_delta": 0.4,
            "carry_delta": 0.5,
            "pass_efficiency_delta": 0.2,
            "rush_efficiency_delta": 0.1,
        },
    )
    return replace(row, **changes)


def quote(row, market=None, line=None, **changes):
    market = (
        market
        or {
            "RB": "player_rush_yds",
            "QB": "player_pass_yds",
            "WR": "player_reception_yds",
            "K": "player_field_goals",
        }[row.position]
    )
    line = (
        line if line is not None else {"RB": 59.5, "QB": 209.5, "WR": 49.5, "K": 1.5}[row.position]
    )
    out = PlayerPropQuote(
        "event",
        row.team,
        row.opponent,
        market,
        row.player_name,
        line,
        -110,
        -110,
        "draftkings",
        "DraftKings",
        NOW.isoformat(),
        player_id=row.player_id,
    )
    return replace(out, **changes)


def slate(players, quotes=(), context=None):
    return SimpleNamespace(
        season=2026,
        week=4,
        player_projections=players,
        player_prop_quotes=list(quotes),
        assembled_at_utc=NOW.isoformat(),
        prop_context=context or {},
        scheme_profiles={},
        scheme_matchups={},
        injuries=[],
        games=[],
        weekly_prop_report={},
    )


def test_four_top_tens_unique_players_and_consistent_sort():
    clubs = list(teams.all_abbrs())[:24]
    players = [
        player(pos, f"{club}-{pos}", club, clubs[i ^ 1])
        for i, club in enumerate(clubs)
        for pos in weekly_props.GROUPS
    ]
    quotes = [quote(p) for p in players]
    # Duplicate provider records and multiple markets must not duplicate people.
    quotes += [quotes[0], quote(players[0], "player_receptions", 1.5)]
    report = weekly_props.build(slate(players, quotes), now=NOW)
    for group in report["groups"].values():
        assert group["published"] == group["quoted"] == 10
        rows = group["rows"]
        assert len({r["player_id"] for r in rows}) == 10
        assert rows == sorted(rows, key=weekly_props._sort)
        assert [r["rank"] for r in rows] == list(range(1, 11))
        assert all(r["conservative_hit_probability"] <= r["hit_probability"] for r in rows)
        assert all(not r["may_bet"] for r in rows)


def test_unpriced_quotes_never_publish_nominal_prices():
    p = player()
    row = weekly_props.build(slate([p], [quote(p, priced=False)]), now=NOW)["groups"]["RB"]["rows"][
        0
    ]
    assert row["line"] == 59.5 and row["price"] is None and not row["priced"]


def test_missing_lines_are_research_milestones_and_shortfalls_explicit():
    report = weekly_props.build(slate([player()]), now=NOW)
    rb = report["groups"]["RB"]
    assert rb["published"] == 1 and rb["shortfall"] == 9
    row = rb["rows"][0]
    assert row["line"] is None and row["book"] is None and row["price"] is None
    assert row["threshold_source"] == "research_milestone" and not row["calibrated"]
    assert report["groups"]["K"]["published"] == 0


@pytest.mark.parametrize("designation", ["Out", "Doubtful", "Inactive", "Injured Reserve"])
def test_unavailable_players_never_fill_rankings(designation):
    p = player(injury_status=designation)
    report = weekly_props.build(slate([p], [quote(p)]), now=NOW)
    assert report["groups"]["RB"]["published"] == 0
    assert report["excluded"]["unavailable_player"] == 1


def test_started_future_wrong_fixture_and_wrong_identity_quotes_are_excluded():
    p, started = (
        player(),
        player(identity="started", kickoff_utc=(NOW - timedelta(hours=1)).isoformat()),
    )
    bad = [
        quote(p, player_id="wrong"),
        quote(p, home_team="BUF"),
        quote(p, last_update=(NOW + timedelta(hours=1)).isoformat()),
        quote(p, last_update=(NOW - timedelta(days=4)).isoformat()),
    ]
    report = weekly_props.build(slate([p, started], bad), now=NOW)
    assert report["groups"]["RB"]["quoted"] == 0
    assert report["excluded"]["started_or_undated_game"] == 1
    assert report["excluded"]["stale_or_future_quote"] == 2


def test_malformed_projection_and_unknown_kickoff_do_not_crash_or_publish():
    p = player(metrics={"rushing_yards": float("nan")})
    report = weekly_props.build(slate([p, player(identity="undated", kickoff_utc="")]), now=NOW)
    assert report["groups"]["RB"]["published"] == 0


def test_qb_rushing_requires_separate_scheme_evidence():
    p = player("QB")
    report = weekly_props.build(slate([p], [quote(p, "player_rush_yds", 19.5)]), now=NOW)
    assert all(
        r["metric"] not in {"rushing_yards", "rush_attempts"}
        for r in report["groups"]["QB"]["rows"]
    )


@pytest.mark.parametrize(
    "metric,mean", [("fg_made", 2.0), ("pat_made", 2.5), ("receptions", 5.0), ("passing_tds", 1.5)]
)
def test_discrete_push_is_separate_and_probabilities_conserve_mass(metric, mean):
    out = weekly_props.probability(metric, {metric: mean}, 2, quoted=True)
    assert out["push"] > 0
    assert out["over"] + out["under"] + out["push"] == pytest.approx(1)
    if metric in {"fg_made", "pat_made"}:
        assert not out["calibrated"]


def test_combined_probabilities_bound_independent_and_comonotone_outcomes():
    # Count marginals allow exact comparison to two distinct dependence models.
    means = {"fg_made": 2.0, "pat_made": 2.5}
    out = weekly_props.probability("kicking_points", means, 7.5, quoted=True)
    independent = sum(
        exp(-4.5) * 2**a * 2.5**b / (factorial(a) * factorial(b))
        for a in range(25)
        for b in range(25)
        if 3 * a + b > 7.5
    )
    low, high = out["over_interval"]
    assert low <= independent <= high

    def quantile(mean, u):
        cumulative = 0.0
        for count in range(40):
            cumulative += exp(-mean) * mean**count / factorial(count)
            if cumulative >= u:
                return count
        raise AssertionError("Unaccounted Poisson tail")

    # Same uniform quantile in both components gives maximal positive rank dependence.
    dependent = (
        sum(
            3 * quantile(2.0, (i + 0.5) / 4096) + quantile(2.5, (i + 0.5) / 4096) > 7.5
            for i in range(4096)
        )
        / 4096
    )
    assert low <= dependent <= high
    assert not out["calibrated"] and out["basis"] == "dependence_bound"
    assert weekly_props.probability("kicking_points", means, 7) is None


def test_scouting_zero_rates_and_prior_season_dates_are_retained():
    p = player("WR")
    game = {
        "home": "NE",
        "away": "SEA",
        "kickoff_utc": KICKOFF.isoformat(),
        "home_scheme_current": {
            "source_seasons": [2026],
            "offense_plays": 180,
            "offense": {"personnel": {"motion_rate": 0.0}},
        },
        "away_scheme": {
            "source_seasons": [2025],
            "participation_source_seasons": [2025],
            "defense_plays": 900,
            "defense": {"coverage": {"cover_4_rate": 0.3}},
        },
    }
    d = prop_scouting.dossier(slate([p], context={"games": [game]}), p)
    concepts = next(s for s in d["sections"] if s["family"] == "concepts")
    coverage = next(s for s in d["sections"] if s["family"] == "opponent_coverage")
    assert concepts["values"]["motion_rate"] == 0.0
    assert coverage["provenance"]["cover_4_rate"]["source_seasons"] == [2025]
    assert "zone/gap/power" in " ".join(prop_scouting.MISSING_ASSIGNMENTS)


def test_coverage_sample_shrinkage_and_reception_specific_response():
    splits = {
        "all": {"yards_per_target": 8, "catch_rate": 0.7},
        "man": {"targets": 3, "yards_per_target": 20, "catch_rate": 0.2, "source_season": 2025},
    }
    yardage = prop_scouting._coverage_response(splits, {"man_rate": 0.5}, "WR")
    catches = prop_scouting._coverage_response(splits, {"man_rate": 0.5}, "WR", "catch_rate")
    assert 0 < yardage["response_delta"] < 1
    assert catches["response_delta"] < 0
    assert catches["covered_mass"] == 0.5


def test_prior_look_is_not_compared_with_current_season_baseline():
    splits = {
        "all": {"yards_per_attempt": 7, "source_season": 2026},
        "man": {"attempts": 80, "yards_per_attempt": 11, "source_season": 2025},
    }
    response = prop_scouting._coverage_response(splits, {"man_rate": 0.6}, "QB")
    assert response["status"] == "unavailable"
    assert response["matchup_response"] == 7 and response["covered_mass"] == 0


def test_pressure_and_blitz_are_separate_sampled_axes_with_uncovered_baseline():
    splits = {
        "all": {"yards_per_attempt": 8, "source_season": 2025},
        "pressure": {"attempts": 48, "yards_per_attempt": 4, "source_season": 2025},
        "blitz": {"attempts": 48, "yards_per_attempt": 12, "source_season": 2025},
    }
    out = prop_scouting._conditional_responses(
        splits, {"pressure_rate": 0.4, "blitz_rate": 0.2}, "QB", "yards_per_attempt"
    )
    assert out["pressure"]["matchup_response"] == pytest.approx(7.2)
    assert out["pressure"]["covered_mass"] == 0.4
    assert out["blitz"]["matchup_response"] == pytest.approx(8.4)
    assert out["blitz"]["covered_mass"] == 0.2


def test_rb_box_response_uses_carry_samples_without_pass_pressure_axes():
    splits = {
        "all": {"yards_per_carry": 4, "source_season": 2025},
        "stacked_box": {"carries": 24, "yards_per_carry": 2, "source_season": 2025},
    }
    out = prop_scouting._conditional_responses(
        splits, {"stacked_box_rate": 0.3}, "RB", "yards_per_carry"
    )
    assert out["box_count"]["matchup_response"] == pytest.approx(3.7)
    assert "pressure" not in out


def test_tracking_retains_dates_samples_and_excludes_forecast_week():
    p = player("QB")
    profiles = [
        {
            "player_id": p.player_id,
            "source_season": 2025,
            "tracking": {
                "season": 2025,
                "week": 18,
                "attempts": 500,
                "avg_time_to_throw": 2.8,
                "pressure_dropbacks": 600,
                "pressure_rate": 0.4,
                "source": "NGS / PFR",
            },
        },
        {
            "player_id": p.player_id,
            "source_season": 2026,
            "tracking": {"season": 2026, "week": 4, "attempts": 130, "avg_time_to_throw": 3.2},
        },
    ]
    tracked = prop_scouting._tracking({"home": "NE", "home_player_scheme": profiles}, p)
    assert tracked["values"]["avg_time_to_throw"] == 2.8
    assert tracked["provenance"]["avg_time_to_throw"]["through_week"] == 18
    assert tracked["provenance"]["pressure_rate"]["sample"] == 600
    assert tracked["provenance"]["avg_time_to_throw"]["sample"] == 500


def test_reception_evidence_preserves_selected_metric_and_does_not_mutate_dossier():
    p = player("WR")
    d = prop_scouting.dossier(slate([p]), p)
    section = next(s for s in d["sections"] if s["family"] == "individual_look_response")
    section["passing_or_receiving"] = {
        "all": {"catch_rate": 0.7, "yards_per_target": 8, "source_season": 2025},
        "man": {"targets": 24, "catch_rate": 0.3, "yards_per_target": 20, "source_season": 2025},
    }
    d["defense"] = {"man_rate": 0.5}
    original = deepcopy(d)
    angle = prop_scouting.explain(p, "receptions", 3.5, "OVER", d)
    evidence = next(s for s in angle["evidence"] if s["family"] == "individual_look_response")
    assert evidence["selected_market_responses"]["coverage"]["metric"] == "catch_rate"
    assert evidence["selected_market_responses"]["coverage"]["response_delta"] < 0
    assert d == original


def test_run_point_proxy_uses_same_season_usage_and_shrinks_small_lanes():
    rushing = {
        "all": {"carries": 50, "source_season": 2026},
        "gap_guard": {"carries": 25, "source_season": 2026},
        "gap_end": {"carries": 100, "source_season": 2025},
    }
    lanes = {
        "gap_guard": {"carries": 6, "yards_per_carry_allowed": 8},
        "gap_end": {"carries": 30, "yards_per_carry_allowed": 10},
    }
    out = prop_matchup_paths.run_points(rushing, lanes, {"yards_per_carry": 4})
    assert out["matched_run_point_yards_per_carry"] == pytest.approx(4.4)
    assert out["covered_mass"] == 0.5 and out["status"] == "proxy"


def test_td_conversion_points_in_opposite_directions_for_fg_and_pats():
    path = {
        "status": "proxy",
        "matchup": {"td_rate": 0.8, "score_rate": 0.9, "trips_per_game": 3.5},
        "league": {"td_rate": 0.6, "score_rate": 0.85, "trips_per_game": 3.5},
    }
    fg = prop_matchup_paths.kicking_signals("fg_made", True, path)
    pat = prop_matchup_paths.kicking_signals("pat_made", True, path)
    assert not fg[0] and fg[1]
    assert pat[0] and not pat[1]


def snapshot():
    return {
        "schema": "chase-public-slate/1",
        "sport": "nfl",
        "generated_at_utc": (NOW - timedelta(minutes=10)).isoformat(),
        "games": [
            {
                "home": "NE",
                "away": "SEA",
                "kickoff_utc": KICKOFF.isoformat(),
                "scheme_source": {
                    "season": 2026,
                    "week": 4,
                    "observed_at_utc": (NOW - timedelta(hours=1)).isoformat(),
                },
            }
        ],
    }


def test_source_snapshot_requires_matching_dated_pre_forecast_evidence():
    good = snapshot()
    assert chase_context.validate(good, season=2026, week=4, now=NOW)["state"] == "fresh"
    for bad in [
        dict(good, generated_at_utc=(NOW + timedelta(hours=1)).isoformat()),
        dict(good, generated_at_utc=(NOW - timedelta(days=4)).isoformat()),
        dict(good, sport="mlb"),
        dict(good, invalid=float("nan")),
    ]:
        assert not chase_context.validate(bad, season=2026, week=4, now=NOW)["games"]
    assert not chase_context.validate(good, season=2026, week=3, now=NOW)["games"]
    bad = deepcopy(good)
    bad["games"][0]["scheme_source"]["observed_at_utc"] = KICKOFF.isoformat()
    assert not chase_context.validate(bad, season=2026, week=4, now=NOW)["games"]


@pytest.mark.parametrize(
    "key,value", [("pressure_rate", 1.1), ("carry_share", -0.1), ("targets", -1)]
)
def test_impossible_observed_measures_are_rejected(key, value):
    bad = snapshot()
    bad["games"][0]["home_scheme_current"] = {"offense": {"response": {key: value}}}
    assert not chase_context.validate(bad, season=2026, week=4, now=NOW)["games"]


def test_fresh_wrapper_cannot_hide_stale_scouting_evidence():
    bad = snapshot()
    bad["games"][0]["scheme_source"]["observed_at_utc"] = (NOW - timedelta(days=4)).isoformat()
    assert not chase_context.validate(bad, season=2026, week=4, now=NOW)["games"]


def test_half_catch_threshold_requires_one_actual_catch():
    p = player()
    d = prop_scouting.dossier(slate([p]), p)
    angle = prop_scouting.explain(p, "receptions", 0.5, "OVER", d)
    assert angle["threshold_requirements"]["catch_rate_at_projected_volume"] == 0.25
    assert "at least 1 catch" in angle["thesis"]


def test_espn_combined_and_kicking_markets_parse_verified_names():
    names = [
        "Total Passing Attempts (incl. overtime)",
        "Total Rushing Plus Receiving Yards (incl. overtime)",
        "Total Passing Plus Rushing Yards (incl. overtime)",
        "Total Field Goals Made (incl. overtime)",
        "Total Kicking Points (incl. overtime)",
        "Total Extra Points Made (incl. overtime)",
    ]
    items = [
        {
            "type": {"name": name},
            "athlete": {"$ref": "https://espn.com/athletes/123"},
            "current": {"target": {"value": 1.5}},
        }
        for name in names
    ]
    quotes = espn_props.parse(
        {"items": items}, event_id="e", home="NE", away="SEA", players={"123": ("player", "Player")}
    )
    assert len(quotes) == 6
    assert all(not q.priced for q in quotes)


def test_html_has_four_groups_and_escapes_external_content():
    p = player(player_name='<script>alert("x")</script>')
    report = weekly_props.build(slate([p], [quote(p)]), now=NOW)
    rendered = prop_report_view.render(report)
    assert rendered.count('class="wp-group"') == 4
    assert "<script>alert" not in rendered and "&lt;script&gt;" in rendered
    assert "Complete evidence dossier" in rendered and "weekly-props.json" in rendered


def test_shared_report_is_identical_for_consumers_and_does_not_mutate_projection():
    p = player()
    before = deepcopy(p.metrics)
    s = slate([p], [quote(p)])
    first = weekly_props.for_slate(s)
    assert weekly_props.for_slate(s) is first
    assert p.metrics == before
