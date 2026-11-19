import math
from types import SimpleNamespace

import pytest

from nflmodel import player_props, prop_distributions, prop_intelligence, prop_pricing
from nflmodel.sources import nflverse


def play(season=2025, week=1, play_id="1", **updates):
    row = {
        "season": str(season),
        "week": str(week),
        "season_type": "REG",
        "game_id": f"{season}_{week:02d}_BUF_NE",
        "play_id": play_id,
        "posteam": "NE",
        "defteam": "BUF",
        "pass_attempt": "1",
        "sack": "0",
        "receiver_player_id": "wr",
        "passer_player_id": "qb",
        "cp": "0.5",
        "air_yards": "10",
        "xyac_mean_yardage": "6",
        "complete_pass": "1",
        "passing_yards": "20",
        "receiving_yards": "20",
        "yardline_100": "10",
    }
    return {**row, **updates}


def test_process_uses_opportunity_expectation_and_excludes_future_games_and_duplicates():
    row = play()
    base = prop_intelligence.process_games([row], season=2025, week=2)
    actual = prop_intelligence.process_games(
        [row, row, play(week=2, receiving_yards="9999"), play(season=2026)], season=2025, week=2
    )
    assert actual == base
    receiver = next(v for k, v in actual.items() if k[2] == "wr")
    assert receiver["receive_expected_catches"] == 0.5
    assert receiver["receive_expected_yards"] == 8
    assert receiver["receive_yards"] == 20
    assert receiver["receive_endzone"] == 1


def test_unknown_completion_expectation_is_not_a_zero_probability():
    missing = prop_intelligence.process_games([play(cp="")], season=2025, week=2)
    known_zero = prop_intelligence.process_games([play(cp="0")], season=2025, week=2)
    key = (prop_intelligence.order(2025, 1), "NE", "wr")
    assert "receive_cp_opps" not in missing[key]
    assert known_zero[key]["receive_cp_opps"] == 1
    assert known_zero[key]["receive_expected_catches"] == 0


def test_current_season_participation_is_unavailable_even_in_a_retrospective_replay():
    old, current = play(), play(season=2026)
    chart = [
        {"nflverse_game_id": r["game_id"], "play_id": "1", "defense_man_zone_type": "MAN_COVERAGE"}
        for r in (old, current)
    ]
    profiles = prop_intelligence.process_games(
        [old, current], season=2026, week=2, participation_rows=chart
    )
    assert profiles[(prop_intelligence.order(2025, 1), "NE", "wr")]["receive_man_opps"] == 1
    assert "receive_man_opps" not in profiles[(prop_intelligence.order(2026, 1), "NE", "wr")]


def test_ftn_boolean_strings_retain_observed_play_design():
    row = play()
    chart = [
        {
            "nflverse_game_id": row["game_id"],
            "nflverse_play_id": "1",
            "is_play_action": "TRUE",
            "is_screen_pass": "FALSE",
            "n_blitzers": "1",
        }
    ]
    profiles = prop_intelligence.process_games([row], season=2025, week=2, charting_rows=chart)
    cell = profiles[(prop_intelligence.order(2025, 1), "NE", "wr")]
    assert cell["receive_play_action_opps"] == 1
    assert cell["receive_no_screen_opps"] == 1
    assert cell["receive_blitz_opps"] == 1


def test_current_stint_usage_cannot_inherit_a_previous_clubs_role():
    rows = [
        {
            "season": 2025,
            "week": w,
            "team": "BUF",
            "player_id": "wr",
            "position": "WR",
            "targets": 20,
        }
        for w in range(1, 6)
    ]
    rows += [
        {
            "season": 2025,
            "week": 6,
            "team": "NE",
            "player_id": "wr",
            "position": "WR",
            "targets": 2,
        },
        {
            "season": 2025,
            "week": 6,
            "team": "NE",
            "player_id": "other",
            "position": "WR",
            "targets": 18,
        },
    ]
    histories, totals = player_props._history_index(rows, season=2025, week=7)
    assert player_props._share(histories["wr"], "targets", "targets", totals, team="NE") == (
        pytest.approx(0.1)
    )
    assert player_props._usage_games(histories["wr"], "NE") == pytest.approx(0.5 ** (1 / 6))


def test_missing_air_yards_does_not_turn_a_receiver_into_a_zero_adot_player():
    base = {
        "season": 2025,
        "week": 1,
        "team": "NE",
        "player_id": "wr",
        "position": "WR",
        "targets": 20,
        "receptions": 12,
        "receiving_yards": 150,
    }
    missing, _ = player_props._history_index([base], season=2025, week=2)
    short, _ = player_props._history_index(
        [{**base, "receiving_air_yards": 0}], season=2025, week=2
    )
    assert (
        player_props._receiver_rates(missing["wr"], "WR")[1]
        > (player_props._receiver_rates(short["wr"], "WR")[1])
    )


def test_each_team_keeps_its_own_pre_kickoff_depth_snapshot():
    raw = (
        b"dt,team,player\n2026-11-19T12:00:00Z,NE,old\n"
        b"2026-11-20T12:00:00Z,NE,new\n2026-11-19T12:00:00Z,LAR,old\n"
        b"2026-11-20T12:00:00Z,LAR,new\n2026-11-23T12:00:00Z,LAR,future\n"
    )
    rows = nflverse._parse_latest(
        raw, "dt", before_by_team={"NE": "2026-11-20T01:00:00Z", "LA": "2026-11-22T17:00:00Z"}
    )
    assert {(r["team"], r["player"]) for r in rows} == {("NE", "old"), ("LAR", "new")}


def test_last_weeks_out_designation_is_not_reused_as_a_new_report(monkeypatch):
    monkeypatch.setattr(
        nflverse,
        "fetch_csv",
        lambda *a, **kw: [{"week": "10", "gsis_id": "wr", "report_status": "Out"}],
    )
    assert nflverse.injuries(2026, week=11) == []


def row(pid, position, metrics, trend=0):
    return {
        "player_id": pid,
        "position": position,
        "game_id": "game",
        "team": "NE",
        "metrics": metrics,
        "features": {"role_trend": trend},
        "evidence": {},
    }


def spec(intercept=0.0, coefficient=0.0):
    return {
        "intercept": intercept,
        "features": ["role_trend"],
        "center": [0],
        "scale": [1],
        "coefficients": [coefficient],
        "limit": 0.5,
    }


def test_learned_role_changes_redistribute_one_pool_and_reconcile_rate_components():
    players = [
        row(
            "qb",
            "QB",
            {
                "pass_attempts": 30,
                "completions": 20,
                "passing_yards": 230,
                "rush_attempts": 4,
                "rushing_yards": 20,
            },
        ),
        row("a", "WR", {"targets": 7, "receptions": 4, "receiving_yards": 60}, 1),
        row("b", "WR", {"targets": 7, "receptions": 4, "receiving_yards": 60}, -1),
        row("te", "TE", {"targets": 5, "receptions": 3, "receiving_yards": 35}),
    ]
    updated = prop_intelligence.correct_metrics(
        players,
        {
            "corrections": {
                "WR:targets": spec(coefficient=0.4),
                "WR:receiving_yards": spec(intercept=0.5),
            }
        },
    )
    assert sum(r["metrics"].get("targets", 0) for r in updated) == pytest.approx(19, abs=0.02)
    assert updated[1]["metrics"]["targets"] > updated[2]["metrics"]["targets"]
    part = updated[1]["components"]["receiving_yards"]
    assert part["final_mean"] == pytest.approx(
        part["structural_mean"] + part["opportunity_effect"] + part["rate_effect"], abs=0.003
    )


def test_qb_workload_changes_preserve_team_carry_and_target_budgets():
    players = [
        row(
            "qb",
            "QB",
            {
                "pass_attempts": 30,
                "completions": 20,
                "passing_yards": 230,
                "rush_attempts": 4,
                "rushing_yards": 20,
            },
        ),
        row(
            "rb",
            "RB",
            {
                "carries": 20,
                "rushing_yards": 80,
                "targets": 4,
                "receptions": 3,
                "receiving_yards": 25,
            },
        ),
        row("wr", "WR", {"targets": 15, "receptions": 10, "receiving_yards": 100}),
    ]
    updated = prop_intelligence.correct_metrics(
        players,
        {
            "corrections": {
                "QB:rush_attempts": spec(intercept=0.4),
                "QB:pass_attempts": spec(intercept=-0.2),
            }
        },
    )
    assert updated[0]["metrics"]["rush_attempts"] + updated[1]["metrics"]["carries"] == (
        pytest.approx(24, abs=0.02)
    )
    assert sum(r["metrics"].get("targets", 0) for r in updated) == pytest.approx(
        19 * math.exp(-0.2), abs=0.03
    )


def test_quarterback_change_features_are_censored_and_player_specific():
    players = [
        row("qb", "QB", {"pass_attempts": 30, "completions": 21, "passing_yards": 240}),
        row("wr", "WR", {"targets": 8, "receptions": 5, "receiving_yards": 60}),
    ]
    old = (prop_intelligence.order(2026, 1), "NE")
    future = (prop_intelligence.order(2026, 3), "NE")
    totals = {
        old: {"attempts": 30, "completions": 15, "passing_yards": 150},
        future: {"attempts": 30, "completions": 30, "passing_yards": 9000},
    }
    prop_intelligence.add_quarterback_features(players, totals, season=2026, week=2)
    assert players[1]["features"]["qb_completion_shift"] == pytest.approx(0.2)
    assert players[1]["features"]["qb_yards_shift"] == pytest.approx(3.0)


def test_high_volume_count_distribution_retains_its_tail_and_tie_mass():
    pmf = prop_distributions._pmf(44, 5)
    assert max(map(int, pmf)) > 59
    assert sum(pmf.values()) == pytest.approx(1, abs=3e-6)
    assert pmf["44"] > 0


def test_negative_yardage_can_finish_under_a_negative_threshold(monkeypatch):
    from nflmodel import weekly_props

    distribution = {"q": [-5, 0, 2, 5, 10], "q_levels": [0.1, 0.25, 0.5, 0.75, 0.9]}
    monkeypatch.setattr(prop_distributions, "distribution", lambda *a, **kw: distribution)
    over, under, push = weekly_props._marginal("rushing_yards", 2, -0.5, position="QB")
    assert 0 < under < 1 and 0 < over < 1
    assert over + under + push == pytest.approx(1)
    assert prop_distributions.over_probability(distribution, -10) > 0.9


def test_previous_projection_versions_betting_evidence_cannot_price_v3():
    player = SimpleNamespace(model_version="nfl-player-projections/3.0.0")
    assert prop_pricing.price("receiving_yards", 70, 50.5, player=player) is None
    assert prop_pricing.price("receiving_yards", 70, 50.5) is not None


def test_missing_fitted_artifact_cannot_silently_publish_an_unrevised_model(monkeypatch):
    monkeypatch.setattr(prop_intelligence, "parameters", lambda: {"state": "not_fitted"})
    with pytest.raises(ValueError, match="parameters are missing"):
        prop_intelligence.revise(
            [],
            history_rows=[],
            pbp_rows=[],
            participation_rows=[],
            charting_rows=[],
            season=2026,
            week=11,
            scheme_matchups={},
        )


def test_a_fitted_future_season_or_different_model_cannot_enter_a_forecast(monkeypatch):
    for fit_year, version, message in (
        (2026, "nfl-player-projections/3.0.0", "not prior"),
        (2025, "nfl-player-projections/9.0.0", "different"),
    ):
        monkeypatch.setattr(
            prop_intelligence,
            "parameters",
            lambda: {
                "state": "time_forward_research",
                "fit_season_max": fit_year,
                "model_version": version,
            },
        )
        with pytest.raises(ValueError, match=message):
            prop_intelligence.revise(
                [],
                history_rows=[],
                pbp_rows=[],
                participation_rows=[],
                charting_rows=[],
                season=2026,
                week=11,
                scheme_matchups={},
            )


def test_predictability_depends_on_outcome_variability_and_current_role():
    stable = {"current_stint_games": 12, "recent_opportunity_cv": 0.1, "warnings": []}
    passing = prop_intelligence.predictability(
        "passing_yards", 240, {"p10": 160, "p50": 235, "p90": 325}, stable
    )
    touchdown = prop_intelligence.predictability(
        "anytime_td_probability", 0.3, {"p10": 0, "p50": 0, "p90": 1}, stable
    )
    assert passing["label"] == "relatively stable"
    assert touchdown["label"] == "high variance"
    uncertain = {**stable, "warnings": ["Player availability is unresolved"]}
    assert (
        prop_intelligence.predictability(
            "passing_yards", 240, {"p10": 160, "p50": 235, "p90": 325}, uncertain
        )["label"]
        == "availability uncertain"
    )


def test_raw_line_comparisons_keep_every_quote_and_never_invent_a_price():
    player = SimpleNamespace(
        player_id="wr",
        player_name="Receiver",
        position="WR",
        team="NE",
        opponent="BUF",
        metrics={"receiving_yards": 60, "receptions": 5},
        model_version="nfl-player-projections/3.0.0",
        intelligence={},
        predictability={},
    )
    quotes = [
        SimpleNamespace(
            player_id="wr",
            player_name="Receiver",
            market=market,
            line=line,
            book_title="DraftKings via ESPN",
            over_price=-110,
            under_price=-110,
            priced=False,
            last_update="2026-11-19T21:00:00Z",
        )
        for market, line in (("player_reception_yds", 50.5), ("player_receptions", 4.5))
    ]
    comparisons = prop_intelligence.quote_comparisons(
        SimpleNamespace(player_projections=[player], player_prop_quotes=quotes)
    )
    assert len(comparisons) == 2
    assert all(r["over_price"] is None and r["under_price"] is None for r in comparisons)
    assert all(not r["market_calibrated"] and not r["priced"] for r in comparisons)
