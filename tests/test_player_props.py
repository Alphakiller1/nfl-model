from types import SimpleNamespace

import pytest

from nflmodel import export, player_props
from nflmodel.sources import nflverse


def _roster(player_id, name, position, team="NE", status="ACT"):
    return {
        "season": "2026", "week": "1", "team": team, "position": position,
        "status": status, "full_name": name, "gsis_id": player_id,
        "headshot_url": "",
    }


def _depth(player_id, name, position, rank, team="NE"):
    return {
        "dt": "2026-09-02T12:00:00Z", "team": team, "player_name": name,
        "gsis_id": player_id, "pos_abb": "PK" if position == "K" else position,
        "pos_slot": f"{position}{rank}", "pos_rank": str(rank),
    }


def _history(player_id, name, position, team, week, **stats):
    row = {
        "player_id": player_id, "player_display_name": name, "position": position,
        "team": team, "opponent_team": "BUF", "season": "2025", "week": str(week),
        "season_type": "REG", "attempts": "0", "completions": "0",
        "passing_yards": "0", "passing_tds": "0", "passing_interceptions": "0",
        "carries": "0", "rushing_yards": "0", "rushing_tds": "0",
        "targets": "0", "receptions": "0", "receiving_yards": "0",
        "receiving_tds": "0", "fg_att": "0", "fg_made": "0", "pat_made": "0",
    }
    row.update({key: str(value) for key, value in stats.items()})
    return row


@pytest.fixture
def player_build():
    roster = [
        _roster("qb", "Current QB", "QB"),
        _roster("rb", "Current RB", "RB"),
        _roster("wr1", "Returning WR", "WR"),
        _roster("wr2", "Transferred WR", "WR"),
        _roster("rookie", "Rookie WR", "WR"),
        _roster("out", "Unavailable WR", "WR"),
        _roster("te", "Current TE", "TE"),
        _roster("k", "Current K", "K"),
        _roster("cut", "Cut WR", "WR", status="CUT"),
    ]
    depth = [
        _depth("qb", "Current QB", "QB", 1),
        _depth("rb", "Current RB", "RB", 1),
        _depth("wr1", "Returning WR", "WR", 1),
        _depth("wr2", "Transferred WR", "WR", 2),
        _depth("rookie", "Rookie WR", "WR", 3),
        _depth("out", "Unavailable WR", "WR", 4),
        _depth("te", "Current TE", "TE", 1),
        _depth("k", "Current K", "K", 1),
        _depth("cut", "Cut WR", "WR", 4),
    ]
    history = []
    for week in range(1, 18):
        history.extend([
            _history("qb", "Current QB", "QB", "NE", week, attempts=34,
                     completions=22, passing_yards=245, passing_tds=1.6,
                     passing_interceptions=0.8, carries=4, rushing_yards=20),
            _history("rb", "Current RB", "RB", "NE", week, carries=15,
                     rushing_yards=66, rushing_tds=0.4, targets=4, receptions=3,
                     receiving_yards=22, receiving_tds=0.1),
            _history("wr1", "Returning WR", "WR", "NE", week, targets=9,
                     receptions=6, receiving_yards=78, receiving_tds=0.4),
            # Extreme old usage must not be treated as an unchanged NE role.
            _history("wr2", "Transferred WR", "WR", "PHI", week, targets=15,
                     receptions=10, receiving_yards=140, receiving_tds=0.8),
            _history("te", "Current TE", "TE", "NE", week, targets=6,
                     receptions=4, receiving_yards=45, receiving_tds=0.3),
            _history("k", "Current K", "K", "NE", week, fg_att=2,
                     fg_made=1.7, pat_made=2.2),
        ])
    game = SimpleNamespace(
        home="NE", away="SEA", kickoff="Sun Sep 13", kickoff_utc="2026-09-13T17:00:00Z",
        book_total=45.0, book_margin=3.0, model_margin=2.0,
        projected_home_score=24.0, projected_away_score=21.0,
    )
    scheme_matchup = SimpleNamespace(
        model_version="test-scheme/1", source_seasons=(2025,), confidence="medium",
        expected_man_rate=0.4, expected_zone_rate=0.6,
        expected_motion_rate=0.5, expected_play_action_rate=0.25,
        expected_blitz_rate=0.3, expected_pressure_rate=0.35,
        target_multipliers={"RB": 1.08, "WR": 0.96, "TE": 1.03},
        pass_attempt_delta=0.8, carry_delta=-0.4,
        pass_efficiency_delta=0.1, rush_efficiency_delta=-0.05,
        factors=("test point-in-time coverage response",),
    )
    return player_props.project(
        season=2026,
        week=1,
        games=[{"game_id": "2026_01_SEA_NE", "home_team": "NE", "away_team": "SEA"}],
        game_projections=[game],
        roster=roster,
        depth=depth,
        injuries=[{"team": "NE", "gsis_id": "out", "report_status": "Out"}],
        history_rows=history,
        scheme_matchups={("NE", "SEA"): scheme_matchup},
    )


def test_only_active_supported_positions_and_non_out_players_are_projected(player_build):
    names = {row.player_name for row in player_build.projections}
    assert "Cut WR" not in names
    assert "Unavailable WR" not in names
    assert {row.position for row in player_build.projections} <= set(player_props.POSITIONS)


def test_current_role_changes_discount_history_instead_of_copying_it(player_build):
    by_name = {row.player_name: row for row in player_build.projections}
    returning = by_name["Returning WR"]
    transferred = by_name["Transferred WR"]
    rookie = by_name["Rookie WR"]
    assert returning.persistence_weight == pytest.approx(0.72)
    assert transferred.persistence_weight == pytest.approx(0.18)
    assert rookie.persistence_weight == 0.0
    assert transferred.metrics["targets"] < returning.metrics["targets"]
    assert "discounted" in transferred.role_reason


def test_player_opportunities_reconcile_to_one_team_pool(player_build):
    players = player_build.projections
    quarterback = next(row for row in players if row.position == "QB")
    targets = sum(row.metrics.get("targets", 0.0) for row in players)
    # Targetable attempts exclude throwaways while QB attempts include them.
    assert targets <= quarterback.metrics["pass_attempts"] * 1.04
    assert targets >= quarterback.metrics["pass_attempts"] * 0.80
    assert all(value >= 0 for row in players for value in row.metrics.values())
    assert all(row.scheme_context["model_version"] == "test-scheme/1" for row in players)
    assert all("scheme matrix" in row.team_environment_source for row in players)


def test_projection_contract_never_invents_a_player_line_or_edge(slate, player_build):
    slate.player_projections = player_build.projections
    slate.player_status = player_build.status
    payload = export.payload(slate)
    assert payload["player_projections"]
    assert payload["player_model"]["pricing"].startswith("none")
    assert all(row["sportsbook_line"] is None for row in payload["player_projections"])
    assert all(row["edge"] is None for row in payload["player_projections"])
    assert all(row["scheme_context"] for row in payload["player_projections"])


def test_latest_depth_parser_respects_the_point_in_time_cutoff():
    raw = (
        b"dt,team,player_name\n"
        b"2026-09-01T12:00:00Z,NE,First\n"
        b"2026-09-02T12:00:00Z,NE,Second\n"
        b"2026-09-03T12:00:00Z,NE,Third\n"
    )
    rows = nflverse._parse_latest(
        raw, "dt", before="2026-09-02T18:00:00Z"
    )
    assert [row["player_name"] for row in rows] == ["Second"]
