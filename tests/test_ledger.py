from dataclasses import replace
from datetime import datetime, timedelta, timezone

from nflmodel import ledger
from nflmodel.player_props import PlayerProjection
from nflmodel.sources.oddsapi import BookLine


def _priced_projection(slate, now):
    base = slate.projections[0]
    book = BookLine(
        "draftkings", "DraftKings", -3.5, 47.5, -170, 145,
        "2026-09-02T12:00:00Z", (now + timedelta(days=2)).isoformat(),
    )
    return replace(
        base,
        season=2026,
        week=1,
        kickoff_utc=(now + timedelta(days=2)).isoformat(),
        book_name=book.book_title,
        book_key=book.book,
        book_margin=book.home_margin,
        book_total=book.total,
        book_last_update=book.last_update,
        book_commence_time=book.commence_time,
        home_moneyline=book.home_moneyline,
        away_moneyline=book.away_moneyline,
    )


def test_ledger_records_before_kickoff_then_grades_without_duplicate(slate, tmp_path):
    now = datetime(2026, 9, 2, 13, tzinfo=timezone.utc)
    projection = _priced_projection(slate, now)
    path = tmp_path / "ledger.json"
    first = ledger.update(
        season=2026, projections=[projection], schedule=[], path=path, recorded_at=now
    )
    assert len(first["snapshots"]) == 1
    result = {
        "season": 2026, "week": 1, "home_team": projection.home,
        "away_team": projection.away, "home_score": 31, "away_score": 20,
    }
    second = ledger.update(
        season=2026, projections=[projection], schedule=[result], path=path,
        recorded_at=now + timedelta(days=3),
    )
    assert len(second["snapshots"]) == 1
    assert second["summary"]["games_graded"] == 1
    assert second["summary"]["ats"] == {"win": 1, "loss": 0, "push": 0}


def test_ledger_refuses_to_record_after_kickoff(slate, tmp_path):
    now = datetime(2026, 9, 2, 13, tzinfo=timezone.utc)
    projection = replace(
        _priced_projection(slate, now),
        kickoff_utc=(now - timedelta(hours=1)).isoformat(),
    )
    payload = ledger.update(
        season=2026, projections=[projection], schedule=[],
        path=tmp_path / "ledger.json", recorded_at=now,
    )
    assert payload["snapshots"] == []


def test_player_ledger_records_and_grades_the_exact_pre_kickoff_projection(
    slate, tmp_path
):
    now = datetime(2026, 9, 2, 13, tzinfo=timezone.utc)
    game = _priced_projection(slate, now)
    player = PlayerProjection(
        season=2026, week=1, game_id="2026_01_BUF_KC", kickoff="",
        kickoff_utc=(now + timedelta(days=2)).isoformat(), team="KC", opponent="BUF",
        home=True, player_id="qb-1", player_name="Test QB", position="QB",
        depth_rank=1, depth_slot="QB1", roster_status="ACT", injury_status=None,
        headshot_url="", history_games=17, last_team="KC", role_continuity="same team",
        persistence_weight=0.72, role_reason="current depth plus same-team usage",
        confidence="high", team_environment_source="DraftKings game total and spread",
        implied_team_points=26.0,
        metrics={"pass_attempts": 34.0, "passing_yards": 250.0},
    )
    path = tmp_path / "ledger.json"
    first = ledger.update(
        season=2026, projections=[game], player_projections=[player], schedule=[],
        path=path, recorded_at=now,
    )
    assert len(first["player_snapshots"]) == 1
    schedule = [{
        "season": 2026, "week": 1, "home_team": "KC", "away_team": "BUF",
        "home_score": 27, "away_score": 20,
    }]
    stats = [{
        "season": 2026, "week": 1, "team": "KC", "player_id": "qb-1",
        "attempts": 36, "passing_yards": 270,
    }]
    second = ledger.update(
        season=2026, projections=[], player_projections=[], player_results=stats,
        schedule=schedule, path=path, recorded_at=now + timedelta(days=3),
    )
    row = second["player_snapshots"][0]
    assert row["status"] == "graded"
    assert row["absolute_errors"] == {"pass_attempts": 2.0, "passing_yards": 20.0}
    assert second["summary"]["players"]["players_graded"] == 1
