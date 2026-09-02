from dataclasses import replace
from datetime import datetime, timedelta, timezone

from nflmodel import ledger
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
