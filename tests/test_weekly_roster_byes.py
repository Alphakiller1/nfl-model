from nflmodel.sources import nflverse


def _rows():
    # Week 6: KC on bye, so its latest roster is week 5.
    return [
        {"season": "2026", "week": "5", "team": "KC", "full_name": "Patrick Mahomes"},
        {"season": "2026", "week": "5", "team": "BUF", "full_name": "Old Buffalo QB"},
        {"season": "2026", "week": "6", "team": "BUF", "full_name": "Josh Allen"},
    ]


def test_a_club_off_a_bye_keeps_its_latest_roster(monkeypatch):
    """2026-10-19: the week-7 board built from the week-6 roster lost every
    club that was on bye in week 6 (24/28 teams playing)."""
    monkeypatch.setattr(nflverse, "fetch_csv", lambda *a, **k: _rows())
    names = {r["full_name"] for r in nflverse.weekly_roster(2026, week=7)}
    assert names == {"Patrick Mahomes", "Josh Allen"}


def test_weeks_after_the_board_are_ignored(monkeypatch):
    monkeypatch.setattr(nflverse, "fetch_csv", lambda *a, **k: _rows())
    names = {r["full_name"] for r in nflverse.weekly_roster(2026, week=5)}
    assert names == {"Patrick Mahomes", "Old Buffalo QB"}
