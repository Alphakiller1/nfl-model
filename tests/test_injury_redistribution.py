from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from nflmodel import player_props
from nflmodel.sources import espn_injuries
from test_player_props import _depth, _history, _roster


def houston(woody_out):
    roster = [
        _roster(pid, name, position, team="HOU")
        for pid, name, position in [
            ("qb", "Starting QB", "QB"),
            ("montgomery", "David Montgomery", "RB"),
            ("marks", "Woody Marks", "RB"),
            ("demercado", "Emari Demercado", "RB"),
            ("wr", "Receiver", "WR"),
            ("te", "Tight End", "TE"),
        ]
    ]
    depth = [
        _depth(
            r["gsis_id"],
            r["full_name"],
            r["position"],
            {"marks": 2, "demercado": 3}.get(r["gsis_id"], 1),
            team="HOU",
        )
        for r in roster
    ]
    history = []
    for week in range(1, 11):
        for pid, name, position, stats in [
            (
                "qb",
                "Starting QB",
                "QB",
                {
                    "attempts": 34,
                    "completions": 22,
                    "passing_yards": 240,
                    "carries": 3,
                    "rushing_yards": 12,
                },
            ),
            (
                "montgomery",
                "David Montgomery",
                "RB",
                {"carries": 9, "rushing_yards": 36, "targets": 2, "receptions": 1},
            ),
            (
                "marks",
                "Woody Marks",
                "RB",
                {"carries": 11, "rushing_yards": 44, "targets": 2, "receptions": 1},
            ),
            (
                "demercado",
                "Emari Demercado",
                "RB",
                {"carries": 3, "rushing_yards": 12, "targets": 1, "receptions": 1},
            ),
            ("wr", "Receiver", "WR", {"targets": 20, "receptions": 14, "receiving_yards": 160}),
            ("te", "Tight End", "TE", {"targets": 7, "receptions": 5, "receiving_yards": 50}),
        ]:
            history.append(_history(pid, name, position, "HOU", week, season=2026, **stats))
    game = SimpleNamespace(
        home="HOU",
        away="IND",
        book_total=44.5,
        book_margin=10,
        kickoff="",
        kickoff_utc="2026-11-20T01:15:00Z",
    )
    injuries = (
        [{"team": "HOU", "gsis_id": "marks", "full_name": "Woody Marks", "report_status": "Out"}]
        if woody_out
        else []
    )
    result = player_props.project(
        season=2026,
        week=11,
        games=[{"game_id": "game", "home_team": "HOU", "away_team": "IND"}],
        game_projections=[game],
        roster=roster,
        depth=depth,
        injuries=injuries,
        history_rows=history,
    )
    return {p.player_id: p for p in result.projections}


def test_woody_marks_out_redistributes_carries_to_montgomery_without_creating_team_work():
    healthy, unavailable = houston(False), houston(True)
    assert "marks" not in unavailable
    before, after = healthy["montgomery"].metrics, unavailable["montgomery"].metrics
    assert after["carries"] > 1.5 * before["carries"]
    assert after["rushing_yards"] > 1.5 * before["rushing_yards"]
    for metric in ("carries", "targets"):
        assert sum(p.metrics.get(metric, 0) for p in healthy.values()) == pytest.approx(
            sum(p.metrics.get(metric, 0) for p in unavailable.values()), abs=0.05
        )
    assert after["rushing_yards"] / after["carries"] == pytest.approx(
        before["rushing_yards"] / before["carries"], abs=0.02
    )


def test_complete_league_feed_catches_a_player_missing_from_the_game_spotlight(monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 11, 19, 22, tzinfo=timezone.utc)

    payload = {
        "season": {"year": 2026},
        "injuries": [
            {
                "displayName": "Houston Texans",
                "injuries": [
                    {
                        "status": "Out",
                        "date": "2026-11-18T20:51Z",
                        "athlete": {
                            "displayName": "Woody Marks",
                            "position": {"abbreviation": "RB"},
                            "links": [
                                {"href": "https://www.espn.com/nfl/player/_/id/4429059/woody-marks"}
                            ],
                        },
                    },
                    {
                        "status": "Active",
                        "date": "2026-11-18T19:56Z",
                        "athlete": {"id": "4035538", "displayName": "David Montgomery"},
                    },
                ],
            },
            {"displayName": "Indianapolis Colts", "injuries": []},
        ],
    }

    def get(url):
        assert url == espn_injuries.LEAGUE_INJURIES
        return payload

    monkeypatch.setattr(espn_injuries, "datetime", Clock)
    monkeypatch.setattr(espn_injuries.espn_odds, "_get", get)
    roster = [
        {"espn_id": "4429059", "gsis_id": "marks"},
        {"espn_id": "4035538", "gsis_id": "montgomery"},
    ]
    rows = espn_injuries.fetch(
        {("HOU", "IND")},
        roster,
        season=2026,
        week=11,
        events=[{"kickoff_utc": "2026-11-20T01:15:00Z"}],
    )
    assert {r["gsis_id"]: r["report_status"] for r in rows} == {"marks": "Out", "montgomery": ""}
    assert espn_injuries.LAST["feed"] == "complete_league_injuries"


def test_weaker_secondary_designation_cannot_erase_an_official_out():
    official = [{"team": "HOU", "gsis_id": "marks", "report_status": "Out"}]
    weaker = [{"team": "HOU", "gsis_id": "marks", "report_status": "Questionable"}]
    merged = espn_injuries.merge(official, weaker)
    assert merged[0]["report_status"] == "Out"
    assert merged[0]["status_conflict"]
