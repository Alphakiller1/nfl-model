from dataclasses import replace
from datetime import datetime, timedelta, timezone

from nflmodel import ledger, sharp
from nflmodel.sources import dk_splits

PAGE = """
<div class="tb-se-title"><span>BUF Bills @ </span><span>KC Chiefs</span><span>10/4</span>
<div>Spread</div><div>KC Chiefs -2.5</div><a>-110</a><div>35%</div><div>70%</div>
<div>BUF Bills +2.5</div><a>-110</a><div>65%</div><div>30%</div>
<div>Total</div><div>Over 47.5</div><a>-110</a><div>52%</div><div>48%</div>
<div>Under 47.5</div><a>-110</a><div>48%</div><div>52%</div></div>
"""
KICKOFF = datetime(2026, 10, 4, 17, tzinfo=timezone.utc)


def _game(slate):
    return replace(slate.projections[0], season=2026, week=5, home="KC", away="BUF",
                   kickoff_utc=KICKOFF.isoformat(), book_margin=2.5, book_total=47.5,
                   model_margin=1.0, projected_total=46.0)


def test_dk_names_resolve_to_team_codes():
    assert sharp.resolve("IND Colts") == "IND"
    assert sharp.resolve("WAS Commanders") == "WAS"
    assert sharp.resolve("Nowhere FC") is None


def test_money_on_the_dog_with_reverse_line_movement(slate):
    games = dk_splits.parse(PAGE)
    opening = [{"season": 2026, "week": 5, "home": "KC", "away": "BUF",
                "recorded_at": "2026-09-29T12:00:00Z", "book_margin": 3.5, "book_total": 47.5}]
    spots = sharp.build([_game(slate)], games, opening, season=2026, week=5)
    assert len(spots) == 1                                   # total split is too even
    spot = spots[0]
    assert spot.selection == "BUF +2.5" and spot.side == "away"
    assert spot.move == 1.0 and spot.reverse is True        # +3.5 -> +2.5, 30% of bets
    assert spot.model_agrees is True                         # model KC by 1 < 2.5
    assert spot.concentration == 35.0 + 3.0 + 5.0


def test_spots_are_graded_with_closing_line_value(slate, tmp_path):
    games = dk_splits.parse(PAGE)
    spot = sharp.build([_game(slate)], games, [], season=2026, week=5)[0]
    path = tmp_path / "ledger.json"
    ledger.update(season=2026, projections=[], schedule=[], sharp_spots=[spot.to_json()],
                  path=path, recorded_at=KICKOFF - timedelta(days=1))
    payload = ledger._load(path)
    payload["snapshots"].append({"season": 2026, "week": 5, "home": "KC", "away": "BUF",
                                 "recorded_at": "2026-10-04T16:00:00Z", "book_margin": 1.5,
                                 "book_total": 47.5, "status": "pending"})
    ledger._write(path, payload)
    result = [{"season": 2026, "week": 5, "home_team": "KC", "away_team": "BUF",
               "home_score": 24, "away_score": 23}]
    payload = ledger.update(season=2026, projections=[], schedule=result, path=path,
                            recorded_at=KICKOFF + timedelta(days=1))
    graded = payload["sharp_spots"][0]
    assert graded["result"] == "win" and graded["clv"] == 1.0   # took +2.5, closed +1.5
    assert payload["summary"]["sharp_spots"]["spread"]["mean_clv"] == 1.0
