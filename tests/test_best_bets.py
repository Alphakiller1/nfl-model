from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from nflmodel import best_bets as bb
from nflmodel import ledger
from nflmodel.player_props import PlayerProjection
from nflmodel.sources.oddsapi import PlayerPropQuote

NOW = datetime(2026, 9, 2, 13, tzinfo=timezone.utc)
KICKOFF = NOW + timedelta(days=2)


def _game(slate, *, model=6.0, book=2.5, total=51.0, book_total=46.5, qb_out=()):
    base = slate.projections[0]
    return replace(
        base, season=2026, week=1, kickoff_utc=KICKOFF.isoformat(),
        model_margin=model, book_margin=book, book_total=book_total, projected_total=total,
        rating_margin=5.0, efficiency_margin=7.0, qb_out=qb_out,
        availability_margin=-4.185 if qb_out else 0.0, book_name="DraftKings",
        projected_home_score=28.5, projected_away_score=22.5,
    )


INJURIES = [
    {"week": 1, "team": "BUF", "position": "WR", "full_name": "Deep Threat",
     "report_status": "Out"},
    {"week": 1, "team": "BUF", "position": "CB", "full_name": "Top Corner",
     "report_status": "Doubtful"},
    {"week": 2, "team": "BUF", "position": "QB", "full_name": "Next Week", "report_status": "Out"},
    {"week": 1, "team": "BUF", "position": "TE", "full_name": "Sore Hamstring",
     "report_status": "", "practice_status": "Did Not Participate In Practice",
     "practice_primary_injury": "Hamstring"},
    {"week": 1, "team": "BUF", "position": "RB", "full_name": "Rest Day",
     "report_status": "", "practice_status": "Did Not Participate In Practice",
     "practice_primary_injury": "Not injury related - resting player"},
]


def test_spread_pick_names_paths_injuries_and_qb_adjustment(slate, monkeypatch):
    _skilled(monkeypatch)
    game = _game(slate, qb_out=("BUF QB Backup Starter",))
    away = game.away
    injuries = bb.injury_index([dict(r, team=away) for r in INJURIES], 1)
    pick = bb.spread_pick(game, slate, injuries)
    assert pick.side == "home" and pick.edge == 3.5
    assert "3.5-point disagreement" in pick.angle
    assert f"Power ratings make it {game.home} by 5.0" in pick.angle
    assert f"efficiency makes it {game.home} by 7.0" in pick.angle
    assert "fitted 4.2-point adjustment" in pick.angle
    assert f"{away} has 2 ruled out or doubtful: WR Deep Threat (Out)" in pick.angle
    assert "Next Week" not in pick.angle          # other weeks' report rows ignored
    assert "TE Sore Hamstring missed practice with an injury" in pick.angle
    assert "Rest Day" not in pick.angle
    assert "qb-out" in pick.tags and 0.5 < pick.probability < 0.7


def test_small_gaps_are_not_picks(slate):
    assert bb.spread_pick(_game(slate, model=3.0, book=2.5), slate, {}) is None
    assert bb.total_pick(_game(slate, total=48.0, book_total=46.5), slate, {}) is None


def test_total_pick(slate, monkeypatch):
    _skilled(monkeypatch)
    pick = bb.total_pick(_game(slate), slate, {})
    assert pick.selection == "Over 46.5" and pick.edge == 4.5
    assert "Scoreline:" in pick.angle and "not inputs to the total" in pick.angle


def test_a_total_pick_needs_the_model_total_to_earn_its_weight(slate):
    # Measured weight 0.02: even a 9-point disagreement prices near even.
    assert bb.total_pick(_game(slate, total=47.8, book_total=38.5), slate, {}) is None


def _skilled(monkeypatch):
    """Give spreads and totals full skill, to test pick mechanics (not live pricing)."""
    from nflmodel import calibration

    for market in ("spread", "total"):
        old = calibration.SKILL[market]
        monkeypatch.setitem(calibration.SKILL, market,
                            calibration.MarketSkill(1.0, old.sigma, "test"))


def test_no_spread_or_total_pick_publishes_at_measured_skill(slate):
    assert bb.spread_pick(_game(slate, model=12.0, book=2.5), slate, {}) is None
    assert bb.total_pick(_game(slate, total=47.8, book_total=38.5), slate, {}) is None


def _player(**changes):
    values = dict(
        season=2026, week=1, game_id="2026_01_BUF_KC", kickoff="",
        kickoff_utc=KICKOFF.isoformat(), team="KC", opponent="BUF", home=True,
        player_id="rb-1", player_name="Lead Back", position="RB", depth_rank=1,
        depth_slot="RB1", roster_status="ACT", injury_status=None, headshot_url="",
        history_games=17, last_team="KC", role_continuity="same team",
        persistence_weight=0.72, role_reason="current depth plus same-team usage",
        confidence="high", team_environment_source="DraftKings", implied_team_points=27.5,
        scheme_context={"carry_delta": 2.5}, metrics={"rushing_yards": 92.0},
    )
    values.update(changes)
    return PlayerProjection(**values)


def _quote(line, over=-115, under=-105):
    return PlayerPropQuote("e1", "KC", "BUF", "player_rush_yds", "Lead Back", line, over, under,
                           "draftkings", "DraftKings", None)


def _open_gate(monkeypatch):
    """Let the fixture player's line through the slip evidence rules, to test
    pick mechanics; the rules themselves are tested in test_slips.py."""
    from types import SimpleNamespace

    from nflmodel import slips

    monkeypatch.setattr(slips, "candidate_legs", lambda slate, now=None: (
        [SimpleNamespace(player_id="rb-1", metric="rushing_yards")], {}, []))


@pytest.fixture
def pricing(monkeypatch):
    """Calibration off (p_over = the distribution's own probability)."""
    from nflmodel import prop_pricing

    _open_gate(monkeypatch)
    monkeypatch.setattr(prop_pricing, "CALIBRATION", {"s": 1.0, "c": 0.5})
    return prop_pricing


def test_prop_pick_is_priced_from_the_matrix_distribution(slate, pricing):
    s = replace(slate, week=1, player_projections=[_player()],
                player_prop_quotes=[_quote(68.5)])
    picks = bb.prop_picks(s, {})
    assert len(picks) == 1 and picks[0].side == "over"
    assert "Matrix projects 92.0 rushing yards" in picks[0].angle
    assert "Role: RB1" in picks[0].angle and "+2.5 team carries" in picks[0].angle
    assert "Held-out record of these leans" in picks[0].angle
    # A line at the projection's median is even: no lean either way.
    from nflmodel import prop_distributions

    median = prop_distributions.distribution("rushing_yards", 92.0)["p50"]
    assert bb.prop_picks(replace(s, player_prop_quotes=[_quote(median, -110, -110)]), {}) == []


def test_leans_publish_every_week_and_say_how_strong_they_are(slate, monkeypatch):
    from nflmodel import prop_pricing

    _open_gate(monkeypatch)
    monkeypatch.setattr(prop_pricing, "CALIBRATION", {"s": 0.1, "c": 0.5})
    s = replace(slate, week=1, player_projections=[_player()],
                player_prop_quotes=[_quote(68.5, -110, -110)])
    pick = bb.prop_picks(s, {})[0]
    # Calibrated toward even, this one is a lean, not a play.
    assert pick.probability < bb.MIN_PROP_PROBABILITY and "lean" in pick.tags
    assert not prop_pricing.research_only() and "research-only" not in pick.tags
    monkeypatch.setattr(prop_pricing, "EVIDENCE", {**prop_pricing.EVIDENCE, "plays": 69,
                                                   "wins": 32, "hit_rate": 0.464})
    pick = bb.prop_picks(s, {})[0]
    assert "research-only" in pick.tags   # still published, labelled


def test_a_prop_outside_the_evidence_rules_never_publishes(slate):
    # Same line, real gate: an RB rushing-yards over has no record (17-18) and the
    # fixture player has no games this season, so best bets and slips agree: no.
    s = replace(slate, week=1, player_projections=[_player()],
                player_prop_quotes=[_quote(68.5)], player_results=[])
    assert bb.prop_picks(s, {}) == []


def test_an_id_matched_line_without_prices_reads_as_fifty_fifty(slate, pricing):
    quote = PlayerPropQuote("e1", "KC", "BUF", "player_rush_yds", "L. Back", 68.5, -110, -110,
                            "draftkings", "DraftKings (via ESPN)", None, player_id="rb-1",
                            priced=False)
    picks = bb.prop_picks(replace(slate, week=1, player_projections=[_player()],
                                  player_prop_quotes=[quote]), {})
    assert len(picks) == 1 and "versus 50% priced in" in picks[0].angle


def test_best_bets_are_logged_and_graded(slate, tmp_path, pricing, monkeypatch):
    _skilled(monkeypatch)
    game = _game(slate)
    s = replace(slate, week=1, projections=[game], player_projections=[_player()],
                player_prop_quotes=[_quote(68.5)], injuries=[])
    picks = [p.to_json() for p in bb.build(s)]
    assert {p["family"] for p in picks} == {"spread", "total", "prop"}
    path = tmp_path / "ledger.json"
    ledger.update(season=2026, projections=[], schedule=[], best_bets=picks, path=path,
                  recorded_at=NOW)
    result = [{"season": 2026, "week": 1, "home_team": game.home, "away_team": game.away,
               "home_score": 27, "away_score": 17}]
    stats = [{"season": 2026, "week": 1, "team": "KC", "player_id": "rb-1",
              "rushing_yards": 101}]
    payload = ledger.update(season=2026, projections=[], schedule=result, player_results=stats,
                            path=path, recorded_at=KICKOFF + timedelta(days=1))
    results = {b["family"]: b["result"] for b in payload["best_bets"]}
    assert results == {"spread": "win", "total": "loss", "prop": "win"}
    summary = payload["summary"]["best_bets"]
    assert summary["total"]["units"] == -1.0 and summary["spread"]["win"] == 1


def test_a_published_pick_is_locked_at_its_first_line(slate, tmp_path, monkeypatch):
    _skilled(monkeypatch)
    s = replace(slate, week=1, projections=[_game(slate)], player_projections=[],
                player_prop_quotes=[], injuries=[])
    picks = [p.to_json() for p in bb.build(s)]
    path = tmp_path / "ledger.json"
    ledger.update(season=2026, projections=[], schedule=[], best_bets=picks, path=path,
                  recorded_at=NOW)
    spread = next(p for p in picks if p["family"] == "spread")
    moved = dict(spread, line=spread["line"] + 1.5)
    payload = ledger.update(season=2026, projections=[], schedule=[], best_bets=[moved],
                            path=path, recorded_at=NOW + timedelta(hours=6))
    assert len(payload["best_bets"]) == len(picks)          # nothing withdrawn
    assert next(b for b in payload["best_bets"]
                if b["family"] == "spread")["line"] == spread["line"]


def test_evidence_gate_line():
    from nflmodel import site

    assert "None graded yet" in site._gate_line({})
    assert "not met" in site._gate_line({"spread": {"win": 30, "loss": 20}})
    assert "met." in site._gate_line({"spread": {"win": 160, "loss": 100}})
