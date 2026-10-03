from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from nflmodel import ledger, slips
from nflmodel.sources import espn_injuries

NOW = datetime(2026, 10, 3, 18, tzinfo=timezone.utc)
KICK = (NOW + timedelta(days=1)).isoformat()


def _leg(pid, game, side="under", p=0.58, metric="receptions", team="KC", line=3.5):
    return slips.Leg(2026, 4, "KC", "BUF", f"P{pid}", pid, team, "BUF", "TE", game, KICK,
                     metric, side, line, 2.5, p, (36, 61))


def _row(pid, team, week, position="WR", targets=0, carries=0, attempts=0, receptions=0):
    return {"season": "2026", "week": str(week), "team": team, "player_id": pid,
            "position": position, "targets": str(targets), "carries": str(carries),
            "attempts": str(attempts), "receptions": str(receptions)}


def _proj(pid, team, position="WR", depth=1):
    return SimpleNamespace(player_id=pid, team=team, position=position, depth_rank=depth)


def test_only_markets_with_a_record_qualify():
    assert slips.market_allowed("under", "RB", "receptions")       # 33-16
    assert slips.market_allowed("under", "TE", "receiving_yards")  # 25-14
    assert not slips.market_allowed("under", "RB", "carries")      # 24-26: books know bell cows
    assert not slips.market_allowed("under", "WR", "receptions")   # 43-38
    assert not slips.market_allowed("over", "WR", "receptions")    # too few plays


def test_a_missing_regular_or_a_new_quarterback_flags_role_expansion():
    rows = []
    for week in (1, 2, 3):
        rows += [_row("wr1", "IND", week, targets=8), _row("wr2", "IND", week, targets=6),
                 _row("te1", "IND", week, "TE", targets=3),
                 _row("qb1", "IND", week, "QB", attempts=33),
                 _row("rb1", "IND", week, "RB", carries=20, targets=2)]
    # wr1 is Out this week; a backup QB starts.
    projections = [_proj("wr2", "IND"), _proj("te1", "IND", "TE"), _proj("rb1", "IND", "RB"),
                   _proj("qb2", "IND", "QB")]
    flags = slips.role_changes(rows, projections, season=2026, week=4)["IND"]
    assert {"receiver", "receiver:WR", "quarterback"} <= flags and "rusher" not in flags
    assert slips._blocked_by_role("under", "TE", "receptions", flags)
    assert slips._blocked_by_role("under", "RB", "receptions", {"quarterback"})
    assert not slips._blocked_by_role("over", "TE", "receptions", flags)
    assert slips._blocked_by_role("over", "QB", "passing_tds", flags)    # new starter
    assert not slips._blocked_by_role("under", "QB", "passing_tds", {"receiver"})
    assert slips._blocked_by_role("under", "QB", "rush_attempts", flags)


def test_recent_form_and_line_moves_guard_the_pick():
    rows = [{"week": str(w), "receptions": str(v)} for w, v in ((1, 5), (2, 6), (3, 3))]
    assert slips._form_against(rows, "receptions", "under", 4.5)      # cleared it twice
    assert not slips._form_against(rows, "receptions", "under", 6.5)
    assert slips._moved_against("under", 3.5, 4.5)                    # worse number for an under
    assert not slips._moved_against("under", 6.5, 5.5)
    assert slips._moved_against("over", 8.5, 7.5)


def test_slips_use_each_player_once_and_never_pair_a_game():
    legs = [_leg(str(i), f"g{i // 2}") for i in range(12)]
    built = slips.build_slips(legs, 2, 6)
    assert len(built) == 6
    assert all(a.game_id != b.game_id for a, b in built)
    assert len({leg.player_id for slip in built for leg in slip}) == 12


def test_the_draft_reaches_past_a_game_clash_to_fill_every_slip():
    # Two strong legs share a game: one slip cannot take both, so a weaker
    # leg further down fills it.
    legs = [_leg("a", "g1", p=0.6), _leg("b", "g1", p=0.59), _leg("c", "g2", p=0.55),
            _leg("d", "g3", p=0.54)]
    built = slips.build_slips(legs, 2, 2)
    assert len(built) == 2 and all(a.game_id != b.game_id for a, b in built)


def test_one_winning_two_pick_clears_a_fifty_dollar_target():
    legs = [_leg(str(i), f"g{i}", p=0.99) for i in range(12)]
    built = slips.build_slips(legs, 2, 6)
    est = slips.simulate(built, "power2", 25.0, 50.0, runs=500)
    assert est["p_payout_above_target"] > 0.95
    again = slips.simulate(built, "power2", 25.0, 50.0, runs=500)
    assert est == again                                   # seeded, reproducible


def test_the_ledger_freezes_the_plan_at_first_kickoff_and_grades_it(tmp_path):
    legs = [_leg("a", "g1"), _leg("b", "g2", side="over", metric="passing_tds", line=1.5)]
    plan = slips.SlipPlan(2026, 4, "power2", 1, 25.0, 50.0, slips=[legs]).to_json()
    plan["dropped"] = [{**plan["slips"][0]["legs"][0], "player_id": "c",
                        "reason": "recent form against the pick"}]
    path = tmp_path / "ledger.json"
    ledger.update(season=2026, projections=[], schedule=[], prop_slips=plan, path=path,
                  recorded_at=NOW)
    schedule = [{"season": 2026, "week": 4, "home_team": "KC", "away_team": "BUF",
                 "home_score": 24, "away_score": 20}]
    stats = [{"season": 2026, "week": 4, "team": "KC", "player_id": "a", "receptions": 2},
             {"season": 2026, "week": 4, "team": "KC", "player_id": "b", "passing_tds": 2},
             {"season": 2026, "week": 4, "team": "KC", "player_id": "c", "receptions": 5}]
    payload = ledger.update(season=2026, projections=[], schedule=schedule, player_results=stats,
                            path=path, recorded_at=NOW + timedelta(days=2))
    entry = payload["prop_slips"][0]
    assert entry["status"] == "graded" and entry["payout"] == 75.0 and entry["beat_target"]
    summary = payload["summary"]["prop_slips"]
    assert summary["legs"] == {"win": 2, "loss": 0, "void": 0}
    assert summary["dropped_by_rule"] == {"recent form against the pick": {"win": 0, "loss": 1}}


def test_espn_game_day_status_overrides_the_friday_report():
    payload = {"injuries": [{"team": {"displayName": "Indianapolis Colts", "abbreviation": "IND"},
                             "injuries": [{"athlete": {"id": "15818", "displayName": "Keenan Allen",
                                                       "position": {"abbreviation": "WR"}},
                                           "status": "Out"},
                                          {"athlete": {"id": "1", "displayName": "Someone",
                                                       "position": {"abbreviation": "WR"}},
                                           "status": "Injured Reserve"}]}]}
    fresh = espn_injuries.parse(payload, gsis_by_espn={"15818": "00-0030279", "1": "00-1"},
                                season=2026, week=4)
    assert [r["report_status"] for r in fresh] == ["Out", "Out"]
    official = [{"team": "IND", "gsis_id": "00-0030279", "full_name": "Keenan Allen",
                 "report_status": "Questionable"}]
    merged = espn_injuries.merge(official, fresh)
    allen = next(r for r in merged if r["gsis_id"] == "00-0030279")
    assert allen["report_status"] == "Out" and len(merged) == 2
