import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import audit_consistency as audit  # noqa: E402


def _board(**changes):
    board = {
        "best_bets": [],
        "games": [{"away": "GB", "home": "TB", "projected_home_score": 20.7,
                   "projected_away_score": 23.9, "projected_total": 44.63, "model_margin": -3.23,
                   "published_margin": -3.0, "win_probability": 0.41, "total_modelled": True,
                   "qb_out": ["Baker Mayfield (injury report: Out)"],
                   "availability_margin": -4.18, "availability_total": -3.13}],
        "player_projections": [
            {"player_id": "qb", "player_name": "QB", "team": "TB", "position": "QB",
             "depth_rank": 1, "injury_status": None, "metrics": {"pass_attempts": 30.0}},
            {"player_id": "te", "player_name": "TE", "team": "TB", "position": "TE",
             "depth_rank": 1, "injury_status": None, "metrics": {"targets": 5.0}}],
        "prop_slips": {"slips": []},
    }
    board.update(changes)
    return board


def test_a_consistent_board_passes():
    assert audit.audit(_board()) == []


def test_a_total_priced_as_if_the_model_were_the_truth_fails():
    pick = {"family": "total", "selection": "Over 38.5", "probability": 0.763, "edge": 9.3,
            "model_number": 47.8, "book_number": 38.5, "side": "over"}
    failures = audit.audit(_board(best_bets=[pick]))
    assert any("exceeds the market's measured skill" in f for f in failures)


def test_a_quarterback_out_without_a_total_adjustment_fails():
    game = dict(_board()["games"][0], availability_total=0.0)
    assert any("total is unadjusted" in f for f in audit.audit(_board(games=[game])))


def test_scoreline_win_probability_and_injury_contradictions_fail():
    game = dict(_board()["games"][0], projected_total=50.0, win_probability=0.7)
    out = audit.audit(_board(games=[game]))
    assert any("does not add to total" in f for f in out)
    assert any("favours the other side" in f for f in out)
    players = _board()["player_projections"] + [
        {"player_id": "wr", "player_name": "WR", "team": "TB", "position": "WR", "depth_rank": 1,
         "injury_status": "Out", "metrics": {"targets": 40.0}}]
    out = audit.audit(_board(player_projections=players))
    assert any("listed Out" in f for f in out) and any("exceed team attempts" in f for f in out)


def test_a_prop_pick_in_a_market_without_a_record_fails():
    players = _board()["player_projections"] + [
        {"player_id": "rb", "player_name": "RB", "team": "TB", "position": "RB", "depth_rank": 1,
         "injury_status": None, "metrics": {"carries": 16.0}}]
    pick = {"family": "prop", "selection": "RB under 21.5 rush attempts", "probability": 0.56,
            "player_id": "rb", "metric": "rush_attempts", "side": "under", "line": 21.5}
    out = audit.audit(_board(player_projections=players, best_bets=[pick]))
    assert any("market without a record" in f for f in out)


def _report_row(**changes):
    row = {"game": "KC @ LV", "selection": "Model lower", "market": 47.5, "model": 41.9,
           "season_only": 49.0, "gap": 5.6, "conflicts": ["this season's form"]}
    row.update(changes)
    return row


def test_report_gaps_must_not_be_calls_and_must_name_disagreement():
    ok = _board(weekly_report={"top_totals": [_report_row()], "top_spreads": []})
    assert audit.audit(ok) == []
    call = _board(weekly_report={"top_totals": [_report_row(selection="UNDER")]})
    assert any("presented as a call" in f for f in audit.audit(call))
    hidden = _board(weekly_report={"top_totals": [_report_row(conflicts=[])]})
    assert any("disagrees but is not named" in f for f in audit.audit(hidden))
