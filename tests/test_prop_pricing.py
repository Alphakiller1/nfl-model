from nflmodel import prop_pricing
from nflmodel.sources import espn_props

ATHLETE = "http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2026/athletes/{}?lang=en"


def _item(athlete, name, line, opened=None):
    item = {"athlete": {"$ref": ATHLETE.format(athlete)}, "type": {"id": "13", "name": name},
            "lastUpdated": "2026-10-03T18:32Z", "current": {"target": {"value": line}}}
    if opened is not None:
        item["open"] = {"target": {"value": opened}}
    return item


def test_espn_prop_lines_are_identified_by_athlete_id_and_deduplicated():
    payload = {"items": [
        _item("111", "Total Receiving Yards (incl. overtime)", 54.5, 48.5),
        _item("111", "Total Receiving Yards (incl. overtime)", 54.5, 48.5),  # the under row
        _item("111", "Receiving Yards Milestones", 75.0),                     # alt line: ignored
        _item("222", "Total Carries (incl. overtime)", 14.5),
        _item("999", "Total Receptions (incl. overtime)", 4.5),               # not on a roster
    ]}
    players = espn_props.player_index([
        {"espn_id": "111.0", "gsis_id": "00-1", "full_name": "Wide Out"},
        {"espn_id": "222", "gsis_id": "00-2", "full_name": "Run Back"},
    ])
    quotes = espn_props.parse(payload, event_id="e1", home="KC", away="BUF", players=players)
    assert [(q.player_id, q.market, q.line) for q in quotes] == [
        ("00-1", "player_reception_yds", 54.5), ("00-2", "player_rush_attempts", 14.5)]
    assert quotes[0].open_line == 48.5 and quotes[0].priced is False
    assert quotes[0].player_name == "Wide Out"


def test_price_moves_the_line_part_way_and_is_even_at_the_line():
    priced = prop_pricing.price("passing_yards", 260.0, 230.5)
    coef = prop_pricing.COEFFICIENTS["passing"]
    assert priced["fair"] == round(230.5 + coef["w"] * 29.5, 2)
    assert priced["p_over"] > 0.5
    even = prop_pricing.price("passing_yards", 230.5, 230.5)
    assert abs(even["p_over"] - 0.5) < 1e-9          # passing has no lean
    assert prop_pricing.price("kicking_points", 8.0, 7.5) is None


def test_props_stay_research_only_until_the_held_out_record_clears_break_even(monkeypatch):
    monkeypatch.setattr(prop_pricing, "EVIDENCE", {"weeks": [], "plays": 500, "wins": 250,
                                                   "hit_rate": 0.50})
    assert prop_pricing.research_only()
    monkeypatch.setattr(prop_pricing, "EVIDENCE", {"weeks": [], "plays": 500, "wins": 275,
                                                   "hit_rate": 0.55})
    assert not prop_pricing.research_only()
    monkeypatch.setattr(prop_pricing, "EVIDENCE", {"weeks": [], "plays": 40, "wins": 30,
                                                   "hit_rate": 0.75})
    assert prop_pricing.research_only()
