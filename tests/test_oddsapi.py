from nflmodel.sources import oddsapi


def test_full_team_names_match_exactly_without_ambiguous_city_guessing():
    index = oddsapi.team_index()
    assert oddsapi.match_team("Los Angeles Rams", index) == "LA"
    assert oddsapi.match_team("Los Angeles Chargers", index) == "LAC"
    assert oddsapi.match_team("Los Angeles", index) is None


def test_book_line_flips_book_spread_to_expected_home_margin():
    line = oddsapi.BookLine(
        "draftkings", "DraftKings", -3.5, 47.5, -170, 145, None, None
    )
    assert line.home_margin == 3.5


def test_exact_book_selection_does_not_fall_through():
    books = [{"key": "fanduel", "title": "FanDuel"}]
    assert oddsapi._pick_book(books, "draftkings") is None


def test_fetch_lines_parses_all_three_markets_and_locks_draftkings(monkeypatch):
    captured = {}
    monkeypatch.setattr(oddsapi, "remaining", lambda: 100)

    def fake_get(path, params):
        captured.update(params)
        return ([{
            "home_team": "Buffalo Bills",
            "away_team": "New York Jets",
            "commence_time": "2026-09-13T17:00:00Z",
            "bookmakers": [{
                "key": "draftkings", "title": "DraftKings",
                "last_update": "2026-09-02T12:00:00Z",
                "markets": [
                    {"key": "spreads", "outcomes": [
                        {"name": "Buffalo Bills", "point": -6.5},
                        {"name": "New York Jets", "point": 6.5},
                    ]},
                    {"key": "totals", "outcomes": [
                        {"name": "Over", "point": 44.5},
                        {"name": "Under", "point": 44.5},
                    ]},
                    {"key": "h2h", "outcomes": [
                        {"name": "Buffalo Bills", "price": -260},
                        {"name": "New York Jets", "price": 210},
                    ]},
                ],
            }],
        }], {"source": "live", "remaining": "97", "fetched_at": "now"})

    monkeypatch.setattr(oddsapi, "_get", fake_get)
    lines = oddsapi.fetch_lines()
    line = lines[("BUF", "NYJ")]
    assert captured["bookmakers"] == "draftkings"
    assert captured["markets"] == "h2h,spreads,totals"
    assert line.home_margin == 6.5
    assert line.total == 44.5
    assert line.home_moneyline == -260
    assert line.away_moneyline == 210


def test_quota_floor_refuses_the_paid_request(monkeypatch):
    monkeypatch.setattr(oddsapi, "remaining", lambda: 4)
    called = False

    def fake_get(path, params):
        nonlocal called
        called = True
        return [], {}

    monkeypatch.setattr(oddsapi, "_get", fake_get)
    try:
        oddsapi.fetch_lines(min_remaining=20)
    except oddsapi.QuotaExhausted:
        pass
    else:
        raise AssertionError("quota floor did not fail closed")
    assert called is False


def test_player_props_require_paired_over_under_at_the_same_line():
    payload = {
        "id": "event-1",
        "home_team": "Buffalo Bills",
        "away_team": "New York Jets",
        "bookmakers": [{
            "key": "draftkings", "title": "DraftKings",
            "markets": [{
                "key": "player_pass_yds", "last_update": "2026-09-20T12:00:00Z",
                "outcomes": [
                    {"name": "Over", "description": "Josh Allen", "point": 267.5,
                     "price": -115},
                    {"name": "Under", "description": "Josh Allen", "point": 267.5,
                     "price": -105},
                    {"name": "Over", "description": "Missing Under", "point": 199.5,
                     "price": -110},
                ],
            }],
        }],
    }
    quotes = oddsapi._parse_player_props(payload, "draftkings")
    assert len(quotes) == 1
    quote = quotes[0]
    assert quote.player_name == "Josh Allen"
    assert quote.line == 267.5
    assert quote.over_price == -115
    assert quote.under_price == -105


ESPN_QUOTE = {
    "event_id": "401", "home_name": "Buffalo Bills", "away_name": "New York Jets",
    "commence_time": "2026-10-04T17:00Z", "home_spread": -6.5, "total": 44.5,
    "home_moneyline": -280.0, "away_moneyline": 225.0,
}


def test_quota_floor_falls_back_to_the_same_book_on_espn(monkeypatch):
    from nflmodel.sources import espn_odds

    monkeypatch.setattr(oddsapi, "remaining", lambda: 2)
    def paid(*args, **kwargs):
        raise AssertionError("paid call")

    monkeypatch.setattr(oddsapi, "_get", paid)
    monkeypatch.setattr(espn_odds, "lines",
                        lambda requested="draftkings": ([ESPN_QUOTE], "2026-10-01T00:00:00+00:00"))
    lines = oddsapi.fetch_lines(min_remaining=20)
    line = lines[("BUF", "NYJ")]
    assert (line.home_spread, line.total) == (-6.5, 44.5)
    assert (line.home_moneyline, line.away_moneyline) == (-280.0, 225.0)
    assert line.book == "draftkings" and line.event_id == "espn:401"
    assert oddsapi.status_report()["state"] == "fresh"


def test_a_game_the_provider_left_incomplete_is_filled_from_espn(monkeypatch):
    from nflmodel.sources import espn_odds

    monkeypatch.setattr(oddsapi, "remaining", lambda: 500)
    event = {
        "id": "e1", "home_team": "Buffalo Bills", "away_team": "New York Jets",
        "commence_time": "2026-10-04T17:00:00Z",
        "bookmakers": [{"key": "draftkings", "title": "DraftKings", "markets": [
            {"key": "h2h", "outcomes": [{"name": "Buffalo Bills", "price": -280},
                                         {"name": "New York Jets", "price": 225}]}]}],
    }
    monkeypatch.setattr(oddsapi, "_get",
                        lambda *a, **k: ([event], {"source": "live", "remaining": 499}))
    monkeypatch.setattr(espn_odds, "lines", lambda requested="draftkings": ([ESPN_QUOTE], "t"))
    line = oddsapi.fetch_lines()[("BUF", "NYJ")]
    assert line.total == 44.5 and line.home_spread == -6.5


def test_the_board_looks_ahead_once_only_monday_night_remains():
    from nflmodel import season

    def game(week, day, score):
        return {"season": 2026, "week": week, "game_type": "REG", "gameday": day,
                "home_score": score}
    week3 = [game(3, "2026-09-24", 20), game(3, "2026-09-27", 17), game(3, "2026-09-27", 24),
             game(3, "2026-09-28", None)]
    week4 = [game(4, "2026-10-01", None), game(4, "2026-10-04", None)]
    assert season.current_week(week3 + week4, 2026) == 4
    # Sunday still in play: week 3 stays the slate.
    sunday = [game(3, "2026-09-24", 20), game(3, "2026-09-27", None), game(3, "2026-09-28", None)]
    assert season.current_week(sunday + week4, 2026) == 3
    # A week that has not started is not skipped.
    fresh = [game(4, "2026-10-01", None), game(4, "2026-10-04", None)]
    assert season.current_week([game(3, "2026-09-27", 20)] + fresh, 2026) == 4
