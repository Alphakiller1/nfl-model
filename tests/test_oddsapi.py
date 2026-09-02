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
