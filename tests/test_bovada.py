"""Bovada is the only free source that can price a prop before kickoff."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from nflmodel.sources import bovada

ROSTER = [
    {"gsis_id": "00-0000001", "full_name": "Jordan Love"},
    {"gsis_id": "00-0000002", "full_name": "Matthew Stafford"},
    {"gsis_id": "00-0000003", "full_name": "Christian Watson"},
    # Two active players share this name, so neither may claim its quotes.
    {"gsis_id": "00-0000004", "full_name": "Mike Williams"},
    {"gsis_id": "00-0000005", "full_name": "Mike Williams"},
]


def _coupon(start_ms: int, markets: list[dict]) -> list[dict]:
    return [{"events": [{
        "id": "25556103",
        "description": "Green Bay Packers @ Los Angeles Rams",
        "startTime": start_ms,
        "competitors": [{"name": "Los Angeles Rams", "home": True},
                        {"name": "Green Bay Packers", "home": False}],
        "displayGroups": [{"description": "Passing Yards", "markets": markets}],
    }]}]


def _market(description: str, rungs: list[tuple[float, str, str]], period: str = "Game") -> dict:
    outcomes = []
    for line, over, under in rungs:
        outcomes.append({"type": "O", "description": "Over",
                         "price": {"handicap": line, "american": over}})
        outcomes.append({"type": "U", "description": "Under",
                         "price": {"handicap": line, "american": under}})
    return {"description": description, "period": {"description": period},
            "outcomes": outcomes}


def _soon() -> int:
    return int((datetime.now(UTC) + timedelta(hours=8)).timestamp() * 1000)


def test_full_name_clubs_resolve_to_abbreviations():
    """`teams.canonical` passes a full club name straight through; the first
    version of this source stamped quotes "LOS ANGELES RAMS" and matched no game."""
    quotes, status = bovada.parse(
        _coupon(_soon(), [_market("Total Passing Yards - Jordan Love (GB)",
                                  [(242.5, "-115", "-115")])]),
        roster=ROSTER)
    assert status.state == "fresh"
    assert [(q.home_team, q.away_team) for q in quotes] == [("LA", "GB")]


def test_only_the_rung_closest_to_even_is_kept():
    """A ladder lets a model pick the rung that flatters it. Keep the book's
    own coin-flip rung instead."""
    quotes, _ = bovada.parse(
        _coupon(_soon(), [_market("Total Passing Yards - Jordan Love (GB)", [
            (222.5, "-310", "+230"),
            (242.5, "-115", "-115"),
            (262.5, "+175", "-220"),
        ])]),
        roster=ROSTER)
    assert len(quotes) == 1
    assert quotes[0].line == 242.5
    assert (quotes[0].over_price, quotes[0].under_price) == (-115.0, -115.0)


def test_even_is_read_as_plus_one_hundred():
    quotes, _ = bovada.parse(
        _coupon(_soon(), [_market("Total Receptions - Christian Watson (GB)",
                                  [(4.5, "-130", "EVEN")])]),
        roster=ROSTER)
    assert quotes[0].under_price == 100.0


def test_half_and_quarter_markets_are_not_full_game_props():
    quotes, status = bovada.parse(
        _coupon(_soon(), [_market("Total Passing Yards - Jordan Love (GB)",
                                  [(120.5, "-115", "-115")], period="First Half")]),
        roster=ROSTER)
    assert quotes == []
    assert status.state == "unavailable"
    assert status.error


def test_a_started_game_is_no_longer_a_pregame_quote():
    started = int((datetime.now(UTC) - timedelta(minutes=5)).timestamp() * 1000)
    quotes, _ = bovada.parse(
        _coupon(started, [_market("Total Passing Yards - Jordan Love (GB)",
                                  [(242.5, "-115", "-115")])]),
        roster=ROSTER)
    assert quotes == []


def test_an_ambiguous_name_is_dropped_not_guessed():
    quotes, status = bovada.parse(
        _coupon(_soon(), [_market("Total Receiving Yards - Mike Williams (LAR)",
                                  [(45.5, "-115", "-115")])]),
        roster=ROSTER)
    assert quotes == []
    assert status.unmatched_players == 1


def test_unknown_total_markets_are_reported_not_silently_skipped():
    quotes, status = bovada.parse(
        _coupon(_soon(), [_market("Total Kickoff Return Yards - Jordan Love (GB)",
                                  [(12.5, "-115", "-115")])]),
        roster=ROSTER)
    assert quotes == []
    assert "total kickoff return yards" in status.unknown_markets


def test_a_one_sided_rung_is_unusable():
    """De-vigging needs both sides; a lone Over is not a market."""
    market = {"description": "Total Passing Yards - Jordan Love (GB)",
              "period": {"description": "Game"},
              "outcomes": [{"type": "O", "description": "Over",
                            "price": {"handicap": 242.5, "american": "-115"}}]}
    quotes, _ = bovada.parse(_coupon(_soon(), [market]), roster=ROSTER)
    assert quotes == []


def test_needed_filters_to_the_slate(monkeypatch):
    payload = _coupon(_soon(), [_market("Total Passing Yards - Jordan Love (GB)",
                                        [(242.5, "-115", "-115")])])
    monkeypatch.setattr(bovada, "parse", bovada.parse)

    class _Response:
        def __init__(self, raw):
            self._raw = raw

        def read(self, _limit=None):
            return self._raw

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    import json as _json
    monkeypatch.setattr(bovada.urllib.request, "urlopen",
                        lambda *a, **k: _Response(_json.dumps(payload).encode()))
    inside, status = bovada.fetch([], needed={("LA", "GB")})
    assert status.state == "unavailable"          # empty roster matches no player
    outside, _ = bovada.fetch(ROSTER, needed={("KC", "BUF")})
    assert outside == []
    matched, status = bovada.fetch(ROSTER, needed={("LA", "GB")})
    assert len(matched) == 1 and status.state == "fresh"
    assert inside == []


def test_a_read_failure_names_itself(monkeypatch):
    def boom(*_a, **_k):
        raise OSError("connection reset")

    monkeypatch.setattr(bovada.urllib.request, "urlopen", boom)
    quotes, status = bovada.fetch(ROSTER)
    assert quotes == []
    assert status.state == "error"
    assert "coupon fetch failed" in status.error


def test_an_empty_coupon_says_so_rather_than_blaming_the_slate(monkeypatch):
    """Observed live: the endpoint served 16 events, then a bare `[]` two hours
    later. "no coupon game matched the slate" would send the next reader
    hunting for a team-matching bug that is not there."""
    class _Response:
        def read(self, _limit=None):
            return b"[]"

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(bovada.urllib.request, "urlopen", lambda *a, **k: _Response())
    quotes, status = bovada.fetch(ROSTER, needed={("LA", "GB")})
    assert quotes == []
    assert status.events == 0
    assert status.error == "coupon returned no events"
