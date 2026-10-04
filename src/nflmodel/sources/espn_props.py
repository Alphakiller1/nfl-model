"""DraftKings player-prop lines as published by ESPN's core odds API.

The Odds API is the only source that prices player props, and its allowance
ran dry in September 2026; the board then published `player_prop_quotes: 0`
and every prop comparison went dark. ESPN's core API carries DraftKings'
player over/under lines (provider 100) for every game at no cost, including
games already played, which is what makes grading against the line possible.

ESPN publishes the line (open and current) but not the prices, so these quotes
are marked ``priced=False`` and carry a nominal -110 on both sides; pricing
reads them as a de-vigged 50/50 line.

Players are identified by ESPN athlete id, joined to gsis through the weekly
roster's ``espn_id`` column, so a quote never relies on name matching.
"""

from __future__ import annotations

import concurrent.futures

from .. import teams
from . import espn_odds
from .oddsapi import PlayerPropQuote, normalise, team_index

PROP_BETS = ("https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/"
             "{event}/competitions/{event}/odds/100/propBets?limit=1000")
NOMINAL_PRICE = -110.0
# ESPN prop type name -> the Odds API market key the rest of the model speaks.
MARKETS = {
    "total passing yards (incl. overtime)": "player_pass_yds",
    "total passing touchdowns (incl. overtime)": "player_pass_tds",
    "total pass completions (incl. overtime)": "player_pass_completions",
    "total pass attempts (incl. overtime)": "player_pass_attempts",
    "total passing attempts (incl. overtime)": "player_pass_attempts",
    "total passing interceptions (incl. overtime)": "player_pass_interceptions",
    "total rushing yards (incl. overtime)": "player_rush_yds",
    "total carries (incl. overtime)": "player_rush_attempts",
    "total receiving yards (incl. overtime)": "player_reception_yds",
    "total receptions (incl. overtime)": "player_receptions",
    "total rushing plus receiving yards (incl. overtime)": "player_rush_reception_yds",
    "total passing plus rushing yards (incl. overtime)": "player_pass_rush_yds",
    "total field goals made (incl. overtime)": "player_field_goals",
    "total kicking points (incl. overtime)": "player_kicking_points",
    "total extra points made (incl. overtime)": "player_pats",
}


def _athlete_id(item: dict) -> str | None:
    ref = str((item.get("athlete") or {}).get("$ref") or "")
    if "/athletes/" not in ref:
        return None
    return ref.split("/athletes/")[1].split("?")[0].split("/")[0] or None


def parse(payload: dict, *, event_id: str, home: str, away: str,
          players: dict[str, tuple[str, str]]) -> list[PlayerPropQuote]:
    """Quotes from one event's propBets payload.

    ``players`` maps ESPN athlete id -> (gsis id, display name). An athlete the
    roster cannot identify is skipped rather than guessed by name.
    """
    out: list[PlayerPropQuote] = []
    seen: set[tuple[str, str]] = set()
    for item in payload.get("items") or []:
        market = MARKETS.get(str((item.get("type") or {}).get("name") or "").strip().lower())
        athlete = _athlete_id(item)
        line = ((item.get("current") or {}).get("target") or {}).get("value")
        if market is None or athlete is None or line is None:
            continue
        identity = players.get(athlete)
        # Over and under arrive as two identical rows; keep one.
        if identity is None or (athlete, market) in seen:
            continue
        seen.add((athlete, market))
        opened = ((item.get("open") or {}).get("target") or {}).get("value")
        out.append(PlayerPropQuote(
            event_id=event_id, home_team=home, away_team=away, market=market,
            player_name=identity[1], line=float(line),
            over_price=NOMINAL_PRICE, under_price=NOMINAL_PRICE,
            book="draftkings", book_title="DraftKings (via ESPN)",
            last_update=item.get("lastUpdated"),
            player_id=identity[0], open_line=None if opened is None else float(opened),
            priced=False,
        ))
    return out


def player_index(roster: list[dict]) -> dict[str, tuple[str, str]]:
    index = {}
    for row in roster:
        espn = str(row.get("espn_id") or "").split(".")[0].strip()
        gsis = str(row.get("gsis_id") or "").strip()
        if espn and gsis:
            index[espn] = (gsis, str(row.get("full_name") or row.get("player_name") or ""))
    return index


def fetch(needed: set[tuple[str, str]], roster: list[dict], *,
          events: list[dict] | None = None) -> list[PlayerPropQuote]:
    """Prop lines for the (home, away) games in ``needed``.

    ``events`` is ESPN scoreboard rows as `espn_odds.lines` returns them; it is
    read from the scoreboard when omitted.
    """
    if events is None:
        events, _ = espn_odds.lines()
    aliases = team_index()
    players = player_index(roster)
    wanted = []
    for event in events:
        home = aliases.get(normalise(event.get("home_name") or ""))
        away = aliases.get(normalise(event.get("away_name") or ""))
        if home and away and (teams.canonical(home), teams.canonical(away)) in needed:
            wanted.append((str(event["event_id"]), teams.canonical(home), teams.canonical(away)))

    def one(spec):
        event_id, home, away = spec
        try:
            payload = espn_odds._get(PROP_BETS.format(event=event_id))
        except Exception:
            return []  # one unreadable game must not take the slate's props down
        return parse(payload, event_id=event_id, home=home, away=away, players=players)

    out: list[PlayerPropQuote] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for quotes in pool.map(one, wanted):
            out.extend(quotes)
    return out

