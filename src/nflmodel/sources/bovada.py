"""Pregame NFL player props from Bovada's public coupon feed.

Why this source exists
----------------------
On 2026-11-25 every prop line source this repo had was dark, and the board had
been publishing `player_prop_quotes: 0` without saying why:

* **The Odds API** prices props but `fetch_player_props` refused to spend below
  an 80-credit floor against a 70-credit balance, so it broke before its first
  request. Even with the floor lowered it cannot cover a slate: the event-odds
  endpoint bills one credit per market per region, so nine markets across
  sixteen games costs about 150 credits a refresh.
* **ESPN** (`espn_props`) carries DraftKings' prop lines for free, but the
  `propBets` endpoint **404s until a game kicks off** — verified 2026-11-25,
  where week 12's played games returned 1,500-1,900 items each and every week 13
  game returned 404. It grades; it cannot price.
* **PrizePicks** (`prizepicks`) read `<article ... projection>` cards off the
  public research pages. Those pages now render only historical game logs; zero
  cards are served. That parser fails closed, correctly, and returns nothing.

Bovada publishes its whole NFL coupon -- game lines and player props, with
American prices on both sides and alternate ladders -- as one unauthenticated
JSON document. No key, no quota, and it is posted days before kickoff.

What it is and is not
---------------------
It is a real book's two-sided pregame market, so it de-vigs honestly. It is
**not DraftKings**, and that matters for pricing: `prop_pricing` fitted its
shrink and fair-number blend on DraftKings closing lines, and Bovada holds
noticeably more (-245/+180 on a passing-yards line is a ~7% hold against DK's
usual 4-5%). Quotes therefore carry ``book="bovada"`` so a consumer can tell
which record applies to them, and `prop_pricing.calibration_applies` is what
decides whether a DK-fitted calibration may price one.

The feed posts alternate ladders for most markets. Only the **primary** line is
kept -- the rung whose two prices are closest to even -- because a ladder lets a
model pick the rung that flatters it most, which is a way of fitting the line
rather than forecasting the outcome.

Players are named, not keyed: the market description reads
``Total Passing Yards - Jordan Love (GB)``. Names are resolved against the
weekly roster, and a name the roster cannot place is dropped rather than guessed.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime

from .. import teams
from .oddsapi import PlayerPropQuote, match_team, normalise, team_index

COUPON_URL = ("https://www.bovada.lv/services/sports/event/coupon/events/A/description/"
              "football/nfl")
# A generic client name: this identifies the software, and never the user.
USER_AGENT = "nfl-model/1.0 (+https://github.com/Alphakiller1/nfl-model)"
TIMEOUT_SECONDS = 30
MAX_BYTES = 24_000_000

BOOK = "bovada"
BOOK_TITLE = "Bovada"

# Bovada market description -> the market key the rest of the model speaks.
# Keyed on the text before the " - <Player>" separator.
MARKETS = {
    "total passing yards": "player_pass_yds",
    "total passing touchdowns": "player_pass_tds",
    "total passing tds": "player_pass_tds",
    "total pass completions": "player_pass_completions",
    "total completions": "player_pass_completions",
    "total passing attempts": "player_pass_attempts",
    "total pass attempts": "player_pass_attempts",
    "total passing interceptions": "player_pass_interceptions",
    "total interceptions": "player_pass_interceptions",
    "total interceptions thrown": "player_pass_interceptions",
    "total rushing yards": "player_rush_yds",
    "total rushing attempts": "player_rush_attempts",
    "total rush attempts": "player_rush_attempts",
    "total carries": "player_rush_attempts",
    "total receiving yards": "player_reception_yds",
    "total receptions": "player_receptions",
    "total rushing + receiving yards": "player_rush_reception_yds",
    "total rushing and receiving yards": "player_rush_reception_yds",
    "total rushing & receiving yards": "player_rush_reception_yds",
    "total passing + rushing yards": "player_pass_rush_yds",
    "total passing and rushing yards": "player_pass_rush_yds",
    "total passing & rushing yards": "player_pass_rush_yds",
    "total passing & rushing yards o/u": "player_pass_rush_yds",
}

# Only the full game. A first-half or first-quarter line is a different bet and
# the projection layer does not forecast one.
FULL_GAME = {"game", "regular time", "full game"}


class BovadaError(RuntimeError):
    """The coupon feed could not be read."""


@dataclass(frozen=True)
class BovadaStatus:
    state: str
    fetched_at: str | None = None
    events: int = 0
    matched_events: int = 0
    quotes: int = 0
    markets: int = 0
    unmatched_players: int = 0
    unknown_markets: tuple[str, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "provider": BOOK,
            "state": self.state,
            "source": COUPON_URL,
            "fetched_at": self.fetched_at,
            "events": self.events,
            "matched_events": self.matched_events,
            "quotes": self.quotes,
            "priced_markets": self.markets,
            "unmatched_players": self.unmatched_players,
            "unknown_markets": list(self.unknown_markets),
            "error": self.error,
            "note": ("Bovada pregame two-sided prices. Not DraftKings: a "
                     "DraftKings-fitted calibration does not transfer to these lines."),
        }


def _stamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _american(value) -> float | None:
    """Bovada prints 'EVEN' for +100 and a signed string otherwise."""
    text = str(value or "").strip().upper()
    if not text:
        return None
    if text in {"EVEN", "EVN", "PK"}:
        return 100.0
    try:
        return float(text.replace("+", ""))
    except ValueError:
        return None


def _implied(price: float) -> float:
    return 100.0 / (price + 100.0) if price > 0 else -price / (-price + 100.0)


def _split_market(description: str) -> tuple[str, str] | None:
    """'Total Passing Yards - Jordan Love (GB)' -> ('total passing yards', 'Jordan Love')."""
    text = str(description or "")
    if " - " not in text:
        return None
    label, _, player = text.rpartition(" - ")
    player = player.strip()
    if player.endswith(")") and "(" in player:
        player = player[: player.rindex("(")].strip()
    if not player:
        return None
    return label.strip().lower(), player


def player_index(roster: list[dict]) -> dict[str, tuple[str, str]]:
    """normalised name -> (gsis id, display name), for one club-agnostic lookup.

    A name held by two active players is dropped: a prop credited to the wrong
    player is worse than a prop that is missing.
    """
    seen: dict[str, set[str]] = {}
    named: dict[str, tuple[str, str]] = {}
    for row in roster:
        gsis = str(row.get("gsis_id") or "").strip()
        full = str(row.get("full_name") or row.get("player_name") or "").strip()
        if not gsis or not full:
            continue
        key = normalise(full)
        seen.setdefault(key, set()).add(gsis)
        named[key] = (gsis, full)
    return {key: value for key, value in named.items() if len(seen[key]) == 1}


def _primary_rung(rungs: dict[float, dict[str, float]]) -> float | None:
    """The ladder rung closest to a two-sided even market.

    Bovada posts several handicaps per market. The rung a book expects to be a
    coin flip is the one it prices nearest even, and it is the only rung whose
    two prices are both tight enough to de-vig meaningfully.
    """
    best, best_gap = None, None
    for line, prices in rungs.items():
        over, under = prices.get("over"), prices.get("under")
        if over is None or under is None:
            continue
        gap = abs(_implied(over) - _implied(under))
        if best_gap is None or gap < best_gap:
            best, best_gap = line, gap
    return best


def parse(payload, *, roster: list[dict], now: datetime | None = None
          ) -> tuple[list[PlayerPropQuote], BovadaStatus]:
    """Quotes for every upcoming game in one coupon document."""
    moment = now or datetime.now(UTC)
    players = player_index(roster)
    # Bovada names clubs in full ("Los Angeles Rams"). `teams.canonical` keys on
    # abbreviations and passes a full name straight through, so resolving with it
    # produced quotes stamped "LOS ANGELES RAMS" that matched no slate game.
    aliases = team_index()
    groups = payload if isinstance(payload, list) else [payload]
    events = []
    for group in groups:
        if isinstance(group, dict):
            events.extend(group.get("events") or [])

    out: list[PlayerPropQuote] = []
    unmatched: set[str] = set()
    unknown: set[str] = set()
    matched_events = 0
    for event in events:
        start = event.get("startTime")
        try:
            kickoff = datetime.fromtimestamp(float(start) / 1000.0, UTC)
        except (TypeError, ValueError):
            continue
        if kickoff <= moment:
            continue   # already started: the line is no longer a pregame quote
        home = away = None
        for competitor in event.get("competitors") or []:
            club = match_team(str(competitor.get("name") or ""), aliases)
            if not club:
                continue
            if competitor.get("home"):
                home = club
            else:
                away = club
        if not home or not away:
            continue
        event_id = str(event.get("id") or "")
        # line -> {"over": price, "under": price} per (player, market)
        ladders: dict[tuple[str, str], dict[float, dict[str, float]]] = {}
        names: dict[str, str] = {}
        for group in event.get("displayGroups") or []:
            for market in group.get("markets") or []:
                period = str(((market.get("period") or {}).get("description")) or "").lower()
                if period and period not in FULL_GAME:
                    continue
                split = _split_market(market.get("description"))
                if split is None:
                    continue
                label, player_name = split
                key = MARKETS.get(label)
                if key is None:
                    if label.startswith("total "):
                        unknown.add(label)
                    continue
                identity = players.get(normalise(player_name))
                if identity is None:
                    unmatched.add(player_name)
                    continue
                names[identity[0]] = identity[1]
                rungs = ladders.setdefault((identity[0], key), {})
                for outcome in market.get("outcomes") or []:
                    side = str(outcome.get("type") or "").upper()
                    price = (outcome.get("price") or {})
                    american = _american(price.get("american"))
                    try:
                        line = float(price.get("handicap"))
                    except (TypeError, ValueError):
                        continue
                    if american is None or side not in {"O", "U"}:
                        continue
                    rungs.setdefault(line, {})["over" if side == "O" else "under"] = american
        for (player_id, key), rungs in ladders.items():
            line = _primary_rung(rungs)
            if line is None:
                continue
            prices = rungs[line]
            out.append(PlayerPropQuote(
                event_id=event_id, home_team=home, away_team=away, market=key,
                player_name=names[player_id], line=float(line),
                over_price=float(prices["over"]), under_price=float(prices["under"]),
                book=BOOK, book_title=BOOK_TITLE, last_update=_stamp(),
                player_id=player_id, open_line=None, priced=True,
            ))
        if any(quote.event_id == event_id for quote in out):
            matched_events += 1

    # An empty coupon and a coupon with no usable props are different
    # failures and must not share a message. Observed 2026-11-25: the endpoint
    # served 3.3 MB with 16 events, then two hours later returned a bare `[]`
    # around the slate transition. Bovada is the best free pregame source, not
    # a guaranteed one -- which is the whole reason the chain keeps ESPN behind
    # it for games that have kicked off.
    if not events:
        reason = "coupon returned no events"
    elif not out:
        reason = "coupon read but no full-game player props matched the roster"
    else:
        reason = None
    status = BovadaStatus(
        state="fresh" if out else "unavailable",
        fetched_at=_stamp(),
        events=len(events),
        matched_events=matched_events,
        quotes=len(out),
        markets=len({quote.market for quote in out}),
        unmatched_players=len(unmatched),
        unknown_markets=tuple(sorted(unknown)[:12]),
        error=reason,
    )
    return out, status


def fetch(roster: list[dict], *, needed: set[tuple[str, str]] | None = None,
          now: datetime | None = None) -> tuple[list[PlayerPropQuote], BovadaStatus]:
    """Read the coupon and return pregame player props.

    ``needed`` is the slate's open ``(home, away)`` games; quotes outside it are
    dropped. A read failure returns an empty list and a status that names the
    failure, because a source that goes quiet must say so -- the silent version
    of this is what let the board publish zero props for weeks.
    """
    request = urllib.request.Request(COUPON_URL, headers={
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            raw = response.read(MAX_BYTES + 1)
    except (urllib.error.URLError, OSError) as exc:
        detail = getattr(exc, "code", None) or type(exc).__name__
        return [], BovadaStatus(state="error", fetched_at=_stamp(),
                                error=f"coupon fetch failed: {detail}")
    if len(raw) > MAX_BYTES:
        return [], BovadaStatus(state="error", fetched_at=_stamp(),
                                error=f"coupon exceeded {MAX_BYTES} bytes")
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        return [], BovadaStatus(state="error", fetched_at=_stamp(),
                                error=f"coupon is not JSON: {exc}")
    quotes, status = parse(payload, roster=roster, now=now)
    if needed is not None:
        quotes = [q for q in quotes
                  if (teams.canonical(q.home_team), teams.canonical(q.away_team)) in needed]
        status = BovadaStatus(
            state="fresh" if quotes else "unavailable",
            fetched_at=status.fetched_at, events=status.events,
            matched_events=len({q.event_id for q in quotes}), quotes=len(quotes),
            markets=len({q.market for q in quotes}),
            unmatched_players=status.unmatched_players,
            unknown_markets=status.unknown_markets,
            error=None if quotes else (
                status.error if not status.events else "no coupon game matched the slate"),
        )
    return quotes, status
