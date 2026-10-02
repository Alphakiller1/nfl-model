"""Sharp money and line movement: where the money is concentrated, and which way
the number has moved.

Two public signals, combined per game and market (spread, total):

* **Money vs tickets** - DraftKings' % of handle against % of bets on each side
  (`sources.dk_splits`). A side drawing far more of the money than of the bets
  has fewer, larger wagers on it: the usual proxy for sharp action.
* **Line movement** - the DraftKings number when the ledger first saw it against
  now. A move toward the money side is steam; a move toward it while most
  tickets are on the other side is reverse line movement.

``concentration`` = money-minus-tickets gap + 3 per point moved toward the money
side + 5 for reverse line movement. Spots are logged before kickoff and graded on
the result and on closing-line value. Market observations, not model picks.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from . import teams

MIN_DIVERGENCE = 10.0
LIMIT = 6                   # per market
MOVE_WEIGHT = 3.0
RLM_BONUS = 5.0


@dataclass
class Spot:
    season: int
    week: int
    home: str
    away: str
    kickoff: str | None
    family: str             # spread | total
    side: str               # home | away | over | under
    selection: str
    line: float
    price: int | None
    handle_pct: float
    bets_pct: float
    divergence: float
    open_line: float | None
    move: float
    reverse: bool
    concentration: float
    model_agrees: bool | None
    note: str

    @property
    def pick_id(self) -> str:
        return "|".join((str(self.season), str(self.week), "sharp", self.family,
                         self.away, self.home))

    def to_json(self) -> dict:
        out = asdict(self)
        out["pick_id"] = self.pick_id
        return out


def resolve(name: str) -> str | None:
    """'IND Colts' -> 'IND' (DraftKings leads NFL names with the abbreviation)."""
    head = (name or "").strip().split(" ")[0].upper()
    abbr = teams.canonical(head)
    return abbr if abbr in teams.TEAMS else None


def _fmt(value: float) -> str:
    return "PK" if abs(value) < 0.05 else f"{value:+g}"


def first_lines(snapshots: list[dict], season: int, week: int) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for row in snapshots:
        if int(row.get("season", 0)) != season or int(row.get("week", 0)) != week:
            continue
        key = (row["home"], row["away"])
        if key not in out or row.get("recorded_at", "") < out[key].get("recorded_at", ""):
            out[key] = row
    return out


def _team_of(selection: str) -> str | None:
    return resolve(re.sub(r"\s*[+-]?\d+(\.\d+)?$", "", selection))


def _spread(game, p, opening) -> Spot | None:
    sides = [s for s in game.sides if s.market == "spread"]
    if len(sides) != 2 or p.book_margin is None:
        return None
    money = max(sides, key=lambda s: s.handle_pct - s.bets_pct)
    divergence = money.handle_pct - money.bets_pct
    if divergence < MIN_DIVERGENCE:
        return None
    is_home = _team_of(money.selection) == p.home
    line = -p.book_margin if is_home else p.book_margin
    open_line, move = None, 0.0
    if opening and opening.get("book_margin") is not None:
        open_line = -opening["book_margin"] if is_home else opening["book_margin"]
        move = open_line - line
    reverse = move > 0 and money.bets_pct < 50.0
    agrees = (None if p.model_margin is None
              else ((p.model_margin - p.book_margin) > 0) == is_home)
    team = p.home if is_home else p.away
    note = f"{money.handle_pct:.0f}% of the money on {team} from {money.bets_pct:.0f}% of the bets."
    if open_line is not None and abs(move) >= 0.5:
        note += (f" Line {_fmt(open_line)} -> {_fmt(line)}: "
                 f"{'toward' if move > 0 else 'away from'} the money side.")
    if reverse:
        note += " Reverse line movement: the number moved with the money, against the tickets."
    if agrees is not None:
        note += " The model agrees." if agrees else " The model is on the other side."
    score = divergence + MOVE_WEIGHT * max(move, 0.0) + (RLM_BONUS if reverse else 0.0)
    return Spot(p.season, p.week, p.home, p.away, p.kickoff_utc or None, "spread",
                "home" if is_home else "away", f"{team} {_fmt(line)}", line, money.odds,
                money.handle_pct, money.bets_pct, divergence, open_line, round(move, 1),
                reverse, round(score, 1), agrees, note)


def _total(game, p, opening) -> Spot | None:
    sides = [s for s in game.sides if s.market == "total"]
    if len(sides) != 2 or p.book_total is None:
        return None
    money = max(sides, key=lambda s: s.handle_pct - s.bets_pct)
    divergence = money.handle_pct - money.bets_pct
    if divergence < MIN_DIVERGENCE:
        return None
    over = money.selection.lower().startswith("over")
    open_line = opening.get("book_total") if opening else None
    move = 0.0
    if open_line is not None:
        move = (p.book_total - open_line) if over else (open_line - p.book_total)
    reverse = move > 0 and money.bets_pct < 50.0
    agrees = (None if p.projected_total is None
              else (p.projected_total > p.book_total) == over)
    note = (f"{money.handle_pct:.0f}% of the money on the {'over' if over else 'under'} "
            f"from {money.bets_pct:.0f}% of the bets.")
    if open_line is not None and abs(move) >= 0.5:
        note += (f" Total {open_line:g} -> {p.book_total:g}: "
                 f"{'toward' if move > 0 else 'away from'} the money side.")
    if reverse:
        note += " Reverse line movement against the ticket majority."
    if agrees is not None:
        note += " The model agrees." if agrees else " The model is on the other side."
    score = divergence + MOVE_WEIGHT * max(move, 0.0) + (RLM_BONUS if reverse else 0.0)
    return Spot(p.season, p.week, p.home, p.away, p.kickoff_utc or None, "total",
                "over" if over else "under", f"{'Over' if over else 'Under'} {p.book_total:g}",
                p.book_total, money.odds, money.handle_pct, money.bets_pct, divergence,
                open_line, round(move, 1), reverse, round(score, 1), agrees, note)


def build(projections: list, splits: list, snapshots: list[dict], *, season: int,
          week: int) -> list[Spot]:
    opening = first_lines(snapshots, season, week)
    by_pair = {}
    for game in splits:
        home, away = resolve(game.home), resolve(game.away)
        if home and away:
            by_pair[(home, away)] = game
    spots: list[Spot] = []
    for p in projections:
        game = by_pair.get((p.home, p.away))
        if game is None:
            continue
        first = opening.get((p.home, p.away))
        for spot in (_spread(game, p, first), _total(game, p, first)):
            if spot is not None:
                spots.append(spot)
    spots.sort(key=lambda s: -s.concentration)
    return ([s for s in spots if s.family == "spread"][:LIMIT]
            + [s for s in spots if s.family == "total"][:LIMIT])
