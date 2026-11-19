"""Archive every published projection and track its first available wager.

The same stdlib-only collector is vendored in CBB, NBA, NFL and CFB. It never
changes a model or places a bet. Projection vintages and selected numbers are
immutable; only receipts, closing quotes and settlements can be appended.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import io
import json
import math
import re
import statistics
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

SCHEMA = "chase-projection-tracking/1"
ET = ZoneInfo("America/New_York")
LEAGUES = {
    "cbb": "basketball/mens-college-basketball",
    "nba": "basketball/nba",
    "nfl": "football/nfl",
    "cfb": "football/college-football",
}
CORE = {
    "cbb": "basketball/leagues/mens-college-basketball",
    "nba": "basketball/leagues/nba",
    "nfl": "football/leagues/nfl",
    "cfb": "football/leagues/college-football",
}
PROPS = {
    "total passing yards": "passing_yards",
    "total passing touchdowns": "passing_tds",
    "total pass completions": "completions",
    "total pass attempts": "pass_attempts",
    "total passing attempts": "pass_attempts",
    "total passing interceptions": "interceptions",
    "total rushing yards": "rushing_yards",
    "total carries": "rush_attempts",
    "total receiving yards": "receiving_yards",
    "total receptions": "receptions",
    "total field goals made": "fg_made",
    "total kicking points": "kicking_points",
    "total extra points made": "pat_made",
    "total points": "points",
    "total rebounds": "rebounds",
    "total assists": "assists",
    "total three point field goals made": "threes",
    "total three-pointers made": "threes",
    "total points + rebounds + assists": "pra",
    "total points, rebounds and assists": "pra",
}
ALIASES = {
    "pass_yds": "passing_yards",
    "pass_td": "passing_tds",
    "pass_att": "pass_attempts",
    "rush_yds": "rushing_yards",
    "rush_car": "rush_attempts",
    "carries": "rush_attempts",
    "rec": "receptions",
    "rec_yds": "receiving_yards",
    "pts": "points",
    "reb": "rebounds",
    "ast": "assists",
    "tpm": "threes",
    "pass_completions": "completions",
    "passing_interceptions": "interceptions",
    "fg_attempts": "fg_attempts",
}


def stamp(value=None):
    return (
        (value or datetime.now(timezone.utc)).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def moment(value):
    try:
        out = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return out if out.tzinfo else None
    except ValueError:
        return None


def number(value):
    try:
        out = float(str(value).lower().removeprefix("o").removeprefix("u").replace("−", "-"))
        return out if math.isfinite(out) else None
    except (ValueError, TypeError):
        return None


def norm(value):
    return re.sub(
        r"[^a-z0-9]",
        "",
        unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower(),
    )


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read(path):
    raw = Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if str(path).endswith(".gz") else raw)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    if str(path).endswith(".gz"):
        raw = gzip.compress(raw, mtime=0)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return hashlib.sha256(raw).hexdigest()


def request(url):
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0", "Cache-Control": "no-cache"}
    )
    with urllib.request.urlopen(req, timeout=25) as response:
        return response.read()


def recover(root, url):
    """Published manifests are durable storage, independent of the runner cache."""
    root = Path(root)
    local = read(root / "manifest.json") if (root / "manifest.json").exists() else None
    if not url:
        return local or {"schema": SCHEMA, "weeks": {}}
    try:
        remote = json.loads(request(url + ("&" if "?" in url else "?") + "tracking=" + stamp()))
    except urllib.error.HTTPError as exc:
        if exc.code == 404 and local is None:
            return {"schema": SCHEMA, "weeks": {}}
        raise RuntimeError(
            "Cannot recover published tracking; refusing to reset the record"
        ) from exc
    if remote.get("schema") != SCHEMA:
        raise ValueError("Unrecognized tracking manifest")
    for key, entry in remote["weeks"].items():
        target = root / entry["file"]
        if not re.fullmatch(r"[a-z0-9_-]+\.json\.gz", entry["file"]):
            raise ValueError("Unsafe archive path")
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == entry["sha256"]:
            continue
        raw = request(url.rsplit("/", 1)[0] + "/" + entry["file"] + "?sha=" + entry["sha256"])
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("Published tracking archive hash mismatch")
        incoming = json.loads(gzip.decompress(raw))
        if target.exists():
            current = read(target)
            for field in ("observations", "wagers"):
                incoming[field] = {**current.get(field, {}), **incoming.get(field, {})}
        write(target, incoming)
    if local:
        remote["weeks"] = {**local["weeks"], **remote["weeks"]}
    return remote


def identity(value):
    return (
        str(
            value.get("school")
            or value.get("abbreviation")
            or value.get("team")
            or value.get("name")
        )
        if isinstance(value, dict)
        else str(value or "")
    )


def week_id(sport, season, week, tip):
    if sport in ("nfl", "cfb"):
        if number(week) is None:
            raise ValueError("Football projections require an official week")
        return f"{sport}-{season}-week-{int(week):02d}"
    day = tip.astimezone(ET).date()
    return f"{sport}-{(day - timedelta(days=day.weekday())).isoformat()}"


def normalize_board(payload, sport):
    board = payload.get("board") or payload
    generated = board.get("generated_at_utc") or board.get("generated_at")
    if not moment(generated):
        raise ValueError("A dated, published model board is required")
    if not isinstance(board.get("games"), list):
        raise ValueError("Board has no game projections")
    season = board.get("season") or int(str(board.get("as_of_date", generated))[:4]) + (
        sport == "cbb"
    )
    games = []
    for raw in board["games"]:
        home, away = identity(raw.get("home")), identity(raw.get("away"))
        tip = moment(raw.get("kickoff_utc") or raw.get("kickoff"))
        if not tip or not home or not away:
            raise ValueError("Projection lacks a proven fixture identity or tip time")
        key = str(
            raw.get("id") or raw.get("game_id") or f"{season}|{board.get('week')}|{away}@{home}"
        )
        games.append(
            {
                "game_id": key,
                "home": home,
                "away": away,
                "tip": stamp(tip),
                "week_id": week_id(sport, season, raw.get("week", board.get("week")), tip),
                "season": season,
                "official_week": raw.get("week", board.get("week")),
                "home_aliases": sorted(
                    {
                        home,
                        identity(raw.get("home")),
                        str(raw.get("home_name") or home),
                        str((raw.get("home") or {}).get("abbreviation", home))
                        if isinstance(raw.get("home"), dict)
                        else home,
                    }
                ),
                "away_aliases": sorted(
                    {
                        away,
                        identity(raw.get("away")),
                        str(raw.get("away_name") or away),
                        str((raw.get("away") or {}).get("abbreviation", away))
                        if isinstance(raw.get("away"), dict)
                        else away,
                    }
                ),
                "model_margin": number(raw.get("model_margin")),
                "model_total": number(raw.get("independent_total", raw.get("projected_total"))),
                "home_probability": number(
                    raw.get("model_win_probability", raw.get("win_probability"))
                ),
                "raw": raw,
                "players": [],
            }
        )
        if sport == "cbb":
            games[-1]["model_total"] = number(raw.get("projected_total"))
    for player in board.get("player_projections") or []:
        gid = str(player.get("game_id") or player.get("event_id") or "")
        candidates = [g for g in games if g["game_id"] == gid]
        if not candidates:
            team, opponent = identity(player.get("team")), identity(player.get("opponent"))
            candidates = [
                g
                for g in games
                if {norm(g["home"]), norm(g["away"])} == {norm(team), norm(opponent)}
            ]
        if len(candidates) == 1:
            candidates[0]["players"].append(player)
    return board, games, generated


def _names(team):
    out = {
        norm(team.get(k, ""))
        for k in ("abbreviation", "displayName", "shortDisplayName", "location", "name")
    }
    aliases = {
        "wsh": "was",
        "gs": "gsw",
        "ny": "nyk",
        "no": "nop",
        "sa": "sas",
        "utah": "uta",
        "jax": "jac",
    }
    return {aliases.get(n, n) for n in out if n}


def match_event(game, events):
    direct = [e for e in events if str(e["id"]) == game["game_id"]]
    if direct:
        return direct[0]
    matches = []
    for event in events:
        comp = event.get("competitions", [{}])[0]
        sides = {c.get("homeAway"): c.get("team", {}) for c in comp.get("competitors", [])}
        tip = moment(event.get("date"))
        if not tip or abs((tip - moment(game["tip"])).total_seconds()) > 21600:
            continue
        if all(
            any(norm(n) in _names(sides.get(s, {})) for n in game[s + "_aliases"])
            for s in ("home", "away")
        ):
            matches.append(event)
    return matches[0] if len(matches) == 1 else None


def _path(value, *keys):
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
    return value


def game_quotes(item, *, phase="current", core=False):
    """Selected-side line and actual price. Missing prices remain missing."""
    if norm(_path(item, "provider", "name")) != "draftkings":
        return {}
    out = {}
    for side in ("home", "away"):
        line = number(
            _path(item, side + "TeamOdds", phase, "pointSpread", "american")
            if core
            else _path(item, "pointSpread", side, "close" if phase == "current" else phase, "line")
        )
        if line is None and side == "home" and phase == "current":
            line = number(item.get("spread"))
        price = number(
            _path(item, side + "TeamOdds", phase, "spread", "american")
            if core
            else _path(item, "pointSpread", side, "close" if phase == "current" else phase, "odds")
        )
        if line is not None:
            out["spread:" + side] = {"line": line, "price": price}
        price = number(
            _path(item, side + "TeamOdds", phase, "moneyLine", "american")
            if core
            else _path(item, "moneyline", side, "close" if phase == "current" else phase, "odds")
        )
        if price is None and core and phase == "current":
            price = number(_path(item, side + "TeamOdds", "moneyLine"))
        if price is not None:
            out["moneyline:" + side] = {"line": None, "price": price}
    if "spread:home" in out and "spread:away" not in out:
        out["spread:away"] = {"line": -out["spread:home"]["line"], "price": None}
    for side in ("over", "under"):
        line = number(
            _path(item, phase, "total", "american")
            if core
            else _path(item, "total", side, "close" if phase == "current" else phase, "line")
        )
        if line is None and phase == "current":
            line = number(item.get("overUnder"))
        price = number(
            _path(item, phase, side, "american")
            if core
            else _path(item, "total", side, "close" if phase == "current" else phase, "odds")
        )
        if line is not None:
            out["total:" + side] = {"line": line, "price": price}
    return out


def implied(price):
    price = number(price)
    if price is None or abs(price) < 100:
        return None
    return 100 / (price + 100) if price > 0 else -price / (-price + 100)


def clv(wager, closing):
    quote = closing.get(wager["market"] + ":" + wager["side"])
    if not quote:
        return None
    if wager["market"] == "moneyline":
        entry, end = implied(wager.get("price")), implied(quote.get("price"))
        return None if entry is None or end is None else round(100 * (end - entry), 4)
    entry, end = number(wager.get("line")), number(quote.get("line"))
    if entry is None or end is None:
        return None
    return round(end - entry if wager["side"] == "over" else entry - end, 4)


def settle(wager, margin=None, total=None, actual=None, dnp=False, canceled=False):
    if wager.get("result"):
        return
    if dnp or canceled:
        outcome = "void"
    else:
        if wager["market"] == "spread":
            value = margin if wager["side"] == "home" else -margin
            value += wager["line"]
        elif wager["market"] == "moneyline":
            value = margin if wager["side"] == "home" else -margin
        elif wager["market"] == "total":
            value = total - wager["line"] if wager["side"] == "over" else wager["line"] - total
        else:
            if actual is None:
                return
            value = actual - wager["line"] if wager["side"] == "over" else wager["line"] - actual
        outcome = "push" if abs(value) < 1e-9 else "win" if value > 0 else "loss"
    price = number(wager.get("price"))
    units = None
    if outcome in ("void", "push"):
        units = 0.0
    elif price is not None and abs(price) >= 100:
        units = -1.0 if outcome == "loss" else price / 100 if price > 0 else 100 / -price
    wager.update(
        result=outcome, graded_at=stamp(), units=None if units is None else round(units, 6)
    )


def stats(wagers):
    rows = list(wagers)
    out = {r: sum(w.get("result") == r for w in rows) for r in ("win", "loss", "push", "void")}
    out["pending"] = sum(not w.get("result") for w in rows)
    priced = [
        w for w in rows if w.get("result") in ("win", "loss", "push") and w.get("units") is not None
    ]
    out["priced_settled"] = len(priced)
    out["settled_without_price"] = sum(
        w.get("result") in ("win", "loss") and w.get("units") is None for w in rows
    )
    out["profit_units"] = round(sum(w["units"] for w in priced), 4)
    out["roi"] = out["profit_units"] / len(priced) if priced else None
    out["clv"] = {}

    def group(w):
        return "prop:" + w.get("metric", "unknown") if w["market"] == "prop" else w["market"]

    for market in sorted({group(w) for w in rows}):
        values = [w["clv"] for w in rows if group(w) == market and w.get("clv") is not None]
        out["clv"][market] = {
            "n": len(values),
            "mean": statistics.fmean(values) if values else None,
            "positive_rate": sum(v > 0 for v in values) / len(values) if values else None,
            "unit": "probability points"
            if market == "moneyline"
            else market.split(":", 1)[1]
            if market.startswith("prop:")
            else "points",
        }
    return out


def player_metrics(player):
    out = {ALIASES.get(k, k): number(v) for k, v in (player.get("metrics") or {}).items()}
    for key, dist in (player.get("stats") or {}).items():
        if isinstance(dist, dict) and number(dist.get("mean")) is not None:
            out[ALIASES.get(key, key)] = number(dist["mean"])
    if number(player.get("anytime_td")) is not None:
        out["anytime_td_probability"] = number(player["anytime_td"])
    return {k: v for k, v in out.items() if v is not None}


def parse_players(summary):
    """Observed box metrics, keyed by provider athlete id; absent is not zero."""
    out = {}
    for team in _path(summary, "boxscore", "players") or []:
        for group in team.get("statistics") or []:
            labels = group.get("labels") or group.get("names") or []
            family = norm(group.get("name", ""))
            for row in group.get("athletes") or []:
                athlete = row.get("athlete") or {}
                pid = str(athlete.get("id") or "")
                if not pid:
                    continue
                rec = out.setdefault(
                    pid,
                    {
                        "name": athlete.get("displayName", ""),
                        "team": team.get("team", {}),
                        "metrics": {},
                        "played": False,
                    },
                )
                if row.get("didNotPlay"):
                    continue
                rec["played"] = True
                data = dict(zip(labels, row.get("stats") or []))
                metrics = rec["metrics"]
                for key, value in data.items():
                    k, val = norm(key), number(value)
                    if val is not None:
                        plain = {
                            "pts": "points",
                            "reb": "rebounds",
                            "ast": "assists",
                            "3pt": "threes",
                            "min": "minutes",
                            "stl": "steals",
                            "blk": "blocks",
                            "to": "turnovers",
                        }.get(k)
                        if plain:
                            metrics[plain] = val
                    if family == "passing":
                        if k in ("catt", "compatt", "cpatt") and "/" in str(value):
                            a, b = str(value).split("/", 1)
                            metrics.update(completions=number(a), pass_attempts=number(b))
                        field = {
                            "yds": "passing_yards",
                            "td": "passing_tds",
                            "int": "interceptions",
                        }.get(k)
                        if field and val is not None:
                            metrics[field] = val
                    if family in ("rushing", "receiving"):
                        field = (
                            {"car": "rush_attempts", "yds": "rushing_yards", "td": "rushing_tds"}
                            if family == "rushing"
                            else {
                                "rec": "receptions",
                                "yds": "receiving_yards",
                                "td": "receiving_tds",
                            }
                        ).get(k)
                        if field and val is not None:
                            metrics[field] = val
                    if k in ("3pt", "3pm3pa") and "-" in str(value):
                        metrics["threes"] = number(str(value).split("-")[0])
                    if family in ("kicking", "fieldgoals") and k in ("fg", "xp", "xpt", "pat"):
                        pair = re.split(r"[-/]", str(value))
                        if len(pair) == 2:
                            if k == "fg":
                                metrics.update(fg_made=number(pair[0]), fg_attempts=number(pair[1]))
                            else:
                                metrics["pat_made"] = number(pair[0])
    for rec in out.values():
        m = rec["metrics"]
        if all(k in m for k in ("points", "rebounds", "assists")):
            m["pra"] = m["points"] + m["rebounds"] + m["assists"]
        if "rushing_tds" in m or "receiving_tds" in m:
            m["anytime_td_probability"] = float(
                m.get("rushing_tds", 0) + m.get("receiving_tds", 0) > 0
            )
        if "fg_made" in m and "pat_made" in m:
            m["kicking_points"] = 3 * m["fg_made"] + m["pat_made"]
    return out


def athlete_identities(summary, actuals):
    """Pregame summaries can name athletes before any statistics exist."""
    out = dict(actuals)

    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            athlete = value.get("athlete")
            if isinstance(athlete, dict) and athlete.get("id") and athlete.get("displayName"):
                out.setdefault(
                    str(athlete["id"]),
                    {"name": athlete["displayName"], "played": False, "metrics": {}},
                )
            for item in value.values():
                if isinstance(item, (list, dict)):
                    visit(item)

    visit(summary)
    return out


def match_player(player, actuals):
    pid = str(player.get("player_id") or "")
    if pid in actuals:
        return pid
    name = norm(player.get("player_name") or player.get("name") or "")
    found = [k for k, v in actuals.items() if name and norm(v["name"]) == name]
    return found[0] if len(found) == 1 else None


def prop_quotes(payload):
    out = {}
    for item in payload.get("items") or []:
        name = re.sub(r"\s*\(.*?\)", "", str(_path(item, "type", "name") or "")).strip().lower()
        metric = PROPS.get(name)
        ref = str(_path(item, "athlete", "$ref") or "")
        match = re.search(r"/athletes/(\d+)", ref)
        line = number(_path(item, "current", "target", "value"))
        if metric and match and line is not None:
            out[(match[1], metric)] = {
                "line": line,
                "price": None,
                "updated_at": item.get("lastUpdated"),
            }
    return out


class Provider:
    def __init__(self, sport, root, now):
        self.sport, self.root, self.now = sport, Path(root), now
        self.days = {}
        self.issues = []
        self.receipts = {}

    def fetch(self, url):
        try:
            raw = request(url)
            value = json.loads(raw)
            sha = hashlib.sha256(raw).hexdigest()
            self.receipts[sha] = {"url": url, "retrieved_at": stamp(self.now), "sha256": sha}
            return value
        except (OSError, ValueError) as exc:
            self.issues.append(
                {"url": url, "error": type(exc).__name__, "status": getattr(exc, "code", None)}
            )
            return {}

    def events(self, tip):
        key = moment(tip).astimezone(ET).strftime("%Y%m%d")
        if key not in self.days:
            url = f"https://site.api.espn.com/apis/site/v2/sports/{LEAGUES[self.sport]}/scoreboard?dates={key}&limit=1000"
            if self.sport in ("cbb", "cfb"):
                url += "&groups=" + ("50" if self.sport == "cbb" else "80")
            self.days[key] = self.fetch(url).get("events", [])
        return self.days[key]

    def collect(self, game):
        event = match_event(game, self.events(game["tip"]))
        if not event:
            return {}
        event_id = str(event["id"])
        comp = event.get("competitions", [{}])[0]
        state = _path(comp, "status", "type") or _path(event, "status", "type") or {}
        final = state.get("completed") is True and state.get("name") in (
            "STATUS_FINAL",
            "STATUS_FULL_TIME",
            "STATUS_FINAL_OVERTIME",
        )
        canceled = state.get("name") in ("STATUS_CANCELED", "STATUS_CANCELLED")
        pre = state.get("state") == "pre" and moment(event.get("date")) > self.now
        item = next(
            (q for q in comp.get("odds", []) if norm(_path(q, "provider", "name")) == "draftkings"),
            {},
        )
        live = game_quotes(item) if pre else {}
        # Current in-game odds are never treated as pregame closing quotes.
        closing, summary, actual_players, props = {}, {}, {}, {}
        base = f"https://sports.core.api.espn.com/v2/sports/{CORE[self.sport]}/events/{event_id}/competitions/{event_id}/odds/100"
        if final:
            summary = self.fetch(
                f"https://site.api.espn.com/apis/site/v2/sports/{LEAGUES[self.sport]}/summary?event={event_id}"  # noqa: E501 - prose or generated HTML
            )
            actual_players = parse_players(summary)
            core = self.fetch(base)
            closing = game_quotes(core, phase="close", core=True)
        elif pre and game.get("players"):
            props = prop_quotes(self.fetch(base + "/propBets?limit=1000"))
            # Athlete identities are required for props. The pregame summary
            # carries athlete ids and names even before box statistics exist.
            summary = self.fetch(
                f"https://site.api.espn.com/apis/site/v2/sports/{LEAGUES[self.sport]}/summary?event={event_id}"  # noqa: E501 - prose or generated HTML
            )
            actual_players = athlete_identities(summary, parse_players(summary))
        sides = {c.get("homeAway"): c for c in comp.get("competitors", [])}
        scores = {s: number(sides.get(s, {}).get("score")) for s in ("home", "away")}
        return {
            "event_id": event_id,
            "pre": pre,
            "final": final,
            "canceled": canceled,
            "tip": event.get("date"),
            "quotes": live,
            "closing": closing,
            "players": actual_players,
            "prop_quotes": props,
            "scores": scores,
            "retrieved_at": stamp(self.now),
            "source": "ESPN / DraftKings",
        }


def _forecast_value(game):
    return {
        k: game[k]
        for k in (
            "game_id",
            "home",
            "away",
            "tip",
            "week_id",
            "season",
            "official_week",
            "home_aliases",
            "away_aliases",
            "model_margin",
            "model_total",
            "home_probability",
            "raw",
            "players",
        )
    }


def archive_projection(week, game, generated, version, now):
    raw = _forecast_value(game)
    key = digest({"generated": generated, "version": version, "projection": raw})
    week["observations"].setdefault(
        key,
        {
            "id": key,
            "recorded_at": stamp(now),
            "model_generated_at": generated,
            "model_version": version,
            "projection": raw,
            "evaluation_eligible": moment(game["tip"]) > now
            and moment(generated) < moment(game["tip"]),
            "status": "pending" if moment(game["tip"]) > now else "ineligible_after_start",
        },
    )


def _choose(game, quotes):
    picks = []
    for market, mean in (("spread", game["model_margin"]), ("total", game["model_total"])):
        if mean is None:
            continue
        side = "home" if market == "spread" else "over"
        quote = quotes.get(market + ":" + side)
        if not quote:
            continue
        gap = mean + quote["line"] if market == "spread" else mean - quote["line"]
        if gap == 0:
            continue
        if gap < 0:
            side = "away" if market == "spread" else "under"
        chosen = quotes.get(market + ":" + side)
        if chosen:
            picks.append(
                {
                    "market": market,
                    "side": side,
                    **chosen,
                    "model_number": mean,
                    "rank_score": abs(gap) / max(abs(mean), abs(quote["line"]), 1.0),
                    "ranking_basis": "relative model disagreement",
                }
            )
    p = game.get("home_probability")
    if p is not None and 0 < p < 1:
        candidates = []
        for side, chance in (("home", p), ("away", 1 - p)):
            quote = quotes.get("moneyline:" + side)
            fair = implied((quote or {}).get("price"))
            if fair:
                candidates.append(
                    {
                        "market": "moneyline",
                        "side": side,
                        **quote,
                        "probability": chance,
                        "model_number": chance,
                        "rank_score": (chance - fair) / fair,
                        "ranking_basis": "relative model probability disagreement",
                    }
                )
        if candidates:
            picks.append(max(candidates, key=lambda p: p["rank_score"]))
    if p is None and game.get("model_margin") is not None and game["model_margin"] != 0:
        side = "home" if game["model_margin"] > 0 else "away"
        quote = quotes.get("moneyline:" + side)
        if implied((quote or {}).get("price")):
            picks.append(
                {
                    "market": "moneyline",
                    "side": side,
                    **quote,
                    "model_number": game["model_margin"],
                    "rank_score": None,
                    "ranking_basis": "model winner; no published win probability",
                }
            )
    return picks


def update_week(week, games, generated, version, now, provider, *, collect_only=False):
    for game in games:
        if not collect_only:
            archive_projection(week, game, generated, version, now)
    by_game = {g["game_id"]: g for g in games}
    generations = {g["game_id"]: (generated, version) for g in games}
    for observation in week["observations"].values():
        if observation.get("status") in ("pending", "awaiting_player_results"):
            by_game.setdefault(observation["projection"]["game_id"], observation["projection"])
            generations.setdefault(
                observation["projection"]["game_id"],
                (observation["model_generated_at"], observation["model_version"]),
            )
    for wager in week["wagers"].values():
        retry_close = (
            wager["market"] != "prop"
            and wager.get("clv") is None
            and now - moment(wager["tip"]) < timedelta(days=2)
        )
        if not wager.get("result") or retry_close:
            obs = next(
                (
                    o
                    for o in reversed(list(week["observations"].values()))
                    if o["projection"]["game_id"] == wager["game_id"]
                ),
                None,
            )
            if obs:
                by_game.setdefault(wager["game_id"], obs["projection"])
                generations.setdefault(
                    wager["game_id"], (obs["model_generated_at"], obs["model_version"])
                )
    new_picks = []
    for gid, game in by_game.items():
        collected = provider.collect(game)
        if not collected:
            continue
        generation, model_version = generations.get(gid, (generated, version))
        if collected.get("canceled"):
            for obs in week["observations"].values():
                if obs["projection"]["game_id"] == gid and obs.get("status") in (
                    "pending",
                    "awaiting_player_results",
                ):
                    obs.update(status="void", graded_at=stamp(now))
        # Only forecasts created before the tip can produce tracked wagers.
        if collected.get("pre") and moment(game["tip"]) > now:
            picks = _choose(game, collected["quotes"])
            for player in game.get("players", []):
                pid = match_player(player, collected.get("players", {})) or str(
                    player.get("player_id") or ""
                )
                for metric, mean in player_metrics(player).items():
                    quote = collected.get("prop_quotes", {}).get((pid, metric))
                    if not quote or mean == quote["line"]:
                        continue
                    # These are observed lines; ESPN does not publish prop
                    # prices. They must never silently become -110 ROI bets.
                    picks.append(
                        {
                            "market": "prop",
                            "metric": metric,
                            "player_id": str(player.get("player_id") or pid),
                            "provider_player_id": pid,
                            "player_name": player.get("player_name"),
                            "side": "over" if mean > quote["line"] else "under",
                            **quote,
                            "model_number": mean,
                            "rank_score": abs(mean - quote["line"])
                            / max(abs(mean), abs(quote["line"]), 1.0),
                            "ranking_basis": "relative model disagreement",
                        }
                    )
            for pick in picks:
                key = digest(
                    [
                        week["week_id"],
                        gid,
                        pick["market"],
                        pick.get("player_id"),
                        pick.get("metric"),
                        "DraftKings",
                    ]
                )
                if key not in week["wagers"]:
                    entry = {
                        "id": key,
                        "game_id": gid,
                        "event_id": collected["event_id"],
                        "home": game["home"],
                        "away": game["away"],
                        "tip": game["tip"],
                        "week_id": week["week_id"],
                        "book": "DraftKings",
                        "recorded_at": stamp(now),
                        "model_generated_at": generation,
                        "model_version": model_version,
                        "source": collected["source"],
                        "stake_units": 1,
                        "result": None,
                        "clv": None,
                        "clv_status": "pending_close",
                        "top20": False,
                        **pick,
                    }
                    week["wagers"][key] = entry
                    new_picks.append(entry)
        for wager in week["wagers"].values():
            if wager["game_id"] != gid:
                continue
            if collected.get("pre"):
                quote = collected["quotes"].get(wager["market"] + ":" + wager["side"])
                if wager["market"] == "prop":
                    quote = collected.get("prop_quotes", {}).get(
                        (wager.get("provider_player_id"), wager["metric"])
                    )
                if quote:
                    history = wager.setdefault("quote_history", [])
                    prior = history[-1] if history else None
                    if not prior or (prior.get("line"), prior.get("price")) != (
                        quote.get("line"),
                        quote.get("price"),
                    ):
                        history.append(
                            {
                                **quote,
                                "first_observed_at": stamp(now),
                                "last_observed_at": stamp(now),
                            }
                        )
                    else:
                        prior["last_observed_at"] = stamp(now)
                    wager["last_prestart_quote"] = {**quote, "observed_at": stamp(now)}
            if collected.get("canceled"):
                settle(wager, canceled=True)
                for obs in week["observations"].values():
                    if obs["projection"]["game_id"] == gid and obs.get("status") == "pending":
                        obs.update(status="void", graded_at=stamp(now))
            if collected.get("final"):
                home, away = collected["scores"]["home"], collected["scores"]["away"]
                if home is not None and away is not None:
                    if wager["market"] != "prop":
                        settle(wager, margin=home - away, total=home + away)
                    else:
                        player = collected["players"].get(wager.get("provider_player_id"))
                        if player:
                            settle(
                                wager,
                                actual=player["metrics"].get(wager["metric"]),
                                dnp=not player["played"],
                            )
                close = collected["closing"]
                if wager.get("clv") is None:
                    value = clv(wager, close)
                    if value is not None:
                        wager.update(
                            clv=value,
                            clv_status="provider_close",
                            closing_quote=close.get(wager["market"] + ":" + wager["side"]),
                            closing_source=collected["source"],
                            closing_retrieved_at=stamp(now),
                        )
                    elif wager.get("last_prestart_quote"):
                        quote = wager["last_prestart_quote"]
                        age = (
                            moment(collected["tip"]) - moment(quote["observed_at"])
                        ).total_seconds()
                        if 0 <= age <= 3600:
                            value = clv(wager, {wager["market"] + ":" + wager["side"]: quote})
                            wager.update(
                                clv=value,
                                clv_status="last_observed_prestart",
                                closing_quote=quote,
                                closing_age_seconds=age,
                            )
                        else:
                            wager["clv_status"] = "missing_close"
                    else:
                        wager["clv_status"] = "missing_close"
        if collected.get("final"):
            home, away = collected["scores"]["home"], collected["scores"]["away"]
            for obs in week["observations"].values():
                if (
                    obs["projection"]["game_id"] != gid
                    or obs.get("status") not in ("pending", "awaiting_player_results")
                    or not obs.get("evaluation_eligible")
                    or home is None
                    or away is None
                ):
                    continue
                p = obs["projection"]
                accuracy = {}
                for field, actual in (("model_margin", home - away), ("model_total", home + away)):
                    if p.get(field) is not None:
                        accuracy[field] = abs(p[field] - actual)
                player_results = []
                for player in p.get("players", []):
                    pid = match_player(player, collected["players"])
                    actual = collected["players"].get(pid)
                    if actual:
                        errors = {
                            k: abs(v - actual["metrics"][k])
                            for k, v in player_metrics(player).items()
                            if actual["metrics"].get(k) is not None
                        }
                        player_results.append(
                            {
                                "player_id": player.get("player_id"),
                                "played": actual["played"],
                                "actual": actual["metrics"],
                                "absolute_errors": errors,
                            }
                        )
                missing_players = len(p.get("players", [])) - len(player_results)
                waiting = missing_players > 0 and now - moment(game["tip"]) < timedelta(days=7)
                obs.update(
                    status="awaiting_player_results" if waiting else "graded",
                    missing_player_results=missing_players,
                    result={"home": home, "away": away},
                    absolute_errors=accuracy,
                    player_results=player_results,
                    graded_at=stamp(now),
                )
    # Membership is assigned before outcomes, and is never retrospectively
    # replaced. Later opportunities fill any remaining slots; all others stay
    # in the uncapped record. Ranking is a transparent relative disagreement,
    # not a new claim that a model has demonstrated a profitable edge.
    slots = max(0, 20 - sum(w.get("top20", False) for w in week["wagers"].values()))
    ranked = sorted(
        (w for w in new_picks if w.get("rank_score") is not None),
        key=lambda w: (-w["rank_score"], w["id"]),
    )
    for wager in ranked[:slots]:
        wager["top20"] = True
        wager["top20_locked_at"] = stamp(now)
    week["updated_at"] = stamp(now)
    week.setdefault("receipts", {}).update(getattr(provider, "receipts", {}))


POLICY = (
    "Every published game and player projection is archived, including unpriced projections. "
    "A tracked wager is a paper selection at its first observed DraftKings number; side, line and price stay frozen. "  # noqa: E501 - prose or generated HTML
    "Official football weeks apply to NFL/CFB; NBA/CBB use Monday–Sunday Eastern. "
    "The weekly featured cohort has at most 20 entries, locked before kickoff; later candidates fill empty slots. "  # noqa: E501 - prose or generated HTML
    "All remaining selections stay in the full record. Ranking uses absolute model-minus-line disagreement "  # noqa: E501 - prose or generated HTML
    "divided by the larger of the model number, line magnitude, and one; moneylines use model probability "  # noqa: E501 - prose or generated HTML
    "relative to price break-even. This is a tracking rank, not a calibrated edge. "
    "CLV uses a named provider close, or an explicitly labeled last observed prestart quote within 60 minutes. "  # noqa: E501 - prose or generated HTML
    "Missing close and missing prices stay missing. Moneyline CLV is in implied-probability points; "  # noqa: E501 - prose or generated HTML
    "line-market CLV is in points from the selected side. Returns risk one unit and use the actual observed price. "  # noqa: E501 - prose or generated HTML
    "No history is truncated, no historical selections are manufactured, and no real wagers are placed."  # noqa: E501 - prose or generated HTML
)


def render_report(root, manifest, weeks):
    def esc(value):
        return html.escape(str(value))

    sections = []
    for key in sorted(weeks, reverse=True):
        week = weeks[key]
        wagers = list(week["wagers"].values())
        record = stats(wagers)
        top = sorted(
            (w for w in wagers if w.get("top20")), key=lambda w: (-w.get("rank_score", 0), w["id"])
        )
        rows = []
        for w in top:
            selection = (w.get("player_name") + " " if w.get("player_name") else "") + w["side"]
            line = "—" if w.get("line") is None else f"{w['line']:g}"
            price = "missing" if w.get("price") is None else f"{w['price']:+g}"
            closing = "—" if w.get("clv") is None else f"{w['clv']:+g}"
            rows.append(
                "<tr>"
                + "".join(
                    "<td>" + esc(v) + "</td>"
                    for v in (
                        w["away"] + " at " + w["home"],
                        w["market"],
                        selection,
                        line,
                        price,
                        w.get("result") or "pending",
                        closing,
                        {
                            "provider_close": "Provider close",
                            "last_observed_prestart": "Observed before start",
                            "missing_close": "Missing close",
                            "pending_close": "Waiting for close",
                        }.get(w["clv_status"], w["clv_status"]),
                        w["recorded_at"],
                    )
                )
                + "</tr>"
            )
        links = f'<a href="{esc(key)}.csv">Every selection (CSV)</a> · <a href="{esc(key)}.json.gz">Every projection and receipt</a>'  # noqa: E501 - generated HTML
        clv_bits = []
        for market, metric in record["clv"].items():
            avg = "missing" if metric["mean"] is None else f"{metric['mean']:+.2f} {metric['unit']}"
            clv_bits.append(f"{market}: {avg} ({metric['n']} closes)")
        sections.append(
            f"<section><h2>{esc(key)}</h2><p>{len(week['observations']):,} projection vintages · "
            f"{sum(len(o['projection'].get('players', [])) for o in week['observations'].values()):,} player projections · "  # noqa: E501 - prose or generated HTML
            f"{len(wagers):,} tracked selections · {record['win']}–{record['loss']}–{record['push']} "  # noqa: E501 - prose or generated HTML
            f"W–L–P · {record['pending']} pending · {record['void']} void</p>"
            f"<p>{record['profit_units']:+.2f}u on {record['priced_settled']} settled, priced selections. "  # noqa: E501 - prose or generated HTML
            f"{record['settled_without_price']} settled selections have no observed price and are excluded from returns.</p>"  # noqa: E501 - prose or generated HTML
            f"<p>{esc(' · '.join(clv_bits))}</p><p>{links}</p><h3>Weekly top 20 ({len(top)}/20)</h3>"  # noqa: E501 - prose or generated HTML
            '<div class="scroll" tabindex="0" role="region" aria-label="Weekly selections"><table><thead><tr><th>Game</th><th>Market</th><th>Selection</th>'  # noqa: E501 - generated HTML or prose
            "<th>Entry line</th><th>Entry price</th><th>Result</th><th>CLV</th><th>Close quality</th><th>Locked UTC</th>"  # noqa: E501 - prose or generated HTML
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div></section>"
        )
        fields = [
            "id",
            "week_id",
            "recorded_at",
            "model_generated_at",
            "model_version",
            "away",
            "home",
            "tip",
            "market",
            "player_name",
            "metric",
            "side",
            "line",
            "price",
            "book",
            "result",
            "units",
            "clv",
            "clv_status",
            "top20",
        ]
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(wagers)
        (Path(root) / (key + ".csv")).write_text(stream.getvalue(), encoding="utf-8")
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'  # noqa: E501 - prose or generated HTML
        "<title>Projection and wager tracking · Chase Analytics</title><style>"
        ":root{color-scheme:dark}body{background:#08090f;color:#f5f5f7;font:16px/1.55 system-ui;margin:0;padding:24px}"  # noqa: E501 - prose or generated HTML
        "main{max-width:1200px;margin:auto}a{color:#9a6bff}section{border-top:1px solid #38383f;padding:24px 0}"  # noqa: E501 - prose or generated HTML
        "h1{font-size:2rem}h2{font-size:1.35rem}table{border-collapse:collapse;width:100%;font-size:.875rem}"  # noqa: E501 - prose or generated HTML
        "td,th{text-align:left;padding:12px;border-bottom:1px solid #38383f;white-space:nowrap}.scroll{overflow:auto}"  # noqa: E501 - prose or generated HTML
        'p{max-width:100ch}small{color:#b6b6c2}</style><main><a href="../">Model dashboard</a>'
        "<h1>Projection and wager tracking</h1><p>All projections retained. First observed selections frozen. Results and CLV collected consistently.</p>"  # noqa: E501 - prose or generated HTML
        f"<p>Updated {esc(manifest['updated_at'])} · {esc(manifest['sport'].upper())}</p><p>{esc(POLICY)}</p>"  # noqa: E501 - prose or generated HTML
        + "".join(sections)
        + "</main></html>"
    )
    (Path(root) / "index.html").write_text(page, encoding="utf-8")


def run(board, sport, root, *, recover_url=None, now=None, provider=None, collect_only=False):
    now = now or datetime.now(timezone.utc)
    root = Path(root)
    manifest = recover(root, recover_url)
    normalized, games, generated = normalize_board(board, sport)
    if moment(generated) > now + timedelta(minutes=5):
        raise ValueError("Model board timestamp is in the future")
    if manifest.get("sport") not in (None, sport):
        raise ValueError("Tracking archive belongs to another sport")
    weeks = {key: read(root / entry["file"]) for key, entry in manifest.get("weeks", {}).items()}
    groups = {}
    for game in games:
        groups.setdefault(game["week_id"], []).append(game)
    for key in groups:
        weeks.setdefault(
            key,
            {
                "schema": SCHEMA,
                "sport": sport,
                "week_id": key,
                "observations": {},
                "wagers": {},
                "receipts": {},
            },
        )
    provider = provider or Provider(sport, root, now)
    version = (
        normalized.get("model_version")
        or normalized.get("method")
        or normalized.get("schema")
        or normalized.get("schema_version")
        or "unversioned"
    )
    if normalized.get("producer_revision"):
        version = str(version) + "@" + str(normalized["producer_revision"])[:12]
    for key, week in weeks.items():
        update_week(
            week, groups.get(key, []), generated, version, now, provider, collect_only=collect_only
        )
        sha = write(root / (key + ".json.gz"), week)
        manifest.setdefault("weeks", {})[key] = {
            "file": key + ".json.gz",
            "sha256": sha,
            "observations": len(week["observations"]),
            "record": stats(week["wagers"].values()),
            "top20_record": stats(w for w in week["wagers"].values() if w.get("top20")),
        }
    manifest.update(
        schema=SCHEMA,
        sport=sport,
        updated_at=stamp(now),
        policy=POLICY,
        issues=getattr(provider, "issues", []),
    )
    root.mkdir(parents=True, exist_ok=True)
    write(root / "manifest.json", manifest)
    render_report(root, manifest, weeks)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sport", choices=sorted(LEAGUES), required=True)
    parser.add_argument("--board", required=True, help="Local board JSON or its public HTTPS URL")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--recover-url")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--producer-revision", help="Exact source commit of a new model build")
    args = parser.parse_args()
    board = (
        json.loads(request(args.board)) if args.board.startswith("https://") else read(args.board)
    )
    if args.producer_revision:
        if args.collect_only or not re.fullmatch(r"[a-f0-9]{40}", args.producer_revision):
            raise ValueError("A source revision requires a full build and a 40-character commit")
        (board.get("board") or board)["producer_revision"] = args.producer_revision
        if not args.board.startswith("https://"):
            write(Path(args.board), board)
    result = run(
        board, args.sport, args.out, recover_url=args.recover_url, collect_only=args.collect_only
    )
    home = args.out.parent / "index.html"
    if home.exists():
        page = home.read_text(encoding="utf-8")
        if 'href="tracking/"' not in page:
            link = (
                '<p><a href="tracking/">All projections, weekly selections, record and CLV</a></p>'
            )
            page = (
                page.replace("</main>", link + "</main>", 1)
                if "</main>" in page
                else page.replace("</body>", link + "</body>", 1)
            )
            home.write_text(page, encoding="utf-8")
    print(
        json.dumps(
            {
                "sport": args.sport,
                "weeks": len(result["weeks"]),
                "projection_vintages": sum(w["observations"] for w in result["weeks"].values()),
                "provider_issues": len(result["issues"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
