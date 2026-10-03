"""Refit the prop price calibration (src/nflmodel/prop_pricing.py).

Joins graded player projections to the DraftKings line each faced: the line
stored on the ledger row when there is one, otherwise the closing line ESPN
still publishes for the finished game. Every quote gets ``raw`` = P(over)
under the projection's fitted outcome distribution, then:

* s, c - the calibration p = 0.5 + s * (raw - c), chosen to minimise Brier
  score, fitted leave-one-week-out so every week is scored by a model that
  never saw it;
* w - the per-family blend weight of the fair number (least squares).

The evidence `prop_pricing.research_only` reads is the held-out record of the
plays priced >= MIN_PROBABILITY, plus a fit-free check: within each week and
market, the top third of raw prices taken over and the bottom third under.

    PYTHONPATH=src python scripts/fit_prop_pricing.py [ledger.json] [--since 2.1.0]
    PYTHONPATH=src python scripts/fit_prop_pricing.py --rows joined.json

``--since`` keeps ledger rows from that projection version on (refit on the
current model once it has a few weeks of lines). ``--rows`` takes a list of
{week, metric, proj, line, act} dicts, e.g. from the point-in-time replay.
Pure standard library, like the package.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nflmodel import prop_pricing  # noqa: E402
from nflmodel.best_bets import PROP_MARKETS  # noqa: E402
from nflmodel.sources import espn_odds, espn_props, nflverse  # noqa: E402

LEDGER_URL = "https://alphakiller1.github.io/nfl-model/ledger.json.gz"
SCOREBOARD = espn_odds.SCOREBOARD + "?seasontype=2&week={week}&dates={season}"
REPORT = ROOT / "reports" / "prop_pricing_fit.json"


def load_ledger(path: str | None) -> dict:
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    import gzip
    with urllib.request.urlopen(LEDGER_URL, timeout=60) as resp:
        return json.loads(gzip.decompress(resp.read()).decode("utf-8"))


def espn_lines(season: int, week: int) -> dict[tuple[str, str], float]:
    """(gsis id, metric) -> closing line for one finished week."""
    roster = nflverse.weekly_roster(season, week=week)
    players = espn_props.player_index(roster)
    board = espn_odds._get(SCOREBOARD.format(week=week, season=season))
    out = {}
    for event in board.get("events") or []:
        payload = espn_odds._get(espn_props.PROP_BETS.format(event=event["id"]))
        for quote in espn_props.parse(payload, event_id=str(event["id"]), home="", away="",
                                      players=players):
            spec = PROP_MARKETS.get(quote.market)
            if spec:
                out[(quote.player_id, spec[0])] = quote.line
                if spec[0] == "rush_attempts":  # backs project "carries"
                    out[(quote.player_id, "carries")] = quote.line
    return out


def joined_rows(ledger: dict, since: str | None = None) -> list[dict]:
    graded = [r for r in ledger.get("player_snapshots", []) if r.get("status") == "graded"]
    if since:
        graded = [r for r in graded
                  if str(r.get("model_version", "")).split("/")[-1] >= since]
    fetched: dict[tuple[int, int], dict] = {}
    rows = []
    for row in graded:
        key = (int(row["season"]), int(row["week"]))
        stored = row.get("prop_lines") or {}
        for metric, projection in (row.get("metrics") or {}).items():
            actual = (row.get("actual_metrics") or {}).get(metric)
            if prop_pricing.FAMILY.get(metric) is None or actual is None:
                continue
            line = (stored.get(metric) or {}).get("line")
            if line is None:
                if key not in fetched:
                    try:
                        fetched[key] = espn_lines(*key)
                    except Exception as exc:  # an unreachable week is skipped, not fatal
                        print(f"week {key}: ESPN lines unavailable ({exc})", file=sys.stderr)
                        fetched[key] = {}
                line = fetched[key].get((row["player_id"], metric))
            if line is not None:
                rows.append({"week": key, "metric": metric, "proj": float(projection),
                             "line": float(line), "act": float(actual)})
    return rows


def fit_w(rows: list[dict]) -> float:
    num = sum((r["proj"] - r["line"]) * (r["act"] - r["line"]) for r in rows)
    den = sum((r["proj"] - r["line"]) ** 2 for r in rows)
    return max(0.0, num / den) if den else 0.0


def with_raw(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if r["act"] == r["line"]:
            continue
        priced = prop_pricing.price(r["metric"], r["proj"], r["line"])
        if priced is not None:
            out.append({**r, "family": priced["family"], "raw": priced["raw"],
                        "y": float(r["act"] > r["line"])})
    return out


def fit_calibration(rows: list[dict]) -> tuple[float, float]:
    best = None
    for s in [x / 100 for x in range(5, 101, 5)]:
        for c in [x / 200 for x in range(88, 113)]:
            brier = sum((min(max(0.5 + s * (r["raw"] - c), 0.01), 0.99) - r["y"]) ** 2
                        for r in rows) / len(rows)
            if best is None or brier < best[0]:
                best = (brier, s, c)
    return best[1], best[2]


def ranked_record(rows: list[dict]) -> tuple[int, int]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["week"], r["metric"])].append(r)
    wins = plays = 0
    for group in groups.values():
        if len(group) < 9:
            continue
        group.sort(key=lambda r: r["raw"])
        third = len(group) // 3
        wins += sum(r["y"] for r in group[-third:]) + sum(1 - r["y"] for r in group[:third])
        plays += 2 * third
    return int(wins), plays


def main() -> None:
    args = sys.argv[1:]
    since = args[args.index("--since") + 1] if "--since" in args else None
    if "--rows" in args:
        rows = json.loads(Path(args[args.index("--rows") + 1]).read_text(encoding="utf-8"))
        rows = [{**r, "week": tuple(r["week"]) if isinstance(r["week"], list) else r["week"]}
                for r in rows]
    else:
        path = next((a for a in args if not a.startswith("--") and a != since), None)
        rows = joined_rows(load_ledger(path), since)
    rows = with_raw(rows)
    weeks = sorted({r["week"] for r in rows})
    print(f"{len(rows)} graded lines across {len(weeks)} weeks", file=sys.stderr)
    wins = plays = 0
    brier = []
    for week in weeks:
        s, c = fit_calibration([r for r in rows if r["week"] != week])
        for r in rows:
            if r["week"] != week:
                continue
            p = min(max(0.5 + s * (r["raw"] - c), 0.01), 0.99)
            brier.append((p - r["y"]) ** 2)
            if max(p, 1 - p) >= prop_pricing.MIN_PROBABILITY:
                plays += 1
                wins += (p >= 0.5) == (r["y"] == 1.0)
    s, c = fit_calibration(rows)
    ranked_wins, ranked_plays = ranked_record(rows)
    weights = {f: round(fit_w([r for r in rows if r["family"] == f]), 3)
               for f in sorted({r["family"] for r in rows})}
    evidence = {"lines": len(rows), "plays": plays, "wins": wins,
                "hit_rate": round(wins / plays, 3) if plays else 0.0,
                "ranked_wins": ranked_wins, "ranked_plays": ranked_plays}
    report = {"calibration": {"s": s, "c": c}, "blend_weight": weights, "evidence": evidence,
              "held_out_brier": round(sum(brier) / max(len(brier), 1), 4), "coin_brier": 0.25,
              "weeks": [str(w) for w in weeks]}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
