"""Refit the line-anchored prop pricing (src/nflmodel/prop_pricing.py).

Joins every graded player projection in the shadow ledger to the DraftKings
line it faced: the line stored on the row when there is one, otherwise the
closing line ESPN still publishes for the finished game. Then, per market
family:

* w - the blend weight minimising squared error of ``line + w * (proj - line)``
  (least squares through the line, floored at 0);
* b - a ridge-penalised logistic of P(over) on the scaled gap, through 50%.

Every week is scored by a model fitted on the other weeks, and the record of
the plays priced at >= 55% is the evidence that `prop_pricing.research_only`
reads. Prints the constants to paste and writes reports/prop_pricing_fit.json.

    PYTHONPATH=src python scripts/fit_prop_pricing.py [ledger.json]

Pure standard library, like the package.
"""

from __future__ import annotations

import json
import math
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


def joined_rows(ledger: dict) -> list[dict]:
    graded = [r for r in ledger.get("player_snapshots", []) if r.get("status") == "graded"]
    weeks = sorted({(int(r["season"]), int(r["week"])) for r in graded})
    fetched: dict[tuple[int, int], dict] = {}
    rows = []
    for row in graded:
        key = (int(row["season"]), int(row["week"]))
        stored = row.get("prop_lines") or {}
        for metric, projection in (row.get("metrics") or {}).items():
            family = prop_pricing.FAMILY.get(metric)
            actual = (row.get("actual_metrics") or {}).get(metric)
            if family is None or actual is None:
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
            if line is None or float(actual) == float(line):
                continue
            rows.append({"week": key, "family": family, "proj": float(projection),
                         "line": float(line), "act": float(actual)})
    print(f"{len(rows)} graded lines across {len(weeks)} weeks", file=sys.stderr)
    return rows


def fit_w(rows: list[dict]) -> float:
    num = sum((r["proj"] - r["line"]) * (r["act"] - r["line"]) for r in rows)
    den = sum((r["proj"] - r["line"]) ** 2 for r in rows)
    return max(0.0, num / den) if den else 0.0


def fit_logistic(rows: list[dict], l2: float = 1.0) -> tuple[float, float]:
    a = b = 0.0
    for _ in range(200):
        ga = gb = 0.0
        ha = hb = 1e-9
        for r in rows:
            x = prop_pricing.gap_feature(r["proj"], r["line"])
            p = 1.0 / (1.0 + math.exp(-(a + b * x)))
            y = float(r["act"] > r["line"])
            ga += p - y
            gb += (p - y) * x
            ha += p * (1 - p)
            hb += p * (1 - p) * x * x
        # No intercept: a market's base over/under rate is not a reason to back
        # any particular player (fitted freely it priced 269 receiving unders
        # and 0 overs in week 1), and without prices it may only be vig shading.
        b -= (gb + l2 * b) / (hb + l2)
    return a, b


def main() -> None:
    rows = joined_rows(load_ledger(sys.argv[1] if len(sys.argv) > 1 else None))
    weeks = sorted({r["week"] for r in rows})
    families = sorted({r["family"] for r in rows})
    held = {"plays": 0, "wins": 0, "brier": [], "coin": []}
    by_family = defaultdict(lambda: {"plays": 0, "wins": 0, "brier": [], "lines": 0})
    for week in weeks:
        for family in families:
            train = [r for r in rows if r["week"] != week and r["family"] == family]
            test = [r for r in rows if r["week"] == week and r["family"] == family]
            if not train or not test:
                continue
            a, b = fit_logistic(train)
            for r in test:
                x = prop_pricing.gap_feature(r["proj"], r["line"])
                p = 1.0 / (1.0 + math.exp(-(a + b * x)))
                y = float(r["act"] > r["line"])
                stats = by_family[family]
                stats["lines"] += 1
                stats["brier"].append((p - y) ** 2)
                held["brier"].append((p - y) ** 2)
                held["coin"].append(0.25)
                if max(p, 1 - p) >= prop_pricing.MIN_PROBABILITY:
                    won = (p >= 0.5) == (y == 1.0)
                    for bucket in (held, stats):
                        bucket["plays"] += 1
                        bucket["wins"] += won
    coefficients = {}
    for family in families:
        subset = [r for r in rows if r["family"] == family]
        a, b = fit_logistic(subset)
        coefficients[family] = {"w": round(fit_w(subset), 3), "a": round(a, 3), "b": round(b, 3)}
    evidence = {
        "weeks": [f"{s}-{w}" for s, w in weeks],
        "plays": held["plays"],
        "wins": held["wins"],
        "hit_rate": round(held["wins"] / held["plays"], 3) if held["plays"] else 0.0,
    }
    report = {
        "lines": len(rows),
        "coefficients": coefficients,
        "evidence": evidence,
        "held_out_brier": round(sum(held["brier"]) / max(len(held["brier"]), 1), 4),
        "coin_brier": 0.25,
        "by_family": {
            f: {"lines": s["lines"], "plays": s["plays"],
                "hit_rate": round(s["wins"] / s["plays"], 3) if s["plays"] else None,
                "brier": round(sum(s["brier"]) / len(s["brier"]), 4) if s["brier"] else None}
            for f, s in sorted(by_family.items())
        },
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("\nPaste into prop_pricing.py:")
    print("COEFFICIENTS = " + json.dumps(coefficients, indent=4).replace("}\n}", "},\n}"))
    print(f"EVIDENCE = {json.dumps(evidence)}")


if __name__ == "__main__":
    main()
