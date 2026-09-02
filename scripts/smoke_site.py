"""Structural smoke test for the rendered NFL publication bundle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED = (
    "bd-wordmark",
    "DraftKings lines",
    "Production data health",
    "RESEARCH_ONLY",
    "Power Ratings",
    "Offensive player &amp; kicker projections",
    "1-800-GAMBLER",
    "not betting advice",
)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: smoke_site.py <index.html>")
        return 2
    html_path = Path(sys.argv[1])
    html = html_path.read_text(encoding="utf-8")
    failures = [f"missing {value!r}" for value in REQUIRED if value not in html]
    cards = html.count('class="bd-card"')
    if cards <= 0:
        failures.append("board rendered zero games")
    if 'bd-status is-pos">BET<' in html:
        failures.append("research-only page rendered BET")
    for name in ("board.json", "build.json", "record.json"):
        path = html_path.parent / name
        if not path.is_file():
            failures.append(f"missing sibling artifact {name}")
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            failures.append(f"invalid JSON artifact {name}")
    print(f"games={cards} bytes={len(html):,}")
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print("smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
