"""nfl-model CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import authority as auth
from .forecast import DEFAULT_LAMBDA, forecast_slate, write_slate


def _load_games(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["games"] if isinstance(data, dict) else data


def _cmd_status(_: argparse.Namespace) -> int:
    a = auth.current()
    print(f"\n  authority : {a.level.value}")
    print(f"  may_bet   : {a.may_bet}")
    print(f"  lam       : {DEFAULT_LAMBDA}")
    print(f"  evidence  : {a.evidence}\n")
    print(f"  unmet production gates ({len(a.unmet_gates)}/{len(auth.REQUIRED_GATES)}):")
    for g in a.unmet_gates:
        print(f"    - {g}")
    print("\n  At lam = 0 the forecast equals the paired no-vig market by")
    print("  construction. That is a calibrated price, not an edge.\n")
    return 0


def _cmd_forecast(args: argparse.Namespace) -> int:
    games = _load_games(args.games)
    if args.destination:
        print(write_slate(games, args.destination, args.lam))
        return 0
    payload = forecast_slate(games, args.lam)
    print(json.dumps(payload, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nfl-model",
                                description="Market-anchored NFL forecasts with an authority gate.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="show authority and unmet production gates")

    f = sub.add_parser("forecast", help="forecast a slate from paired prices")
    f.add_argument("--games", required=True,
                   help="JSON list (or {'games': [...]}) with home_team, away_team, "
                        "home_american, away_american, optional structural_home")
    f.add_argument("--destination", help="write JSON here instead of stdout")
    f.add_argument("--lam", type=float, default=DEFAULT_LAMBDA,
                   help="deviation shrinkage; tune in nfl-genesis, not here")

    args = p.parse_args(argv)
    return {"status": _cmd_status, "forecast": _cmd_forecast}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
