"""nfl-model CLI.

    nfl-model status                     authority and unmet production gates
    nfl-model board                      the week's slate, scores and totals
    nfl-model ratings                    power ratings, offence and defence split
    nfl-model units                      offensive and defensive rankings
    nfl-model divisions                  simulated division and playoff odds
    nfl-model forecast --games s.json     the moneyline probability contract
    nfl-model export --out board.json     the full JSON contract
    nfl-model build-site --out index.html the static dashboard

Every command that takes `--season`/`--week` defaults to the current season and
the first week that still has an unplayed game, so a scheduled build needs no
arguments and cannot drift onto last week's slate.

Output is deliberately ASCII. These commands run in CI logs and Windows consoles,
where a box-drawing character is an encoding crash rather than a nicer table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import authority as auth
from . import divisions as divisions_mod
from . import export as export_mod
from . import matrix, ratings, teams
from . import season as season_mod
from .forecast import DEFAULT_LAMBDA, SPREAD_LAMBDA, forecast_slate, write_slate


def _load_games(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["games"] if isinstance(data, dict) else data


def _slate(args) -> season_mod.Slate:
    return season_mod.assemble(getattr(args, "season", None), getattr(args, "week", None))


def _outlooks(slate, simulations: int):
    games = divisions_mod.build_games(
        [row for row in slate.schedule if row["season"] == slate.season],
        slate.table,
        margin_of=season_mod.margin_for(slate.table, slate.forms),
    )
    return divisions_mod.simulate(games, slate.table, simulations=simulations)


def _header(slate) -> None:
    a = slate.authority
    print(f"\n  NFL board -- {slate.season} week {slate.week}")
    print(f"  authority: {a.level.value}   may_bet={a.may_bet}   "
          f"({len(a.unmet_gates)} of {len(auth.REQUIRED_GATES)} gates unmet)")
    regime = "in-season form" if slate.in_season else "preseason prior, no games played"
    print(f"  regime: {regime}\n")


def _cmd_status(_: argparse.Namespace) -> int:
    a = auth.current()
    print(f"\n  authority : {a.level.value}")
    print(f"  may_bet   : {a.may_bet}")
    print(f"  lam (ml)  : {DEFAULT_LAMBDA}")
    print(f"  lam (ats) : {SPREAD_LAMBDA}")
    print(f"  matrix    : {matrix.LINEAGE_VERSION} ({matrix.STATUS})")
    print(f"  evidence  : {a.evidence}\n")
    print(f"  unmet production gates ({len(a.unmet_gates)}/{len(auth.REQUIRED_GATES)}):")
    for gate in a.unmet_gates:
        print(f"    - {gate}")
    print("\n  Measured expanding-season on 1,615 games (2020-2025):")
    print("    margin  model 10.2274   market 9.7644   ATS 49.56% [47.1, 52.0]")
    print("    total   model 10.6598   market 10.2833  O/U 49.66%")
    print("  The model does not beat the closing line. At lam = 0 the published")
    print("  price is the market, which is a calibrated price and not an edge.\n")
    return 0


def _cmd_forecast(args: argparse.Namespace) -> int:
    games = _load_games(args.games)
    if args.destination:
        print(write_slate(games, args.destination, args.lam))
        return 0
    print(json.dumps(forecast_slate(games, args.lam), indent=2))
    return 0


def _cmd_board(args: argparse.Namespace) -> int:
    slate = _slate(args)
    _header(slate)
    if not slate.projections:
        print("  no regular-season games found for that week\n")
        return 0
    print(f"  {'kickoff':<24} {'matchup':<16} {'score':>9} {'total':>12} "
          f"{'model':>7} {'market':>7} {'gap':>7}  action")
    print(f"  {'-' * 24} {'-' * 16} {'-' * 9} {'-' * 12} "
          f"{'-' * 7} {'-' * 7} {'-' * 7}  ------")
    for p in slate.projections:
        matchup = (f"{p.away} vs {p.home} (N)" if p.neutral else f"{p.away} @ {p.home}")
        score = ("--" if p.projected_home_score is None
                 else f"{p.projected_away_score:.0f}-{p.projected_home_score:.0f}"
                      + ("" if p.total_modelled else "*"))
        comparison_total = p.comparison_total
        if p.projected_total is None:
            total = "--"
        else:
            total = f"{p.projected_total:.1f}"
            if comparison_total is not None:
                total += f"/{comparison_total:.1f}"
        model = season_mod.signed(p.model_margin)
        market = season_mod.signed(p.comparison_margin)
        # Bracketed so it never reads as a tradable edge.
        gap = ("  --" if p.market_gap is None
               else f"({season_mod.signed(p.market_gap)})")
        print(f"  {p.kickoff[:24]:<24} {matchup[:16]:<16} {score:>9} {total:>12} "
              f"{model:>7} {market:>7} {gap:>7}  {p.action}")
    print(f"\n  {len(slate.projections)} games - score is the MODEL projection "
          f"(away-home), total column is model/market")
    print("  market is exact DraftKings when posted, otherwise the nflverse benchmark")
    print("  (parenthesised) = information gap, not an edge - "
          "published margin equals the market at lam=0")
    print("  * = league-mean total, no form available\n")
    return 0


def _cmd_ratings(args: argparse.Namespace) -> int:
    slate = _slate(args)
    _header(slate)
    outlooks = _outlooks(slate, args.simulations) if args.simulations else []
    wins = {o.team: o.projected_wins for o in outlooks}
    print(f"  {'#':>3} {'team':<5} {'division':<11} {'rating':>8} {'off':>7} "
          f"{'def':>7} {'eff':>7} {'projW':>6}")
    print(f"  {'-' * 3} {'-' * 5} {'-' * 11} {'-' * 8} {'-' * 7} "
          f"{'-' * 7} {'-' * 7} {'-' * 6}")
    for rank, team, value in ratings.rank_table(slate.table):
        form = slate.forms.get(team)
        offense = matrix.offense_index(form) if form else None
        defense = matrix.defense_index(form) if form else None
        efficiency = matrix.efficiency_rating(form) if form else None
        projected = wins.get(team)
        print(f"  {rank:>3} {team:<5} {teams.get(team).division:<11} {value:>+8.2f} "
              f"{'   --' if offense is None else f'{offense:>+7.2f}'} "
              f"{'   --' if defense is None else f'{defense:>+7.2f}'} "
              f"{'   --' if efficiency is None else f'{efficiency:>+7.2f}'} "
              f"{'   --' if projected is None else f'{projected:>6.1f}'}")
    print("\n  rating = opponent-adjusted scoring margin; eff = off + def from the")
    print("  matchup model. The published margin is the 50/50 blend of the two.")
    print(f"  home field: rating path {ratings.HOME_FIELD_POINTS:.2f} + efficiency path "
          f"{matrix.COEFFICIENTS['home_field']:.2f} -> effective "
          f"{(ratings.HOME_FIELD_POINTS + matrix.COEFFICIENTS['home_field']) / 2:.2f}\n")
    return 0


def _cmd_units(args: argparse.Namespace) -> int:
    slate = _slate(args)
    _header(slate)
    for label, index in (("OFFENSE", matrix.offense_index),
                         ("DEFENSE", matrix.defense_index)):
        entries = sorted(
            ((index(form), team, form) for team, form in slate.forms.items()
             if index(form) is not None), reverse=True)
        print(f"  {label} -- points per game above an average unit")
        prefix = "off" if label == "OFFENSE" else "def"
        print(f"  {'#':>3} {'team':<5} {'pts':>7} {'epa/pl':>8} {'1st dn':>8} "
              f"{'expl':>7} {'sack':>7} {'to':>7}")
        for rank, (value, team, form) in enumerate(entries, start=1):
            print(f"  {rank:>3} {team:<5} {value:>+7.2f} "
                  f"{getattr(form, f'{prefix}_epa'):>8.3f} "
                  f"{getattr(form, f'{prefix}_first_down'):>8.3f} "
                  f"{getattr(form, f'{prefix}_explosive'):>7.3f} "
                  f"{getattr(form, f'{prefix}_sack'):>7.3f} "
                  f"{getattr(form, f'{prefix}_turnover'):>7.3f}")
        print("  a defensive row records what OPPONENTS did: low epa/1st down/expl")
        print("  and high sack/turnover are all good defence\n")
    return 0


def _cmd_players(args: argparse.Namespace) -> int:
    slate = _slate(args)
    _header(slate)
    players = slate.player_projections
    if args.position:
        players = [row for row in players if row.position == args.position]
    if args.team:
        selected_team = teams.canonical(args.team)
        players = [row for row in players if row.team == selected_team]
    if not players:
        print("  no active player projections match those filters\n")
        return 0
    print(f"  {'player':<24} {'pos':<4} {'game':<10} {'role':<14} "
          f"{'volume':>9} {'yards':>8} {'td':>7} {'conf':>7}")
    print(f"  {'-' * 24} {'-' * 4} {'-' * 10} {'-' * 14} "
          f"{'-' * 9} {'-' * 8} {'-' * 7} {'-' * 7}")
    for player in players:
        metrics = player.metrics
        if player.position == "QB":
            volume = f"{metrics.get('pass_attempts', 0):.1f} att"
            yards = f"{metrics.get('passing_yards', 0):.1f}p"
            td = f"{metrics.get('passing_tds', 0):.2f}p"
        elif player.position == "RB":
            volume = f"{metrics.get('carries', 0):.1f} car"
            yards = f"{metrics.get('rushing_yards', 0):.1f}r"
            td = f"{metrics.get('anytime_td_probability', 0):.0%}"
        elif player.position in {"WR", "TE"}:
            volume = f"{metrics.get('targets', 0):.1f} tgt"
            yards = f"{metrics.get('receiving_yards', 0):.1f}c"
            td = f"{metrics.get('anytime_td_probability', 0):.0%}"
        else:
            volume = f"{metrics.get('fg_attempts', 0):.1f} FGA"
            yards = "--"
            td = f"{metrics.get('kicking_points', 0):.1f}pt"
        game = f"{player.team}-{player.opponent}"
        print(f"  {player.player_name[:24]:<24} {player.position:<4} {game:<10} "
              f"{player.role_continuity[:14]:<14} {volume:>9} {yards:>8} "
              f"{td:>7} {player.confidence:>7}")
    print("\n  current active roster + pre-kickoff depth chart determine role")
    print("  DraftKings team game lines condition volume; no player lines or edges\n")
    return 0


def _cmd_divisions(args: argparse.Namespace) -> int:
    slate = _slate(args)
    _header(slate)
    outlooks = _outlooks(slate, args.simulations)
    if not outlooks:
        print("  no rateable schedule found\n")
        return 0
    for name, members in divisions_mod.by_division(outlooks).items():
        if not members:
            continue
        print(f"  {name}")
        for outlook in members:
            print(f"    {outlook.team:<5} {outlook.projected_wins:5.1f}-"
                  f"{outlook.projected_losses:<5.1f} div {outlook.win_division:6.1%}  "
                  f"playoff {outlook.make_playoffs:6.1%}  #1 seed {outlook.top_seed:5.1%}")
        print()
    print(f"  {args.simulations:,} simulated seasons over the real "
          f"{slate.season} fixture list. Ratings are frozen for the whole")
    print("  simulation, nobody gets injured, and draws are independent, so the")
    print("  tails are thinner than reality's.\n")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    slate = _slate(args)
    outlooks = _outlooks(slate, args.simulations) if args.simulations else []
    if args.out:
        print(export_mod.write(slate, args.out, outlooks))
    else:
        print(json.dumps(export_mod.payload(slate, outlooks), indent=2))
    return 0


def _cmd_build_site(args: argparse.Namespace) -> int:
    from .site import build_site
    out = build_site(Path(args.out), args.season, args.week, simulations=args.simulations)
    print(f"wrote dashboard to {out}")
    return 0


def _add_slate_args(parser: argparse.ArgumentParser, *, simulations: int | None = None
                    ) -> argparse.ArgumentParser:
    parser.add_argument("--season", type=int, help="default: current season")
    parser.add_argument("--week", type=int,
                        help="default: first week with an unplayed game")
    if simulations is not None:
        parser.add_argument("--simulations", type=int, default=simulations,
                            help="season simulations; 0 skips them")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nfl-model",
        description="NFL projections with an explicit authority gate.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="show authority and unmet production gates")
    _add_slate_args(sub.add_parser("board", help="the week's slate"))
    _add_slate_args(sub.add_parser("ratings", help="power ratings"), simulations=0)
    _add_slate_args(sub.add_parser("units", help="offensive and defensive rankings"))
    p = _add_slate_args(sub.add_parser(
        "players", help="QB/RB/WR/TE/K next-game projections"))
    p.add_argument("--position", choices=("QB", "RB", "WR", "TE", "K"))
    p.add_argument("--team", help="team abbreviation")
    _add_slate_args(sub.add_parser("divisions", help="simulated division races"),
                    simulations=divisions_mod.SIMULATIONS)

    f = sub.add_parser("forecast", help="the moneyline probability contract")
    f.add_argument("--games", required=True,
                   help="JSON list (or {'games': [...]}) with home_team, away_team, "
                        "home_american, away_american, optional structural_home")
    f.add_argument("--destination", help="write JSON here instead of stdout")
    f.add_argument("--lam", type=float, default=DEFAULT_LAMBDA,
                   help="deviation shrinkage; tune in nfl-genesis, not here")

    x = _add_slate_args(sub.add_parser("export", help="the full JSON contract"),
                        simulations=divisions_mod.SIMULATIONS)
    x.add_argument("--out", help="write here instead of stdout")

    s = _add_slate_args(sub.add_parser("build-site", help="render the dashboard"),
                        simulations=divisions_mod.SIMULATIONS)
    s.add_argument("--out", default="docs/index.html")

    args = parser.parse_args(argv)
    return {
        "status": _cmd_status,
        "board": _cmd_board,
        "ratings": _cmd_ratings,
        "units": _cmd_units,
        "players": _cmd_players,
        "divisions": _cmd_divisions,
        "forecast": _cmd_forecast,
        "export": _cmd_export,
        "build-site": _cmd_build_site,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
