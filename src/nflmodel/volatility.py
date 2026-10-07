"""Team volatility: how reliably a team's games land where the model projects.

For each team and each market, the question is the same: when the advanced
metrics project this team's game, how far does the result usually stray, and in
the direction that hurts that bet? A team ranked 1 conforms; a team ranked last
produces outcomes the metrics did not see coming.

Per game, from the team's side, against the model's own pre-game numbers (never
the market's -- the point is conformity to *our* metrics):

    spread     |actual margin - model margin|                  points of miss
    moneyline  |won - p| - 2p(1-p),  p = Phi(model margin / SD) surprise beyond
                                                               what p implied
    over       max(model total - actual total, 0)              shortfall, the
                                                               miss an over eats
    under      max(actual total - model total, 0)              overshoot, the
                                                               miss an under eats

The moneyline term subtracts its own expectation, so a team that only plays
coin flips is not called volatile for losing half of them -- only for losing
more of its favoured games (or winning more of its underdog games) than the
probabilities said. Over and under are scored separately because they fail on
opposite tails: a defence that occasionally collapses breaks unders and leaves
overs alone.

**Predictive, not descriptive.** A season is ten to seventeen games, and most of
the spread in raw team misses is luck. Each team's observed mean is shrunk
toward the pool:

    expected = pool + w * (observed - pool),  w = n_eff / (n_eff + k)
    n_eff    = games this season + decay * games last season

`k` and `decay` are fitted per market by `fit` on a walk-forward replay of the
production path, choosing the pair that best predicts the *second half* of a
team's season from its first half plus the previous season, with every season
scored by parameters chosen on the others. A market whose held-out skill is not
positive overall and in most seasons is a market where volatility is not a
team trait. Shrinkage there would put every team on the pool, so that market is
ranked on this season's observed misses and labelled ``descriptive``: it says
what happened, and makes no claim that it will continue.
"""

from __future__ import annotations

import html
import math
import pprint
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

MARKETS = ("spread", "moneyline", "over", "under")
LABELS = {
    "spread": "Spread",
    "moneyline": "Moneyline",
    "over": "Over",
    "under": "Under",
}
UNITS = {
    "spread": "pts missed / game",
    "moneyline": "excess surprise / game",
    "over": "pts short of total / game",
    "under": "pts over total / game",
}

# Shrinkage candidates in games. INFINITE_K is "no team signal": every team is
# predicted at the pool mean.
K_GRID = (2.0, 4.0, 6.0, 10.0, 15.0, 25.0, 40.0, 60.0, 100.0, 160.0, 250.0)
INFINITE_K = math.inf
DECAY_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
# A team-season needs this many games before it can be split into an observed
# half and a target half for fitting.
MIN_SPLIT_GAMES = 6
# Grades are published only for a market that is a measured trait; within it,
# the top and bottom fifths of the ranked teams are labelled.
GRADE_SHARE = 0.2


@dataclass(frozen=True)
class GradedGame:
    """One completed game with the model's pre-game numbers, home perspective."""

    season: int
    week: int
    home: str
    away: str
    model_margin: float
    actual_margin: float
    model_total: float | None = None
    actual_total: float | None = None


@dataclass(frozen=True)
class TeamGame:
    team: str
    season: int
    week: int
    losses: dict[str, float]
    margin_residual: float
    total_residual: float | None
    win_probability: float
    won: float


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def team_games(games: list[GradedGame], *, margin_sd: float) -> list[TeamGame]:
    """Both sides of every game, each scored from that team's perspective."""
    out: list[TeamGame] = []
    for g in games:
        for team, sign in ((g.home, 1.0), (g.away, -1.0)):
            projected = sign * float(g.model_margin)
            actual = sign * float(g.actual_margin)
            p = _phi(projected / margin_sd)
            won = 1.0 if actual > 0 else 0.0 if actual < 0 else 0.5
            losses = {
                "spread": abs(actual - projected),
                "moneyline": abs(won - p) - 2.0 * p * (1.0 - p),
            }
            total_residual = None
            if g.model_total is not None and g.actual_total is not None:
                total_residual = float(g.actual_total) - float(g.model_total)
                losses["over"] = max(-total_residual, 0.0)
                losses["under"] = max(total_residual, 0.0)
            out.append(TeamGame(team, int(g.season), int(g.week), losses,
                                actual - projected, total_residual, p, won))
    return out


# -- shrinkage ------------------------------------------------------------------
def _weight(n_eff: float, k: float) -> float:
    if n_eff <= 0 or math.isinf(k):
        return 0.0
    return n_eff / (n_eff + k)


def _pooled(current: list[float], prior: tuple[int, float], decay: float
            ) -> tuple[float, float]:
    """(observed mean, effective games) with last season's games discounted.

    `prior` is (games, summed loss) -- all production needs to carry a season.
    """
    n_eff = len(current) + decay * prior[0]
    if n_eff <= 0:
        return 0.0, 0.0
    return (sum(current) + decay * prior[1]) / n_eff, n_eff


def _shrink(observed: float, n_eff: float, pool: float, k: float) -> tuple[float, float]:
    w = _weight(n_eff, k)
    return pool + w * (observed - pool), w


# -- fitting --------------------------------------------------------------------
@dataclass(frozen=True)
class _Case:
    """One team-season split for fitting: observed half, prior season, target half."""

    season: int
    first: list[float]
    prior: list[float]
    target: list[float]


def _cases(rows: list[TeamGame], market: str) -> list[_Case]:
    by: dict[tuple[str, int], list[TeamGame]] = defaultdict(list)
    for r in rows:
        if market in r.losses:
            by[(r.team, r.season)].append(r)
    seasons = sorted({s for _, s in by})
    previous = {s: max((p for p in seasons if p < s), default=None) for s in seasons}
    out = []
    for (team, season), games in by.items():
        if len(games) < MIN_SPLIT_GAMES:
            continue
        games = sorted(games, key=lambda g: g.week)
        half = len(games) // 2
        last = previous[season]
        prior = [g.losses[market] for g in by.get((team, last), [])] if last else []
        out.append(_Case(season, [g.losses[market] for g in games[:half]], prior,
                         [g.losses[market] for g in games[half:]]))
    return out


def _sse(cases: list[_Case], k: float, decay: float) -> float:
    """Target-weighted squared error of the second-half prediction.

    The pool is what production would have: every observed game in the same
    season's first halves plus discounted prior seasons.
    """
    by_season: dict[int, list[_Case]] = defaultdict(list)
    for c in cases:
        by_season[c.season].append(c)
    sse = 0.0
    for season_cases in by_season.values():
        num = sum(sum(c.first) + decay * sum(c.prior) for c in season_cases)
        den = sum(len(c.first) + decay * len(c.prior) for c in season_cases)
        pool = num / den
        for c in season_cases:
            observed, n_eff = _pooled(c.first, (len(c.prior), sum(c.prior)), decay)
            predicted, _ = _shrink(observed, n_eff, pool, k)
            sse += len(c.target) * (statistics.fmean(c.target) - predicted) ** 2
    return sse


def _best(cases: list[_Case]) -> tuple[float, float]:
    candidates = [(k, d) for k in K_GRID for d in DECAY_GRID] + [(INFINITE_K, 0.0)]
    return min(candidates, key=lambda kd: _sse(cases, *kd))


def _corr(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3 or statistics.pstdev(a) == 0 or statistics.pstdev(b) == 0:
        return None
    return statistics.correlation(a, b)


def _split_half(cases: list[_Case]) -> float | None:
    """First-half mean against second-half mean, across team-seasons."""
    return _corr([statistics.fmean(c.first) for c in cases if c.first],
                 [statistics.fmean(c.target) for c in cases if c.first])


def _next_season(rows: list[TeamGame], market: str) -> float | None:
    by: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in rows:
        if market in r.losses:
            by[(r.team, r.season)].append(r.losses[market])
    seasons = sorted({s for _, s in by})
    following = dict(zip(seasons, seasons[1:]))
    a, b = [], []
    for (team, season), values in by.items():
        later = by.get((team, following.get(season)))
        if len(values) >= MIN_SPLIT_GAMES and later and len(later) >= MIN_SPLIT_GAMES:
            a.append(statistics.fmean(values))
            b.append(statistics.fmean(later))
    return _corr(a, b)


def fit(games: list[GradedGame], *, margin_sd: float, source: str) -> dict:
    """Choose `k` and `decay` per market; score each season held out."""
    rows = team_games(games, margin_sd=margin_sd)
    seasons = sorted({g.season for g in games})
    markets = {}
    for market in MARKETS:
        cases = _cases(rows, market)
        if not cases:
            continue
        held_sse = base_sse = 0.0
        folds = []
        for season in seasons:
            test = [c for c in cases if c.season == season]
            train = [c for c in cases if c.season != season]
            if not test or not train:
                continue
            k, decay = _best(train)
            fold_sse, fold_base = _sse(test, k, decay), _sse(test, INFINITE_K, 0.0)
            folds.append({"season": season, "k": k, "decay": decay,
                          "skill": round(1.0 - fold_sse / fold_base, 4) if fold_base else 0.0})
            held_sse += fold_sse
            base_sse += fold_base
        k, decay = _best(cases)
        skill = 1.0 - held_sse / base_sse if base_sse else 0.0
        # A trait must help out of sample overall AND in most seasons; one good
        # season carrying a pooled number is not a property of teams.
        trait = skill > 0.0 and sum(f["skill"] > 0 for f in folds) * 2 > len(folds)
        markets[market] = {
            "k": None if math.isinf(k) or not trait else k,
            "decay": decay if trait else 0.0,
            "trait": trait,
            "held_out_skill": round(skill, 4),
            "seasons_helped": sum(f["skill"] > 0 for f in folds),
            "seasons_tested": len(folds),
            "split_half_r": _round(_split_half(cases)),
            "next_season_r": _round(_next_season(rows, market)),
            "team_seasons": len(cases),
            "folds": [{**f, "k": None if math.isinf(f["k"]) else f["k"]} for f in folds],
        }
    return {
        "source": source,
        "seasons": seasons,
        "games": len(games),
        "margin_sd": margin_sd,
        "markets": markets,
        "prior_season": _season_aggregates(rows, seasons[-1]) if seasons else None,
    }


def _round(value: float | None, places: int = 4) -> float | None:
    return None if value is None else round(value, places)


def _season_aggregates(rows: list[TeamGame], season: int) -> dict:
    """Per-team [games, summed loss] for one season, so production can carry it
    forward without shipping the replay itself."""
    teams: dict[str, dict[str, list]] = defaultdict(dict)
    for r in rows:
        if r.season != season:
            continue
        for market, value in r.losses.items():
            n, total = teams[r.team].get(market, [0, 0.0])
            teams[r.team][market] = [n + 1, total + value]
    return {"season": season,
            "teams": {t: {m: [n, round(v, 4)] for m, (n, v) in sorted(ms.items())}
                      for t, ms in sorted(teams.items())}}


# -- production ranking -----------------------------------------------------------
def rank(
    games: list[GradedGame],
    *,
    season: int,
    params: dict,
    margin_sd: float,
    teams: set[str] | None = None,
) -> dict:
    """Rank every team on each market, from this season's graded games plus the
    previous season (graded games if supplied, else the fit's stored aggregate).

    `teams` limits who is ranked -- e.g. FBS only -- without dropping their
    opponents' games from the pool.
    """
    rows = team_games(games, margin_sd=margin_sd)
    current = [r for r in rows if r.season == season]
    prior_rows = [r for r in rows if r.season == season - 1]
    prior: dict[str, dict[str, tuple[int, float]]] = defaultdict(dict)
    for r in prior_rows:
        for market, value in r.losses.items():
            n, total = prior[r.team].get(market, (0, 0.0))
            prior[r.team][market] = (n + 1, total + value)
    stored = params.get("prior_season") or {}
    if not prior_rows and stored.get("season") == season - 1:
        for team, markets in (stored.get("teams") or {}).items():
            for market, (n, total) in markets.items():
                prior[team][market] = (int(n), float(total))

    by_team: dict[str, list[TeamGame]] = defaultdict(list)
    for r in current:
        by_team[r.team].append(r)
    universe = set(by_team) | set(prior)
    if teams is not None:
        universe &= teams

    market_params = params.get("markets") or {}
    out_markets = {}
    table: dict[str, dict] = {t: {"team": t, "games": len(by_team.get(t, [])),
                                  "prior_games": max((n for n, _ in prior.get(t, {}).values()),
                                                     default=0),
                                  "markets": {}} for t in universe}
    for market in MARKETS:
        mp = market_params.get(market) or {}
        trait = bool(mp.get("trait"))
        k = INFINITE_K if mp.get("k") is None or not trait else float(mp["k"])
        decay = float(mp.get("decay") or 0.0) if trait else 0.0
        observed: dict[str, tuple[float, float, int]] = {}
        num = den = 0.0
        # Pool over every team with data, not just the ranked universe, so an
        # FBS team's FCS opponents still anchor the mean they were scored in.
        for team in set(by_team) | set(prior):
            cur = [r.losses[market] for r in by_team.get(team, []) if market in r.losses]
            mean, n_eff = _pooled(cur, prior.get(team, {}).get(market, (0, 0.0)), decay)
            if n_eff <= 0:
                continue
            observed[team] = (mean, n_eff, len(cur))
            num += mean * n_eff
            den += n_eff
        if not den:
            continue
        pool = num / den
        ranked = []
        for team in universe:
            if team not in observed:
                continue
            mean, n_eff, n_cur = observed[team]
            expected, w = _shrink(mean, n_eff, pool, k)
            cur = [r.losses[market] for r in by_team.get(team, []) if market in r.losses]
            entry = {
                "expected": round(expected, 4),
                "observed": round(statistics.fmean(cur), 4) if cur else None,
                "reliability": round(w, 3),
                "games": n_cur,
            }
            if market == "moneyline":
                games_ = by_team.get(team, [])
                entry["upsets"] = sum(1 for r in games_ if (r.win_probability - 0.5)
                                      * (r.won - 0.5) < 0)
                entry["expected_upsets"] = round(sum(min(r.win_probability,
                                                         1.0 - r.win_probability)
                                                     for r in games_), 2)
            table[team]["markets"][market] = entry
            # A trait ranks on the shrunk forecast. A market with no measured
            # team signal would shrink every team onto the pool and tie them all,
            # so it ranks on what this season actually did and says that is
            # description, not a forecast.
            key = expected if trait else entry["observed"]
            if key is not None:
                ranked.append((key, -n_eff, team))
        ranked.sort()
        band = max(1, round(len(ranked) * GRADE_SHARE)) if trait else 0
        for position, (_, _, team) in enumerate(ranked, start=1):
            entry = table[team]["markets"][market]
            entry["rank"] = position
            if not trait:
                entry["grade"] = None
            elif position <= band:
                entry["grade"] = "steady"
            elif position > len(ranked) - band:
                entry["grade"] = "volatile"
            else:
                entry["grade"] = "typical"
        out_markets[market] = {
            "label": LABELS[market],
            "unit": UNITS[market],
            "pool": round(pool, 4),
            "k": None if math.isinf(k) else k,
            "decay": decay,
            "trait": trait,
            "basis": "predictive" if trait else "descriptive",
            "held_out_skill": mp.get("held_out_skill"),
            "split_half_r": mp.get("split_half_r"),
            "next_season_r": mp.get("next_season_r"),
            "ranked": len(ranked),
        }

    for team, rows_ in by_team.items():
        if team not in table:
            continue
        table[team]["margin_bias"] = round(statistics.fmean(r.margin_residual for r in rows_), 2)
        totals_ = [r.total_residual for r in rows_ if r.total_residual is not None]
        table[team]["total_bias"] = round(statistics.fmean(totals_), 2) if totals_ else None

    through = max((r.week for r in current), default=0)
    return {
        "season": season,
        "through_week": through,
        "graded_games": len({(g.week, g.home, g.away) for g in games if g.season == season}),
        "fit_source": params.get("source"),
        "fit_seasons": params.get("seasons"),
        "markets": out_markets,
        "teams": sorted(table.values(), key=lambda t: t["team"]),
    }


def params() -> dict:
    """The shipped fit, generated into `volatility_fit.py` by the fit script.

    Generated as code rather than read from `reports/` so an installed package
    (the deploy does `pip install .`) carries it. Missing fit: every market is
    descriptive and there is no prior season -- degraded, not broken.
    """
    try:
        from . import volatility_fit
    except ImportError:
        return {}
    return volatility_fit.FIT


def write_params_module(result: dict, path) -> None:
    """Write the fit as an importable module; `reports/` keeps the full JSON."""
    lean = {key: result[key] for key in ("source", "seasons", "games", "margin_sd",
                                         "prior_season")}
    lean["markets"] = {m: {k: v for k, v in row.items() if k != "folds"}
                       for m, row in result["markets"].items()}
    body = pprint.pformat(lean, indent=1, width=96, sort_dicts=True)
    text = ('"""Generated by scripts/fit_volatility.py -- do not edit by hand.\n\n'
            'Evidence, with per-season folds: reports/volatility_fit.json.\n"""\n\n'
            f"FIT = {body}\n")
    Path(path).write_text(text, encoding="utf-8")


def build(snapshots: list[dict], *, season: int, margin_sd: float,
          teams: set[str] | None = None) -> dict:
    """The production ranking from a shadow ledger's snapshots."""
    return rank(graded_from_ledger(snapshots), season=season, params=params(),
                margin_sd=margin_sd, teams=teams)


def graded_from_ledger(snapshots: list[dict]) -> list[GradedGame]:
    """Latest pre-kickoff model numbers per graded game in a shadow ledger.

    The ledger stores one row per sportsbook quote. The volatility question is
    about the model's view going into the game, so the last snapshot recorded
    before kickoff wins; rows recorded after kickoff are ignored.
    """
    latest: dict[tuple, tuple[datetime, dict]] = {}
    for s in snapshots:
        if s.get("status") != "graded" or s.get("model_margin") is None:
            continue
        if s.get("actual_margin") is None:
            continue
        recorded, kickoff = _moment(s.get("recorded_at")), _moment(s.get("kickoff"))
        if recorded is None or (kickoff is not None and recorded > kickoff):
            continue
        key = (s.get("season"), s.get("week"), s.get("home"), s.get("away"))
        if key not in latest or recorded >= latest[key][0]:
            latest[key] = (recorded, s)
    out = []
    for (season, week, home, away), (_, s) in latest.items():
        if season is None or week is None or not home or not away:
            continue
        out.append(GradedGame(
            season=int(season), week=int(week), home=str(home), away=str(away),
            model_margin=float(s["model_margin"]),
            actual_margin=float(s["actual_margin"]),
            model_total=None if s.get("model_total") is None else float(s["model_total"]),
            actual_total=None if s.get("actual_total") is None else float(s["actual_total"]),
        ))
    return sorted(out, key=lambda g: (g.season, g.week, g.home))


def _moment(value) -> datetime | None:
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


# -- rendering ------------------------------------------------------------------
# Shared by both boards, so the section reads the same on the CFB and NFL pages.
# Uses only Chase token variables, which both sites vendor.
CSS = """
.vol-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(440px,100%),1fr));gap:14px}
.vol-card{border:1px solid var(--border-soft);border-radius:16px;padding:14px 16px 10px;
background:var(--card,transparent)}
.vol-head{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.vol-name{font-family:var(--display);font-weight:800;font-size:18px;letter-spacing:.02em}
.vol-basis{font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;border-radius:999px;
padding:2px 9px;border:1px solid var(--border-2);color:var(--text-2)}
.vol-basis--predictive{color:var(--ca-teal);border-color:var(--ca-teal)}
.vol-unit{color:var(--text-3);font-size:11.5px;margin-top:2px}
.vol-note{color:var(--text-2);font-size:12px;line-height:1.5;margin:8px 0 6px}
.vol-sub{font-family:var(--display);font-size:11px;letter-spacing:.14em;text-transform:uppercase;
color:var(--text-3);margin:10px 0 2px}
.vol-t{width:100%;border-collapse:collapse;font-size:13px}
.vol-t td,.vol-t th{padding:5px 6px;border-bottom:1px solid var(--border-soft);text-align:left}
.vol-t th{color:var(--text-3);font-weight:500;font-size:11px}
.vol-t .r{text-align:right;font-variant-numeric:tabular-nums}
.vol-t .n{color:var(--text-3);font-variant-numeric:tabular-nums}
.vol-team{display:flex;align-items:center;gap:8px}
.vol-team img{width:20px;height:20px;object-fit:contain;flex:0 0 20px}
.vol-steady{color:var(--ca-teal)}.vol-volatile{color:var(--ca-red)}
.vol-all{margin-top:16px}
.vol-all summary{cursor:pointer;color:var(--text-2);font-size:13px;padding:6px 0}
.vol-wrap{overflow-x:auto;border:1px solid var(--border-soft);border-radius:16px;margin-top:8px}
.vol-wrap .vol-t td,.vol-wrap .vol-t th{white-space:nowrap;padding:7px 10px}
"""

SHOWN_PER_END = 8


def _esc(value) -> str:
    return html.escape(str(value), quote=True)


def _value(market: str, entry: dict, predictive: bool) -> str:
    if market == "moneyline":
        if entry.get("games"):
            return f"{entry['upsets']} vs {entry['expected_upsets']:.1f}"
        return "&ndash;"
    value = entry["expected"] if predictive else entry.get("observed")
    return "&ndash;" if value is None else f"{value:.1f}"


def _team_cell(team: str, logo) -> str:
    src = logo(team) if logo else None
    img = f'<img src="{_esc(src)}" alt="" loading="lazy">' if src else ""
    return f'<div class="vol-team">{img}<span>{_esc(team)}</span></div>'


def _note(info: dict, payload: dict) -> str:
    seasons = payload.get("fit_seasons") or []
    span = f"{seasons[0]}&ndash;{seasons[-1]}" if seasons else "the replay"
    skill = info.get("held_out_skill")
    skill_text = f"{skill:+.1%}" if skill is not None else "n/a"
    if info.get("trait"):
        prior = (f"; last season counts {info['decay']:.2g}&times; per game"
                 if info.get("decay") else "")
        helped, tested = info.get("seasons_helped"), info.get("seasons_tested")
        record = f" (better in {helped} of {tested} seasons)" if tested else ""
        return (f"A measured team trait: in {span}, held-out seasons predicted second-half "
                f"volatility {skill_text} better than calling every team average{record}. "
                f"Shrunk toward the field by {info['k']:g} games{prior}.")
    helped, tested = info.get("seasons_helped"), info.get("seasons_tested")
    record = f", better in {helped} of {tested} seasons" if tested else ""
    return (f"Not a team trait: in {span}, a team&rsquo;s past volatility did not reliably "
            f"predict its future volatility (held-out {skill_text}{record}). Ranked on this "
            "season&rsquo;s misses only &mdash; what happened, not a forecast.")


def _market_card(market: str, info: dict, teams: list[dict], payload: dict, logo) -> str:
    predictive = bool(info.get("trait"))
    rows = sorted((t for t in teams if "rank" in t["markets"].get(market, {})),
                  key=lambda t: t["markets"][market]["rank"])
    if not rows:
        return ""
    n = min(SHOWN_PER_END, len(rows) // 2) or len(rows)
    column = ("upsets vs exp." if market == "moneyline"
              else "forecast" if predictive else "pts")

    def block(chunk: list[dict], label: str) -> str:
        body = "".join(
            f'<tr><td class="n">{t["markets"][market]["rank"]}</td>'
            f'<td>{_team_cell(t["team"], logo)}</td>'
            f'<td class="r">{_value(market, t["markets"][market], predictive)}</td>'
            f'<td class="r n">{t["markets"][market]["games"]}</td></tr>'
            for t in chunk)
        return (f'<div class="vol-sub">{label}</div><table class="vol-t"><thead><tr>'
                f'<th>#</th><th>Team</th><th class="r">{column}</th><th class="r">G</th>'
                f'</tr></thead><tbody>{body}</tbody></table>')

    basis = "predictive" if predictive else "descriptive"
    return f"""<article class="vol-card">
<div class="vol-head"><span class="vol-name">{_esc(info['label'])}</span>
<span class="vol-basis vol-basis--{basis}">{basis}</span></div>
<div class="vol-unit">{_esc(info['unit'])}</div>
<p class="vol-note">{_note(info, payload)}</p>
{block(rows[:n], "Most conforming")}
{block(rows[-n:], "Most volatile")}
</article>"""


def _full_table(teams: list[dict], payload: dict, logo) -> str:
    markets = [m for m in MARKETS if m in payload["markets"]]

    def cell(team: dict, market: str) -> str:
        entry = team["markets"].get(market) or {}
        if "rank" not in entry:
            return '<td class="r">&ndash;</td>'
        cls = {"steady": "vol-steady", "volatile": "vol-volatile"}.get(entry.get("grade"), "")
        value = _value(market, entry, bool(payload["markets"][market].get("trait")))
        return f'<td class="r {cls}">{entry["rank"]} <span class="n">({value})</span></td>'

    def signed(value) -> str:
        return "&ndash;" if value is None else f"{value:+.1f}"

    body = "".join(
        f'<tr><td>{_team_cell(t["team"], logo)}</td><td class="r">{t["games"]}</td>'
        + "".join(cell(t, m) for m in markets)
        + f'<td class="r">{signed(t.get("margin_bias"))}</td>'
        f'<td class="r">{signed(t.get("total_bias"))}</td></tr>'
        for t in sorted(teams, key=lambda t: t["team"]))
    headers = "".join(f'<th class="r">{_esc(payload["markets"][m]["label"])}</th>'
                      for m in markets)
    return f"""<details class="vol-all"><summary>All {len(teams)} teams &mdash; rank (value).
Green is the steadiest fifth and red the most volatile fifth of a predictive market;
margin and total vs model are this season&rsquo;s average result minus projection.</summary>
<div class="vol-wrap"><table class="vol-t"><thead><tr><th>Team</th><th class="r">G</th>
{headers}<th class="r">Margin vs model</th><th class="r">Total vs model</th></tr></thead>
<tbody>{body}</tbody></table></div></details>"""


TITLE = "Team Volatility Rankings"


def _default_head(eyebrow: str, title: str, blurb: str) -> str:
    return (f'<div class="sec-eyebrow">{_esc(eyebrow)}</div>'
            f'<h2 class="sec-title">{title}</h2><p class="sec-blurb">{blurb}</p>')


def render_section(payload: dict | None, *, logo=None, eyebrow: str = "Volatility",
                   ranked: set[str] | None = None, head=_default_head) -> str:
    """The volatility section: four market cards plus the full table.

    `logo` maps a team to an image URL (or None); `ranked` limits the teams
    shown, e.g. to FBS, without changing anyone's rank; `head(eyebrow, title,
    blurb_html)` returns the site's own section-header markup.
    """
    if not payload or not payload.get("markets"):
        return ('<section id="volatility">'
                + head(eyebrow, TITLE, "No graded games yet this season and no fitted "
                       "prior; the ranking appears once the first week is graded.")
                + "</section>")
    teams = [t for t in payload["teams"] if ranked is None or t["team"] in ranked]
    blurb = f"""How reliably each team&rsquo;s results land where our
metrics project them, scored against the model&rsquo;s own pre-game margin, total and win
probability &mdash; not the market&rsquo;s. Rank 1 conforms most; the bottom of each list
produces the outcomes the metrics did not see coming. <b>Spread</b> is points missed per
game. <b>Moneyline</b> counts losses as the model&rsquo;s favourite plus wins as its
underdog, against how many its probabilities expected. <b>Over</b> and <b>Under</b> are
scored on the miss that loses each bet: how far games fell short of, or ran past, the
projected total. Through week {_esc(payload.get('through_week'))}:
{_esc(payload.get('graded_games'))} graded games this season. Research context, not a
betting signal."""
    cards = "".join(_market_card(m, payload["markets"][m], teams, payload, logo)
                    for m in MARKETS if m in payload["markets"])
    return f"""<section id="volatility">
{head(eyebrow, TITLE, blurb)}
<div class="vol-grid">{cards}</div>
{_full_table(teams, payload, logo)}
</section>"""
