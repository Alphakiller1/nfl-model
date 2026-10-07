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
coin flips is not called volatile for losing half of them. Over and under are
scored separately because they fail on opposite tails.

**Predictive philosophy.** The target is always a team's *future actual* misses --
that is what settles a bet -- and every ingredient below has to improve held-out
prediction of it, in both validation schemes, or it is switched off.

1. **Noise cancellation.** A miss has two parts. The game's own process stats
   (success rate, explosiveness, tempo -- deliberately nothing turnover-driven)
   imply what the margin and total *should* have been; the gap from that to the
   final score is turnovers, red-zone sequencing and bounces, which do not
   repeat. The process miss (stats-implied outcome vs model) is the part a team
   owns. A team's **noise-cancelled score** blends the two:

       observed = (1 - alpha) * raw miss + alpha * process miss

2. **Consistency prior.** Instead of shrinking every team toward the league
   average, shrink toward what its *process consistency* predicts: the
   game-to-game standard deviation of its success and explosive-play rates, on
   offence and defence. A team that sometimes stalls and sometimes does not is a
   team a model built on averages will miss. (Form *levels* -- how good, how fast
   -- were tested first and predict volatility no better than chance.)

       base     = pool + beta . z(consistency)
       expected = base + w * (observed - base),  w = n_eff / (n_eff + k)
       n_eff    = games this season + decay * games last season

3. **Earn it out of sample.** `fit` replays the production path and chooses
   alpha, beta, k and decay to predict the *second half* of a team's season from
   its first half and last season. Every season is scored with parameters chosen
   on the others (leave-one-season-out) and again on earlier seasons only
   (time-forward). Four nested variants are scored -- plain shrinkage, + noise
   cancellation, + consistency prior, both -- and a market is a trait only if
   its chosen variant beats "every team is average" overall and in a majority
   of seasons under *both* schemes.

A market that is not a trait is ranked on this season's observed misses and
labelled ``descriptive``: what happened, with no claim that it will continue.
"""

from __future__ import annotations

import html
import math
import pprint
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
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

# Shrinkage candidates in games. INFINITE_K is "no team signal" beyond the base.
K_GRID = (4.0, 10.0, 25.0, 40.0, 60.0, 100.0, 160.0, 250.0)
INFINITE_K = math.inf
DECAY_GRID = (0.0, 0.5, 1.0)
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
VARIANTS = {
    "plain": {"noise": False, "prior": False},
    "noise_cancelled": {"noise": True, "prior": False},
    "consistency_prior": {"noise": False, "prior": True},
    "full": {"noise": True, "prior": True},
}
# Ridge penalty on the consistency coefficients, as a share of total target weight.
STYLE_RIDGE = 0.05
# Game-to-game spread needs at least this many games to mean anything.
MIN_CONSISTENCY_GAMES = 3
# A team-season needs this many games before it can be split into an observed
# half and a target half for fitting.
MIN_SPLIT_GAMES = 6
# Time-forward scoring needs at least this many earlier team-seasons to fit on.
MIN_TRAIN_CASES = 50
# Grades are published only for a market that is a measured trait; within it,
# the top and bottom fifths of the ranked teams are labelled.
GRADE_SHARE = 0.2


@dataclass(frozen=True)
class GradedGame:
    """One completed game with the model's pre-game numbers, home perspective.

    ``*_process`` is each side's own offensive process line from that game (the
    defence is the other side's line). Optional: without them a game still
    scores raw misses, it just cannot feed noise cancellation or consistency.
    """

    season: int
    week: int
    home: str
    away: str
    model_margin: float
    actual_margin: float
    model_total: float | None = None
    actual_total: float | None = None
    home_process: dict | None = None
    away_process: dict | None = None


@dataclass(frozen=True)
class TeamGame:
    team: str
    season: int
    week: int
    losses: dict[str, float]
    # Same markets, scored against the stats-implied outcome instead of the
    # final score. Empty when the game has no process lines.
    process: dict[str, float]
    margin_residual: float
    total_residual: float | None
    win_probability: float
    won: float
    line: dict | None = None      # this team's offensive process line
    against: dict | None = None   # the opponent's, i.e. this team's defence


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# -- stats-implied outcome ------------------------------------------------------
def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting; `a` is square."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        if abs(m[col][col]) < 1e-12:
            continue
        for r in range(n):
            if r != col and m[r][col]:
                f = m[r][col] / m[col][col]
                m[r] = [x - f * y for x, y in zip(m[r], m[col])]
    return [m[i][n] / m[i][i] if abs(m[i][i]) > 1e-12 else 0.0 for i in range(n)]


def _ols(x: list[list[float]], y: list[float], *, weights: list[float] | None = None,
         ridge: float = 0.0, intercept: bool = True) -> list[float]:
    rows = [([1.0] if intercept else []) + r for r in x]
    w = weights or [1.0] * len(rows)
    p = len(rows[0])
    xtx = [[0.0] * p for _ in range(p)]
    xty = [0.0] * p
    for r, t, wt in zip(rows, y, w):
        for i in range(p):
            xty[i] += wt * r[i] * t
            for j in range(i, p):
                xtx[i][j] += wt * r[i] * r[j]
    for i in range(p):
        for j in range(i):
            xtx[i][j] = xtx[j][i]
        if ridge and not (intercept and i == 0):
            xtx[i][i] += ridge
    return _solve(xtx, xty)


@dataclass(frozen=True)
class OutcomeModel:
    """What a game's own process stats say the margin and total should have been.

    Margin is linear in the difference of the two offensive lines (one side's
    offence is the other's defence), total in their sum. The residual SD of the
    final margin around the implied margin is the game's luck, and sets how
    sharply an implied margin converts into an implied win.
    """

    features: tuple[str, ...]
    margin_coef: tuple[float, ...]   # intercept (home), then per feature
    total_coef: tuple[float, ...]    # intercept, then per feature
    luck_sd: float
    margin_r2: float
    total_r2: float

    def _vector(self, line: dict | None) -> list[float] | None:
        if not line:
            return None
        values = [line.get(f) for f in self.features]
        if any(v is None for v in values):
            return None
        return [float(v) for v in values]

    def implied(self, home: dict | None, away: dict | None) -> tuple[float, float] | None:
        h, a = self._vector(home), self._vector(away)
        if h is None or a is None:
            return None
        margin = self.margin_coef[0] + sum(
            c * (x - y) for c, x, y in zip(self.margin_coef[1:], h, a))
        total = self.total_coef[0] + sum(
            c * (x + y) for c, x, y in zip(self.total_coef[1:], h, a))
        return margin, total

    def to_json(self) -> dict:
        return {"features": list(self.features), "margin_coef": list(self.margin_coef),
                "total_coef": list(self.total_coef), "luck_sd": self.luck_sd,
                "margin_r2": self.margin_r2, "total_r2": self.total_r2}

    @classmethod
    def from_json(cls, data: dict | None) -> OutcomeModel | None:
        if not data:
            return None
        return cls(tuple(data["features"]), tuple(data["margin_coef"]),
                   tuple(data["total_coef"]), float(data["luck_sd"]),
                   float(data.get("margin_r2", 0.0)), float(data.get("total_r2", 0.0)))


def fit_outcome_model(games: list[GradedGame], features: tuple[str, ...]
                      ) -> OutcomeModel | None:
    usable = []
    for g in games:
        if g.actual_total is None or not g.home_process or not g.away_process:
            continue
        h = [g.home_process.get(f) for f in features]
        a = [g.away_process.get(f) for f in features]
        if any(v is None for v in h + a):
            continue
        usable.append((g, [float(v) for v in h], [float(v) for v in a]))
    if len(usable) < 10 * (len(features) + 1):
        return None
    diff = [[x - y for x, y in zip(h, a)] for _, h, a in usable]
    tot = [[x + y for x, y in zip(h, a)] for _, h, a in usable]
    margin_y = [float(g.actual_margin) for g, _, _ in usable]
    total_y = [float(g.actual_total) for g, _, _ in usable]
    mc = _ols(diff, margin_y)
    tc = _ols(tot, total_y)

    def r2(coef, x, y):
        fitted = [coef[0] + sum(c * v for c, v in zip(coef[1:], row)) for row in x]
        resid = [t - f for t, f in zip(y, fitted)]
        return resid, 1.0 - statistics.pvariance(resid) / statistics.pvariance(y)

    margin_resid, mr2 = r2(mc, diff, margin_y)
    _, tr2 = r2(tc, tot, total_y)
    return OutcomeModel(tuple(features), tuple(mc), tuple(tc),
                        statistics.pstdev(margin_resid), mr2, tr2)


def team_games(games: list[GradedGame], *, margin_sd: float,
               outcome: OutcomeModel | None = None) -> list[TeamGame]:
    """Both sides of every game, each scored from that team's perspective, on
    the final score and (where the process lines allow) on the stats-implied one."""
    out: list[TeamGame] = []
    for g in games:
        implied = outcome.implied(g.home_process, g.away_process) if outcome else None
        for team, sign, line, against in ((g.home, 1.0, g.home_process, g.away_process),
                                          (g.away, -1.0, g.away_process, g.home_process)):
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
            process: dict[str, float] = {}
            if implied is not None:
                implied_margin = sign * implied[0]
                q = _phi(implied_margin / outcome.luck_sd)
                process["spread"] = abs(implied_margin - projected)
                # Expected |won - p| if the game is won with probability q, on the
                # same zero-at-calibration scale as the raw term.
                process["moneyline"] = q * (1 - p) + (1 - q) * p - 2.0 * p * (1.0 - p)
                if g.model_total is not None:
                    implied_residual = implied[1] - float(g.model_total)
                    process["over"] = max(-implied_residual, 0.0)
                    process["under"] = max(implied_residual, 0.0)
            out.append(TeamGame(team, int(g.season), int(g.week), losses, process,
                                actual - projected, total_residual, p, won, line, against))
    return out


# -- shrinkage ------------------------------------------------------------------
def _weight(n_eff: float, k: float) -> float:
    if n_eff <= 0 or math.isinf(k):
        return 0.0
    return n_eff / (n_eff + k)


def _blend(raw: float, process: float, alpha: float) -> float:
    return (1.0 - alpha) * raw + alpha * process


@dataclass
class _Case:
    """One team-season split for fitting.

    Sums rather than lists, so a candidate (alpha, decay) is arithmetic, not a
    pass over games. Process sums fall back to raw for games without lines.
    """

    season: int
    n_first: int
    raw_first: float
    proc_first: float
    n_prior: int
    raw_prior: float
    proc_prior: float
    target: float          # mean ACTUAL loss over the second half
    n_target: int
    consistency: dict | None
    z: list[float] = field(default_factory=list)

    def observed(self, alpha: float, decay: float) -> tuple[float, float]:
        n_eff = self.n_first + decay * self.n_prior
        if n_eff <= 0:
            return 0.0, 0.0
        total = (_blend(self.raw_first, self.proc_first, alpha)
                 + decay * _blend(self.raw_prior, self.proc_prior, alpha))
        return total / n_eff, n_eff


def _loss(r: TeamGame, market: str, *, process: bool) -> float:
    if process and market in r.process:
        return r.process[market]
    return r.losses[market]


def consistency_names(stats: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{side}_sd_{s}" for side in ("off", "def") for s in stats)


def consistency(games: list[TeamGame], stats: tuple[str, ...]) -> dict | None:
    """Game-to-game SD of each process rate, on offence and on defence."""
    if not stats:
        return None
    out = {}
    for side, attr in (("off", "line"), ("def", "against")):
        for s in stats:
            values = [float(getattr(g, attr)[s]) for g in games
                      if getattr(g, attr) and getattr(g, attr).get(s) is not None]
            out[f"{side}_sd_{s}"] = (statistics.pstdev(values)
                                     if len(values) >= MIN_CONSISTENCY_GAMES else None)
    return out


def _cases(rows: list[TeamGame], market: str, stats: tuple[str, ...] = ()) -> list[_Case]:
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
        first, second = games[:half], games[half:]
        prior = by.get((team, previous[season]), []) if previous[season] else []
        out.append(_Case(
            season=season, n_first=len(first),
            raw_first=sum(_loss(g, market, process=False) for g in first),
            proc_first=sum(_loss(g, market, process=True) for g in first),
            n_prior=len(prior),
            raw_prior=sum(_loss(g, market, process=False) for g in prior),
            proc_prior=sum(_loss(g, market, process=True) for g in prior),
            target=statistics.fmean(g.losses[market] for g in second),
            n_target=len(second),
            # From the observed half only: what production knows when it ranks.
            consistency=consistency(first, stats),
        ))
    return out


def _standardise(rows: list[dict | None], features: tuple[str, ...]
                 ) -> list[list[float]]:
    """Z-scores within one population; a missing value sits at the mean (0)."""
    columns = []
    for f in features:
        values = [float(x[f]) for x in rows if x and x.get(f) is not None]
        mean = statistics.fmean(values) if values else 0.0
        sd = statistics.pstdev(values) if len(values) > 1 else 0.0
        columns.append((mean, sd))
    out = []
    for x in rows:
        row = []
        for f, (mean, sd) in zip(features, columns):
            v = (x or {}).get(f)
            row.append(0.0 if v is None or sd == 0 else (float(v) - mean) / sd)
        out.append(row)
    return out


def _prepare(cases: list[_Case], features: tuple[str, ...]) -> dict[int, list[_Case]]:
    by: dict[int, list[_Case]] = defaultdict(list)
    for c in cases:
        by[c.season].append(c)
    for season_cases in by.values():
        for c, z in zip(season_cases,
                        _standardise([c.consistency for c in season_cases], features)):
            c.z = z
    return by


def _season_terms(season_cases: list[_Case], alpha: float, decay: float, k: float):
    """Per case: (target - pool - w*(obs - pool), (1-w) * z, weight) and the pool."""
    obs = [c.observed(alpha, decay) for c in season_cases]
    den = sum(n for _, n in obs)
    pool = sum(o * n for o, n in obs) / den if den else 0.0
    terms = []
    for c, (o, n) in zip(season_cases, obs):
        w = _weight(n, k)
        resid = c.target - pool - (w * (o - pool) if n > 0 else 0.0)
        terms.append((resid, [(1.0 - w) * zi for zi in c.z], c.n_target))
    return terms


def _config_terms(by_season: dict[int, list[_Case]], alpha, decay, k):
    return [t for cs in by_season.values() for t in _season_terms(cs, alpha, decay, k)]


def _fit_beta(terms) -> list[float]:
    total_weight = sum(w for _, _, w in terms)
    return _ols([x for _, x, _ in terms], [y for y, _, _ in terms],
                weights=[w for _, _, w in terms],
                ridge=STYLE_RIDGE * total_weight, intercept=False)


def _sse(terms, beta: list[float] | None) -> float:
    if not beta:
        return sum(w * y * y for y, _, w in terms)
    return sum(w * (y - sum(b * xi for b, xi in zip(beta, x))) ** 2 for y, x, w in terms)


def _configs():
    for alpha in ALPHA_GRID:
        for decay in DECAY_GRID:
            for k in K_GRID + (INFINITE_K,):
                yield alpha, decay, k


def _best_per_variant(by_season: dict[int, list[_Case]], prior_ok: bool) -> dict[str, dict]:
    """Best configuration of each nested variant on the given seasons."""
    best: dict[str, dict] = {}
    for alpha, decay, k in _configs():
        terms = _config_terms(by_season, alpha, decay, k)
        plain = _sse(terms, None)
        beta = _fit_beta(terms) if prior_ok else None
        with_prior = _sse(terms, beta) if beta else None
        for name, spec in VARIANTS.items():
            if not spec["noise"] and alpha != 0.0:
                continue
            if spec["prior"] and with_prior is None:
                continue
            sse = with_prior if spec["prior"] else plain
            if name not in best or sse < best[name]["sse"]:
                best[name] = {"alpha": alpha, "decay": decay, "k": k, "sse": sse,
                              "beta": beta if spec["prior"] else None}
    return best


def _held_out(by_season: dict[int, list[_Case]], params: dict) -> float:
    terms = _config_terms(by_season, params["alpha"], params["decay"], params["k"])
    return _sse(terms, params.get("beta"))


def _baseline(by_season: dict[int, list[_Case]]) -> float:
    return _sse(_config_terms(by_season, 0.0, 0.0, INFINITE_K), None)


def _score(cases: list[_Case], features: tuple[str, ...], seasons: list[int], *,
           forward: bool) -> dict[str, dict]:
    """Held-out skill per variant: every season scored by parameters fitted on
    the others (or, `forward`, on earlier seasons only)."""
    prior_ok = bool(features)
    results = {name: {"sse": 0.0, "base": 0.0, "folds": []} for name in VARIANTS}
    for season in seasons:
        test = [c for c in cases if c.season == season]
        train = [c for c in cases if (c.season < season if forward else c.season != season)]
        if not test or len(train) < MIN_TRAIN_CASES:
            continue
        chosen = _best_per_variant(_prepare(train, features), prior_ok)
        test_by = _prepare(test, features)
        base = _baseline(test_by)
        for name, params in chosen.items():
            sse = _held_out(test_by, params)
            r = results[name]
            r["sse"] += sse
            r["base"] += base
            r["folds"].append({"season": season,
                               "skill": round(1.0 - sse / base, 4) if base else 0.0})
    out = {}
    for name, r in results.items():
        if not r["folds"]:
            continue
        skill = 1.0 - r["sse"] / r["base"] if r["base"] else 0.0
        helped = sum(f["skill"] > 0 for f in r["folds"])
        out[name] = {"skill": round(skill, 4), "seasons_helped": helped,
                     "seasons_tested": len(r["folds"]), "folds": r["folds"],
                     "passes": skill > 0 and helped * 2 > len(r["folds"])}
    return out


def _corr(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3 or statistics.pstdev(a) == 0 or statistics.pstdev(b) == 0:
        return None
    return statistics.correlation(a, b)


def _round(value: float | None, places: int = 4) -> float | None:
    return None if value is None else round(value, places)


def fit(games: list[GradedGame], *, margin_sd: float, source: str,
        process_features: tuple[str, ...] = (),
        consistency_stats: tuple[str, ...] = ()) -> dict:
    """Score the four variants per market under both schemes; ship the best
    variant that passes both, or call the market descriptive.

    `process_features` feed the stats-implied outcome; `consistency_stats` are
    the process rates whose game-to-game SD forms the consistency prior.
    """
    outcome = fit_outcome_model(games, process_features) if process_features else None
    rows = team_games(games, margin_sd=margin_sd, outcome=outcome)
    seasons = sorted({g.season for g in games})
    has_lines = bool(consistency_stats) and any(r.line for r in rows)
    stats = consistency_stats if has_lines else ()
    features = consistency_names(stats)
    markets = {}
    for market in MARKETS:
        cases = _cases(rows, market, stats)
        if not cases:
            continue
        loso = _score(cases, features, seasons, forward=False)
        forward = _score(cases, features, seasons, forward=True)
        # Variants without the inputs they need are not candidates.
        candidates = [v for v in VARIANTS
                      if v in loso and v in forward
                      and (outcome or not VARIANTS[v]["noise"])
                      and (features or not VARIANTS[v]["prior"])]
        passing = [v for v in candidates if loso[v]["passes"] and forward[v]["passes"]]
        chosen = max(passing, key=lambda v: loso[v]["skill"] + forward[v]["skill"]) \
            if passing else None
        final = _best_per_variant(_prepare(cases, features), bool(features))
        params = final[chosen] if chosen else None
        first = [c.raw_first / c.n_first for c in cases if c.n_first]
        target = [c.target for c in cases if c.n_first]
        markets[market] = {
            "trait": chosen is not None,
            "variant": chosen,
            "alpha": params["alpha"] if params else 0.0,
            "decay": params["decay"] if params else 0.0,
            "k": None if not params or math.isinf(params["k"]) else params["k"],
            "beta": ({f: round(b, 6) for f, b in zip(features, params["beta"])}
                     if params and params.get("beta") else None),
            "held_out_skill": (loso[chosen]["skill"] if chosen
                               else loso.get("plain", {}).get("skill")),
            "forward_skill": (forward[chosen]["skill"] if chosen
                              else forward.get("plain", {}).get("skill")),
            "seasons_helped": (loso[chosen] if chosen else loso.get("plain", {})
                               ).get("seasons_helped"),
            "seasons_tested": (loso[chosen] if chosen else loso.get("plain", {})
                               ).get("seasons_tested"),
            "variants": {v: {"loso": {k_: loso[v][k_] for k_ in
                                      ("skill", "seasons_helped", "seasons_tested", "passes")},
                             "forward": {k_: forward[v][k_] for k_ in
                                         ("skill", "seasons_helped", "seasons_tested",
                                          "passes")},
                             "loso_folds": loso[v]["folds"],
                             "forward_folds": forward[v]["folds"]}
                         for v in candidates},
            "split_half_r": _round(_corr(first, target)),
            "next_season_r": _round(_next_season(rows, market)),
            "team_seasons": len(cases),
        }
    return {
        "source": source,
        "seasons": seasons,
        "games": len(games),
        "margin_sd": margin_sd,
        "process_features": list(process_features),
        "consistency_stats": list(stats),
        "outcome_model": outcome.to_json() if outcome else None,
        "markets": markets,
        "prior_season": _season_aggregates(rows, seasons[-1]) if seasons else None,
    }


def summary(result: dict) -> list[str]:
    """Human-readable fit report: the ablation per market, both schemes."""
    lines = []
    outcome = result.get("outcome_model")
    if outcome:
        lines.append(f"  stats-implied outcome: margin R2 {outcome['margin_r2']:.3f}, "
                     f"total R2 {outcome['total_r2']:.3f}, luck SD {outcome['luck_sd']:.2f}")
    for market, row in result["markets"].items():
        lines.append(f"  {market:9s} trait={row['trait']!s:5s} variant={row['variant']} "
                     f"alpha={row['alpha']} k={row['k']} decay={row['decay']}")
        for name, v in row["variants"].items():
            lo, fw = v["loso"], v["forward"]
            lines.append(
                f"      {name:16s} LOSO {lo['skill']:+.4f} ({lo['seasons_helped']}/"
                f"{lo['seasons_tested']})  forward {fw['skill']:+.4f} "
                f"({fw['seasons_helped']}/{fw['seasons_tested']})"
                f"{'  PASS' if lo['passes'] and fw['passes'] else ''}")
    return lines


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


def _season_aggregates(rows: list[TeamGame], season: int) -> dict:
    """Per-team [games, summed raw loss, summed process loss] for one season, so
    production can carry it forward without shipping the replay itself."""
    teams: dict[str, dict[str, list]] = defaultdict(dict)
    for r in rows:
        if r.season != season:
            continue
        for market, value in r.losses.items():
            n, raw, proc = teams[r.team].get(market, [0, 0.0, 0.0])
            teams[r.team][market] = [n + 1, raw + value,
                                     proc + _loss(r, market, process=True)]
    return {"season": season,
            "teams": {t: {m: [n, round(raw, 4), round(proc, 4)]
                          for m, (n, raw, proc) in sorted(ms.items())}
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
    outcome = OutcomeModel.from_json(params.get("outcome_model"))
    rows = team_games(games, margin_sd=margin_sd, outcome=outcome)
    current = [r for r in rows if r.season == season]
    prior_rows = [r for r in rows if r.season == season - 1]
    prior: dict[str, dict[str, tuple[int, float, float]]] = defaultdict(dict)
    for r in prior_rows:
        for market, value in r.losses.items():
            n, raw, proc = prior[r.team].get(market, (0, 0.0, 0.0))
            prior[r.team][market] = (n + 1, raw + value,
                                     proc + _loss(r, market, process=True))
    stored = params.get("prior_season") or {}
    if not prior_rows and stored.get("season") == season - 1:
        for team, markets in (stored.get("teams") or {}).items():
            for market, values in markets.items():
                n, raw = int(values[0]), float(values[1])
                proc = float(values[2]) if len(values) > 2 else raw
                prior[team][market] = (n, raw, proc)

    by_team: dict[str, list[TeamGame]] = defaultdict(list)
    for r in current:
        by_team[r.team].append(r)
    everyone = set(by_team) | set(prior)
    universe = everyone & teams if teams is not None else everyone

    # Consistency from this season's games so far, standardised across the
    # field exactly as the fit standardised it within each season.
    features = consistency_names(tuple(params.get("consistency_stats") or ()))
    ordered = sorted(everyone)
    team_consistency = {t: consistency(by_team.get(t, []),
                                       tuple(params.get("consistency_stats") or ()))
                        for t in ordered}
    z_by_team = dict(zip(ordered, _standardise([team_consistency[t] for t in ordered],
                                               features))) if features else {}

    table: dict[str, dict] = {
        t: {"team": t, "games": len(by_team.get(t, [])),
            "prior_games": max((v[0] for v in prior.get(t, {}).values()), default=0),
            "markets": {}}
        for t in universe}
    out_markets = {}
    for market, mp in ((m, (params.get("markets") or {}).get(m) or {}) for m in MARKETS):
        trait = bool(mp.get("trait"))
        k = INFINITE_K if mp.get("k") is None or not trait else float(mp["k"])
        decay = float(mp.get("decay") or 0.0) if trait else 0.0
        alpha = float(mp.get("alpha") or 0.0) if trait else 0.0
        beta = mp.get("beta") if trait else None
        observed: dict[str, tuple[float, float]] = {}
        num = den = 0.0
        # Pool over every team with data, not just the ranked universe, so an
        # FBS team's FCS opponents still anchor the mean they were scored in.
        for team in everyone:
            cur = [r for r in by_team.get(team, []) if market in r.losses]
            n0, raw0, proc0 = prior.get(team, {}).get(market, (0, 0.0, 0.0))
            n_eff = len(cur) + decay * n0
            if n_eff <= 0:
                continue
            total = (sum(_blend(r.losses[market], _loss(r, market, process=True), alpha)
                         for r in cur) + decay * _blend(raw0, proc0, alpha))
            observed[team] = (total / n_eff, n_eff)
            num += total
            den += n_eff
        if not den:
            continue
        pool = num / den
        ranked = []
        for team in universe:
            if team not in observed:
                continue
            mean, n_eff = observed[team]
            base = pool
            prior_shift = 0.0
            if beta and team in z_by_team:
                prior_shift = sum(float(beta.get(f, 0.0)) * z
                                  for f, z in zip(features, z_by_team[team]))
                base += prior_shift
            w = _weight(n_eff, k)
            expected = base + w * (mean - base)
            cur = [r for r in by_team.get(team, []) if market in r.losses]
            raw_now = statistics.fmean(r.losses[market] for r in cur) if cur else None
            proc_cur = [r.process[market] for r in cur if market in r.process]
            entry = {
                "expected": round(expected, 4),
                "observed": _round(raw_now),
                # Noise-cancelled score: this season's miss against what each
                # game's own process stats implied, i.e. with the luck removed.
                # A diagnostic -- the forecast uses it only in proportion alpha.
                "noise_cancelled": _round(statistics.fmean(proc_cur)) if proc_cur else None,
                # Raw minus process: how much of this season's miss was luck.
                "luck": (_round(statistics.fmean(
                    r.losses[market] - r.process[market] for r in cur
                    if market in r.process)) if proc_cur else None),
                "consistency_shift": round(prior_shift, 4),
                "reliability": round(w, 3),
                "games": len(cur),
            }
            if market == "moneyline":
                entry["upsets"] = sum(1 for r in cur
                                      if (r.win_probability - 0.5) * (r.won - 0.5) < 0)
                entry["expected_upsets"] = round(sum(
                    min(r.win_probability, 1.0 - r.win_probability) for r in cur), 2)
            table[team]["markets"][market] = entry
            # A trait ranks on the forecast. A market with no measured team
            # signal ranks on what this season actually did and says so.
            key = expected if trait else raw_now
            if key is not None:
                ranked.append((key, -n_eff, team))
        ranked.sort()
        band = max(1, round(len(ranked) * GRADE_SHARE)) if trait else 0
        for position, (_, _, team) in enumerate(ranked, start=1):
            entry = table[team]["markets"][market]
            entry["rank"] = position
            entry["grade"] = (None if not trait else "steady" if position <= band
                              else "volatile" if position > len(ranked) - band
                              else "typical")
        out_markets[market] = {
            "label": LABELS[market],
            "unit": UNITS[market],
            "pool": round(pool, 4),
            "k": None if math.isinf(k) else k,
            "decay": decay,
            "alpha": alpha,
            "consistency_prior": bool(beta),
            "variant": mp.get("variant"),
            "trait": trait,
            "basis": "predictive" if trait else "descriptive",
            "held_out_skill": mp.get("held_out_skill"),
            "forward_skill": mp.get("forward_skill"),
            "seasons_helped": mp.get("seasons_helped"),
            "seasons_tested": mp.get("seasons_tested"),
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

    current_games = [g for g in games if g.season == season]
    return {
        "season": season,
        "through_week": max((r.week for r in current), default=0),
        "graded_games": len({(g.week, g.home, g.away) for g in current_games}),
        "process_games": sum(1 for g in current_games if outcome
                             and outcome.implied(g.home_process, g.away_process)),
        "fit_source": params.get("source"),
        "fit_seasons": params.get("seasons"),
        "outcome_model": ({"margin_r2": outcome.margin_r2, "total_r2": outcome.total_r2,
                           "luck_sd": outcome.luck_sd} if outcome else None),
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
                                         "process_features", "consistency_stats",
                                         "outcome_model", "prior_season")}
    lean["markets"] = {m: {k: v for k, v in row.items() if k != "variants"}
                       for m, row in result["markets"].items()}
    body = pprint.pformat(lean, indent=1, width=96, sort_dicts=True)
    text = ('"""Generated by scripts/fit_volatility.py -- do not edit by hand.\n\n'
            'Evidence, with per-season folds: reports/volatility_fit.json.\n"""\n\n'
            f"FIT = {body}\n")
    Path(path).write_text(text, encoding="utf-8")


def attach(games: list[GradedGame], *,
           process: dict[tuple[int, int, str, str], dict] | None = None) -> list[GradedGame]:
    """Join each game to both sides' process lines, keyed (season, week, team,
    opponent). Ledger games carry no box stats; this is how they get them."""
    if not process:
        return games
    out = []
    for g in games:
        out.append(GradedGame(**{**g.__dict__,
                                 "home_process": process.get((g.season, g.week, g.home, g.away)),
                                 "away_process": process.get((g.season, g.week, g.away, g.home))}))
    return out


def build(snapshots: list[dict], *, season: int, margin_sd: float,
          teams: set[str] | None = None,
          process: dict[tuple[int, int, str, str], dict] | None = None,
          ) -> dict:
    """The production ranking from a shadow ledger's snapshots."""
    games = attach(graded_from_ledger(snapshots), process=process)
    return rank(games, season=season, params=params(), margin_sd=margin_sd,
                teams=teams)


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
TITLE = "Team Volatility Rankings"


def _esc(value) -> str:
    return html.escape(str(value), quote=True)


def _value(market: str, entry: dict, predictive: bool) -> str:
    if market == "moneyline" and not predictive:
        if entry.get("games"):
            return f"{entry['upsets']} vs {entry['expected_upsets']:.1f}"
        return "&ndash;"
    value = entry["expected"] if predictive else entry.get("observed")
    if value is None:
        return "&ndash;"
    return f"{value:+.3f}" if market == "moneyline" else f"{value:.1f}"


def _noise_cancelled(market: str, entry: dict) -> str:
    value = entry.get("noise_cancelled")
    if value is None:
        return "&ndash;"
    return f"{value:+.3f}" if market == "moneyline" else f"{value:.1f}"


def _team_cell(team: str, logo) -> str:
    src = logo(team) if logo else None
    img = f'<img src="{_esc(src)}" alt="" loading="lazy">' if src else ""
    return f'<div class="vol-team">{img}<span>{_esc(team)}</span></div>'


def _note(info: dict, payload: dict) -> str:
    seasons = payload.get("fit_seasons") or []
    span = f"{seasons[0]}&ndash;{seasons[-1]}" if seasons else "the replay"
    skill = info.get("held_out_skill")
    forward = info.get("forward_skill")
    skill_text = f"{skill:+.1%}" if skill is not None else "n/a"
    forward_text = f"{forward:+.1%}" if forward is not None else "n/a"
    helped, tested = info.get("seasons_helped"), info.get("seasons_tested")
    record = f", better in {helped} of {tested} seasons" if tested else ""
    if info.get("trait"):
        parts = []
        if info.get("alpha"):
            parts.append(f"noise cancellation strips {info['alpha']:.0%} of the luck "
                         "(final score vs what the game&rsquo;s stats implied)")
        if info.get("consistency_prior"):
            parts.append("shrinks toward what each team&rsquo;s game-to-game process "
                         "consistency predicts rather than the league average")
        if info.get("k") is None:
            parts.append("gives a team&rsquo;s own miss record no weight: past misses "
                         "did not predict, its process consistency did")
        else:
            parts.append(f"trusts a team&rsquo;s own record by {info['k']:g} games of "
                         "evidence" + (f", last season at {info['decay']:.2g}&times;"
                                       if info.get("decay") else ""))
        return (f"A measured team trait: held-out {span} seasons predicted second-half "
                f"volatility {skill_text} better than calling every team average"
                f"{record}; {forward_text} fitting on earlier seasons only. The forecast "
                + "; ".join(parts) + ".")
    return (f"Not a team trait: in {span}, a team&rsquo;s past volatility did not reliably "
            f"predict its future volatility, with or without noise cancellation and the "
            f"consistency prior (held-out {skill_text}{record}). Ranked on this season&rsquo;s "
            "misses only &mdash; what happened, not a forecast.")


def _market_card(market: str, info: dict, teams: list[dict], payload: dict, logo) -> str:
    predictive = bool(info.get("trait"))
    rows = sorted((t for t in teams if "rank" in t["markets"].get(market, {})),
                  key=lambda t: t["markets"][market]["rank"])
    if not rows:
        return ""
    n = min(SHOWN_PER_END, len(rows) // 2) or len(rows)
    column = ("forecast" if predictive
              else "upsets vs exp." if market == "moneyline" else "pts")

    def block(chunk: list[dict], label: str) -> str:
        body = "".join(
            f'<tr><td class="n">{t["markets"][market]["rank"]}</td>'
            f'<td>{_team_cell(t["team"], logo)}</td>'
            f'<td class="r">{_value(market, t["markets"][market], predictive)}</td>'
            f'<td class="r n">{_noise_cancelled(market, t["markets"][market])}</td>'
            f'<td class="r n">{t["markets"][market]["games"]}</td></tr>'
            for t in chunk)
        return (f'<div class="vol-sub">{label}</div><table class="vol-t"><thead><tr>'
                f'<th>#</th><th>Team</th><th class="r">{column}</th>'
                f'<th class="r" title="noise-cancelled score">NC</th>'
                f'<th class="r">G</th></tr></thead><tbody>{body}</tbody></table>')

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
    outcome = payload.get("outcome_model") or {}
    luck_line = (f" <b>NC</b> is the noise-cancelled score: this season&rsquo;s miss "
                 f"measured against what each game&rsquo;s own process stats implied instead "
                 f"of the final score (those stats explain {outcome['margin_r2']:.0%} of final "
                 f"margins and {outcome['total_r2']:.0%} of totals; the rest is turnovers, "
                 f"sequencing and bounces). A team whose NC sits well below its raw miss "
                 f"was volatile mostly through luck." if outcome else "")
    blurb = f"""How reliably each team&rsquo;s results land where our
metrics project them, scored against the model&rsquo;s own pre-game margin, total and win
probability &mdash; not the market&rsquo;s. Rank 1 conforms most; the bottom of each list
produces the outcomes the metrics did not see coming. <b>Spread</b> is points missed per
game. <b>Moneyline</b> counts losses as the model&rsquo;s favourite plus wins as its
underdog, against how many its probabilities expected. <b>Over</b> and <b>Under</b> are
scored on the miss that loses each bet: how far games fell short of, or ran past, the
projected total.{luck_line} Through week {_esc(payload.get('through_week'))}:
{_esc(payload.get('graded_games'))} graded games this season. Research context, not a
betting signal."""
    cards = "".join(_market_card(m, payload["markets"][m], teams, payload, logo)
                    for m in MARKETS if m in payload["markets"])
    return f"""<section id="volatility">
{head(eyebrow, TITLE, blurb)}
<div class="vol-grid">{cards}</div>
{_full_table(teams, payload, logo)}
</section>"""
