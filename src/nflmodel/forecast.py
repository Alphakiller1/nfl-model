"""Two forecasts, one authority.

**Probabilities** (`forecast_game`, `forecast_slate`) are market-anchored, exactly
as `nfl-genesis` measured:

    logit(p) = logit(market_fair) + lam * (logit(structural) - logit(market_fair))

`lam` is not fitted here. It is read from the research repo's published evidence,
which selected 0.000 in all five folds from 2021 onward, so the probability
forecast equals the paired no-vig market by construction. This is the contract
`profit-priority` consumes and its shape must not change casually.

**Points** (`project_game`) are new, and they are a different claim. A win
probability anchored to the market says nothing about *how* a game gets there, so
this repo now also produces an opponent-adjusted margin, total and scoreline from
`matrix.py`, `ratings.py` and `totals.py`. Those are model output, not market
output, and they are published as such.

What the points model is worth, measured leave-one-season-out on 2,383 games
(2017-2025), is stated on every board rather than buried:

    margin: model 10.3134   market 9.8708   ATS on disagreements 49.8%
    total:  model 10.8076   market 10.4700  O/U on disagreements 49.1%

So the margin is published **anchored to the market** at `SPREAD_LAMBDA = 0`, for
the same reason the probability is: it does not beat the closing line. The model's
own number is shown beside it so the disagreement is visible, and that
disagreement is reported as a `market_gap`, never as an `edge`. Calling it an edge
would assert the model has found something, when a 49.8% ATS record says it has
not.

The **projected scoreline is built from the model's margin, not the published
one.** At lam = 0 the published margin is the market, and a "projected score"
silently derived from it would be the market's projection wearing the model's
label.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from . import authority as auth
from .market import PairedQuote, prob_to_american

# Published selection from nfl-genesis, folds 2021-2025. Do not tune here; tune in
# the research harness where there is an out-of-sample gate to catch it.
DEFAULT_LAMBDA = 0.0
EVIDENCE = (
    "nfl-genesis reports/EMPIRICAL_BASELINE_2016_2025.md "
    "(market-anchored stacker, selected lam = 0.000 for 2021-2025)"
)


def _logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def _expit(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def anchor(market_fair: float, structural: float | None,
           lam: float = DEFAULT_LAMBDA) -> float:
    """Shrink a structural view toward the market. lam = 0 returns the market."""
    if structural is None or not 0.0 < structural < 1.0 or lam <= 0.0:
        return market_fair
    return _expit(_logit(market_fair) + lam * (_logit(structural) - _logit(market_fair)))


@dataclass(frozen=True)
class GameForecast:
    game: str
    home_team: str
    away_team: str
    home_fair: float
    away_fair: float
    home_american: int
    away_american: int
    overround: float
    lam: float
    structural_home: float | None
    edge_vs_market: float
    action: str
    authority: str
    unmet_gates: int
    evidence: str

    def to_dict(self) -> dict:
        return asdict(self)


def forecast_game(
    *,
    game: str,
    home_team: str,
    away_team: str,
    home_american: float,
    away_american: float,
    structural_home: float | None = None,
    lam: float = DEFAULT_LAMBDA,
    authority_override: auth.Authority | None = None,
) -> GameForecast:
    """One game: paired price in, fair probability plus permitted action out."""
    a = authority_override or auth.current()
    quote = PairedQuote(home_american=home_american, away_american=away_american)
    market_fair = quote.home_fair
    home_fair = anchor(market_fair, structural_home, lam)
    edge = home_fair - market_fair
    return GameForecast(
        game=game,
        home_team=home_team,
        away_team=away_team,
        home_fair=round(home_fair, 6),
        away_fair=round(1.0 - home_fair, 6),
        home_american=prob_to_american(home_fair),
        away_american=prob_to_american(1.0 - home_fair),
        overround=round(quote.overround, 6),
        lam=lam,
        structural_home=structural_home,
        edge_vs_market=round(edge, 6),
        action=a.action_for(edge, has_price=True).value,
        authority=a.level.value,
        unmet_gates=len(a.unmet_gates),
        evidence=EVIDENCE,
    )


def forecast_slate(games: list[dict], lam: float = DEFAULT_LAMBDA) -> dict:
    """Forecast a slate and emit the contract downstream consumers read.

    Games missing a paired price are reported as AVOID rather than dropped —
    a silently shorter list is indistinguishable from a slate with no games.
    """
    a = auth.current()
    rows, skipped = [], []
    for g in games:
        try:
            rows.append(
                forecast_game(
                    game=str(g.get("game") or f"{g.get('away_team')}@{g.get('home_team')}"),
                    home_team=str(g["home_team"]),
                    away_team=str(g["away_team"]),
                    home_american=float(g["home_american"]),
                    away_american=float(g["away_american"]),
                    structural_home=g.get("structural_home"),
                    lam=lam,
                    authority_override=a,
                ).to_dict()
            )
        except (KeyError, TypeError, ValueError) as exc:
            skipped.append({
                "game": g.get("game") or f"{g.get('away_team')}@{g.get('home_team')}",
                "action": auth.Action.AVOID.value,
                "reason": f"{type(exc).__name__}: {exc}",
            })
    return {
        "schema": "nfl-model/forecast/1",
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "authority": a.level.value,
        "may_bet": a.may_bet,
        "unmet_gates": list(a.unmet_gates),
        "lam": lam,
        "evidence": a.evidence,
        "games": rows,
        "skipped": skipped,
        "note": (
            "lam = 0 means this forecast equals the paired no-vig market by "
            "construction. It is a calibrated price, not an edge."
        ) if lam == 0.0 else "",
    }


def write_slate(games: list[dict], destination: str,
                lam: float = DEFAULT_LAMBDA) -> str:
    payload = forecast_slate(games, lam)
    from pathlib import Path
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


# ── points: margin, total and the scoreline ──────────────────────────────────
# Fraction of the model's disagreement with the closing spread that is kept in
# the published margin. Zero, and that is a measurement rather than caution: the
# model's MAE is 10.3134 against the market's 9.8708 and its ATS record on
# disagreements is 1158-1165-60 (49.85%, 95% CI [47.81%, 51.88%]) against a
# 52.38% breakeven. Raising it is a claim about evidence and belongs with a gate
# record, not a config tweak.
SPREAD_LAMBDA = 0.0

POINTS_EVIDENCE = (
    "reports/BASELINE_2016_2025.md — leave-one-season-out on 2,383 games "
    "(2017-2025): margin MAE 10.3134 vs market 9.8708, ATS 49.85%"
)


@dataclass(frozen=True)
class GameProjection:
    """One game's points forecast, with everything needed to audit it."""

    home: str
    away: str
    neutral: bool
    season: int = 0
    week: int = 0
    kickoff: str = ""
    # The two components of the model margin, kept separate so a breakdown can
    # show which one is doing the work.
    rating_margin: float | None = None
    efficiency_margin: float | None = None
    model_margin: float | None = None
    market_margin: float | None = None
    # What the board publishes. Equals the market at SPREAD_LAMBDA = 0.
    margin: float | None = None
    win_probability: float | None = None
    # Model minus market. Always a gap, never an edge, while the ATS record
    # straddles 50% — see `_points_edge`.
    market_gap: float | None = None
    edge_points: float | None = None
    edge_withheld_reason: str | None = None
    projected_total: float | None = None
    market_total: float | None = None
    projected_home_score: float | None = None
    projected_away_score: float | None = None
    total_modelled: bool = False
    home_moneyline: float | None = None
    away_moneyline: float | None = None
    market_fair_home: float | None = None
    home_form: object | None = None
    away_form: object | None = None
    used_efficiency: bool = False
    action: str = auth.Action.AVOID.value
    authority: str = auth.Level.RESEARCH_ONLY.value

    @property
    def has_price(self) -> bool:
        return self.market_margin is not None

    @property
    def total_gap(self) -> float | None:
        """Model total minus market total, or None without both."""
        if self.projected_total is None or self.market_total is None:
            return None
        return self.projected_total - self.market_total


def _points_edge(market_gap: float | None, *, used_efficiency: bool
                 ) -> tuple[float | None, str | None]:
    """Decide whether the model-minus-market difference may be called an edge.

    It may not, and the reason is a measurement rather than a policy. Across
    2,383 out-of-sample games the model's margin MAE is 0.44 points worse than
    the closing line and its ATS record on disagreements is 49.85% with a 95%
    interval of [47.81%, 51.88%] — an interval that contains 50% and sits
    entirely below the 52.38% breakeven. Both estimators are calibrated (slope
    1.035); the market simply conditions on more, chiefly injuries and
    availability this repo does not model.

    So the difference between the two numbers is dominated by what the market
    knows and the model does not. It stays visible as `market_gap` and is not
    called an edge. When a challenger moves the ATS interval above breakeven,
    this function is where that changes.
    """
    if market_gap is None:
        return None, None
    if not used_efficiency:
        return None, "no opponent-adjusted form — rating prior only"
    return None, "model does not beat the closing line (ATS 49.85%) — gap, not edge"


def project_game(
    *,
    home: str,
    away: str,
    team_ratings: dict[str, float],
    home_form=None,
    away_form=None,
    neutral: bool = False,
    market_margin: float | None = None,
    market_total: float | None = None,
    home_moneyline: float | None = None,
    away_moneyline: float | None = None,
    season: int = 0,
    week: int = 0,
    kickoff: str = "",
    lam: float = SPREAD_LAMBDA,
    authority: auth.Authority | None = None,
) -> GameProjection:
    """Forecast one game's points. `market_margin` is the expected HOME margin."""
    from . import ratings as ratings_mod
    from . import totals as totals_mod

    a = authority or auth.current()
    rating_margin = ratings_mod.projected_margin(team_ratings, home, away, neutral=neutral)
    projection = totals_mod.project(home_form, away_form, rating_margin=rating_margin,
                                    neutral=neutral)
    model_margin = projection.margin
    used_efficiency = projection.modelled

    if market_margin is None:
        published, market_gap = model_margin, None
    else:
        market_gap = None if model_margin is None else model_margin - market_margin
        published = market_margin if market_gap is None else market_margin + lam * market_gap

    edge, withheld = _points_edge(market_gap, used_efficiency=used_efficiency)
    win_p = None if published is None else ratings_mod.win_probability(published)

    market_fair = None
    if home_moneyline is not None and away_moneyline is not None:
        try:
            market_fair = PairedQuote(home_american=home_moneyline,
                                      away_american=away_moneyline).home_fair
        except (TypeError, ValueError):
            market_fair = None

    return GameProjection(
        home=home, away=away, neutral=neutral, season=season, week=week, kickoff=kickoff,
        rating_margin=rating_margin,
        efficiency_margin=(matrix_margin(home_form, away_form, neutral)
                           if used_efficiency else None),
        model_margin=model_margin,
        market_margin=market_margin,
        margin=published,
        win_probability=win_p,
        market_gap=market_gap,
        edge_points=edge,
        edge_withheld_reason=withheld,
        projected_total=projection.total,
        market_total=market_total,
        projected_home_score=projection.home_score,
        projected_away_score=projection.away_score,
        total_modelled=projection.modelled,
        home_moneyline=home_moneyline,
        away_moneyline=away_moneyline,
        market_fair_home=market_fair,
        home_form=home_form,
        away_form=away_form,
        used_efficiency=used_efficiency,
        action=a.action_for(edge, market_margin is not None,
                            modelled=model_margin is not None,
                            implausible=auth.IMPLAUSIBLE_EDGE_POINTS).value,
        authority=a.level.value,
    )


def matrix_margin(home_form, away_form, neutral: bool) -> float | None:
    """The efficiency component on its own, for the breakdown."""
    from . import matrix
    if home_form is None or away_form is None:
        return None
    return matrix.margin_points(home_form, away_form, neutral=neutral)
