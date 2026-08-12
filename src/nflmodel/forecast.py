"""Produce NFL game probabilities and the authority attached to them.

The forecast is market-anchored, exactly as `nfl-genesis` measured:

    logit(p) = logit(market_fair) + lam * (logit(structural) - logit(market_fair))

`lam` is NOT fitted here. It is read from the research repo's published evidence,
which selected 0.000 in all five folds from 2021 onward. At lam = 0 the forecast is
identical to the paired no-vig market, and that is the honest output — the
structural component carried no incremental information over a closing line.

That makes this module look almost trivial today, and it should. The value is that
the shape is correct and auditable: when a challenger in nfl-genesis finally moves
lam off zero, only `DEFAULT_LAMBDA` and the evidence pointer change, and every
consumer inherits the improvement together with the promotion state.
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
