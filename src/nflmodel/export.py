"""The JSON contract downstream systems read.

`profit-priority` and the content engine consume this rather than scraping the
dashboard, so the shape is a promise. Two rules follow from that:

* **The authority travels with the numbers.** Every payload carries the level,
  the unmet gates and the evidence pointer, and every game carries the action its
  authority permits. A consumer that reads a margin without reading the
  permission has to do so deliberately.
* **Absences are explicit.** A game the model cannot rate is emitted with null
  fields and a reason, never dropped. A silently shorter list is
  indistinguishable from a shorter slate.

`schema` is versioned; add fields freely, rename or remove them only with a
version bump.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from . import authority as auth
from . import divisions as divisions_mod
from . import forecast, matrix, ratings, teams

SCHEMA = "nfl-model/board/2"


def _round(value, places: int = 3):
    return None if value is None else round(float(value), places)


def _form(form) -> dict | None:
    if form is None:
        return None
    out = {field: _round(getattr(form, field), 5) for field in matrix.TeamForm.FIELDS}
    out["plays"] = _round(form.plays, 2)
    out["offense_index"] = _round(matrix.offense_index(form))
    out["defense_index"] = _round(matrix.defense_index(form))
    out["efficiency_rating"] = _round(matrix.efficiency_rating(form))
    return out


def _projection(p: forecast.GameProjection) -> dict:
    return {
        "away": p.away,
        "home": p.home,
        "kickoff": p.kickoff,
        "neutral": p.neutral,
        "rating_margin": _round(p.rating_margin, 2),
        "efficiency_margin": _round(p.efficiency_margin, 2),
        "model_margin": _round(p.model_margin, 2),
        "market_margin": _round(p.market_margin, 2),
        "published_margin": _round(p.margin, 2),
        "market_gap": _round(p.market_gap, 2),
        # Always null while the model does not beat the closing line. Present as a
        # key so a consumer can check for it rather than infer it from absence.
        "edge_points": _round(p.edge_points, 2),
        "edge_withheld_reason": p.edge_withheld_reason,
        # Both probabilities travel. `win_probability` is the published price
        # (the market at lambda 0); `model_win_probability` is the model's own
        # view, and it is the one that agrees with `model_margin`.
        "win_probability": _round(p.win_probability, 4),
        "model_win_probability": _round(p.model_win_probability, 4),
        "projected_total": _round(p.projected_total, 2),
        "market_total": _round(p.market_total, 2),
        "projected_home_score": _round(p.projected_home_score, 1),
        "projected_away_score": _round(p.projected_away_score, 1),
        "total_modelled": p.total_modelled,
        "home_moneyline": _round(p.home_moneyline, 0),
        "away_moneyline": _round(p.away_moneyline, 0),
        "market_fair_home": _round(p.market_fair_home, 4),
        "action": p.action,
        "authority": p.authority,
    }


def payload(slate, outlooks: list | None = None) -> dict:
    """The whole week: authority, ratings, units, games and season outlook."""
    authority = slate.authority
    ranked = ratings.rank_table(slate.table)
    outlook_by_team = {o.team: o for o in (outlooks or [])}

    team_rows = []
    for rank, team, rating in ranked:
        form = slate.forms.get(team)
        meta = teams.get(team)
        outlook = outlook_by_team.get(team)
        team_rows.append({
            "rank": rank,
            "team": team,
            "name": meta.name,
            "conference": meta.conference,
            "division": meta.division,
            "rating": _round(rating, 2),
            "form": _form(form),
            "projected_wins": _round(getattr(outlook, "projected_wins", None), 2),
            "win_division": _round(getattr(outlook, "win_division", None), 4),
            "make_playoffs": _round(getattr(outlook, "make_playoffs", None), 4),
            "top_seed": _round(getattr(outlook, "top_seed", None), 4),
        })

    champions = {
        name: {"team": o.team, "probability": _round(o.win_division, 4)}
        for name, o in divisions_mod.projected_champions(outlooks or []).items()
    }

    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "season": slate.season,
        "week": slate.week,
        "authority": authority.level.value,
        "may_bet": authority.may_bet,
        "unmet_gates": list(authority.unmet_gates),
        "gates_total": len(auth.REQUIRED_GATES),
        "evidence": authority.evidence,
        "spread_lambda": forecast.SPREAD_LAMBDA,
        "moneyline_lambda": forecast.DEFAULT_LAMBDA,
        "note": (
            "The published margin equals the market at spread_lambda = 0. "
            "model_margin is the model's own number and market_gap is their "
            "difference; that difference is not an edge and edge_points is null."
        ),
        "constants": {
            "home_field_rating_path": ratings.HOME_FIELD_POINTS,
            "home_field_efficiency_path": matrix.COEFFICIENTS["home_field"],
            "margin_residual_sd": ratings.MARGIN_SD,
            "blowout_cap": ratings.BLOWOUT_CAP,
            "matrix_lineage": matrix.LINEAGE_VERSION,
            "matrix_status": matrix.STATUS,
        },
        "teams": team_rows,
        "games": [_projection(p) for p in slate.projections],
        "division_winners": champions,
        "simulations": divisions_mod.SIMULATIONS if outlooks else 0,
    }


def write(slate, destination: str | Path, outlooks: list | None = None) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload(slate, outlooks), indent=2) + "\n",
                    encoding="utf-8")
    return path
