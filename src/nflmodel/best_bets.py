"""Weekly best bets with a written angle, logged and graded like everything else.

The watchlist in `recommendations.py` ranks raw gaps and explains each with the
same scheme sentence. This turns the strongest of them into picks a reader can
act on or ignore, each with the angle that produced it:

* **Spreads / totals** - where the model's margin or total disagrees with
  DraftKings by more than its noise floor. The angle names which path drives it
  (power ratings vs this season's efficiency, which the model blends 50/50), the
  fitted starting-QB adjustment when one applies, the official injury report for
  both teams, and the scheme matchup.
* **Player props** - the fitted prop distribution (`prop_distributions`) against
  the DraftKings line, de-vigged. The angle carries the player's role and depth
  slot, the scheme context, the team's implied points and the injury report.

**Honesty.** The research harness selected a market weight of 0 for NFL margins
(nfl-genesis, 2021-2025) and ATS on disagreements is 49.8%: the model has not
shown it beats the close. So a pick's probability here is the *model's own*
view, labelled as such, and the page shows the picks' live graded record beside
them. Authority stays RESEARCH_ONLY.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from . import prop_distributions, teams

# Market error around the outcome: margin MAE ~10.5, total MAE ~10.3 on the
# 2016-2025 baseline -> Normal SD = MAE * sqrt(pi/2).
MARGIN_SIGMA = 13.2
TOTAL_SIGMA = 12.9
MIN_SPREAD_GAP = 2.0
MIN_TOTAL_GAP = 3.0
MAX_GAP = 14.0            # beyond this the "gap" is a data problem, not a pick
MIN_PROP_PROBABILITY = 0.55
MIN_PROP_EDGE = 0.04
LIMITS = {"spread": 5, "total": 3, "prop": 6}
DEFAULT_PRICE = -110

PROP_MARKETS = {
    "player_pass_yds": ("passing_yards", "passing yards"),
    "player_pass_tds": ("passing_tds", "passing TDs"),
    "player_pass_attempts": ("pass_attempts", "pass attempts"),
    "player_rush_yds": ("rushing_yards", "rushing yards"),
    "player_rush_attempts": ("rush_attempts", "rush attempts"),
    "player_receptions": ("receptions", "receptions"),
    "player_reception_yds": ("receiving_yards", "receiving yards"),
}
_KEY_POSITIONS = ("QB", "WR", "RB", "TE", "T", "G", "C", "OL", "EDGE", "DE", "DT", "LB",
                  "CB", "S")
_UNAVAILABLE = {"out", "doubtful"}


@dataclass
class Pick:
    family: str
    season: int
    week: int
    home: str
    away: str
    kickoff: str | None
    selection: str
    side: str                   # home | away | over | under
    line: float
    price: int
    model_number: float
    book_number: float
    edge: float                 # points for games, probability over book for props
    probability: float          # the model's own view
    angle: str
    tags: list[str] = field(default_factory=list)
    player: str | None = None
    player_id: str | None = None
    team: str | None = None
    metric: str | None = None

    @property
    def pick_id(self) -> str:
        parts = [str(self.season), str(self.week), self.family, self.away, self.home]
        if self.family == "prop":
            parts += [self.player_id or self.player or "", self.metric or ""]
        return "|".join(parts)

    def to_json(self) -> dict:
        out = asdict(self)
        out["pick_id"] = self.pick_id
        return out


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _implied(price) -> float | None:
    try:
        price = float(price)
    except (TypeError, ValueError):
        return None
    if price == 0:
        return None
    return 100.0 / (price + 100.0) if price > 0 else -price / (-price + 100.0)


def _line(value: float) -> str:
    return "PK" if abs(value) < 0.05 else f"{value:+.1f}"


def _by(margin: float, home: str, away: str) -> str:
    """A home margin as words: 'KC by 3.5'."""
    if abs(margin) < 0.05:
        return "even"
    return f"{home if margin > 0 else away} by {abs(margin):.1f}"


def injury_index(rows: list[dict], week: int) -> dict[str, list[tuple[str, str, str]]]:
    """team -> [(position, name, status)] for players Out/Doubtful/Questionable."""
    out: dict[str, list[tuple[str, str, str]]] = {}
    for row in rows or []:
        try:
            if int(float(row.get("week") or 0)) != week:
                continue
        except (TypeError, ValueError):
            continue
        status = str(row.get("report_status") or "").strip()
        if status.lower() not in {"out", "doubtful", "questionable"}:
            # Formal designations arrive Friday; midweek the signal is practice.
            # A rest day is not an injury.
            practice = str(row.get("practice_status") or "")
            reason = str(row.get("practice_primary_injury") or "")
            if not practice.startswith("Did Not Participate") or "Not injury related" in reason:
                continue
            status = "DNP in practice"
        team = teams.canonical(row.get("team") or "")
        out.setdefault(team, []).append(
            (str(row.get("position") or ""), str(row.get("full_name") or ""), status))
    for listed in out.values():
        listed.sort(key=lambda r: (_KEY_POSITIONS.index(r[0]) if r[0] in _KEY_POSITIONS
                                   else len(_KEY_POSITIONS), r[2] != "Out", r[1]))
    return out


def _injury_sentence(team: str, injuries: dict, *, limit: int = 3) -> str | None:
    listed = [r for r in injuries.get(team, []) if r[2].lower() in _UNAVAILABLE]
    practice = [r for r in injuries.get(team, [])
                if r[2] == "DNP in practice" and r[0] in ("QB", "WR", "RB", "TE")]
    parts = []
    if listed:
        named = ", ".join(f"{pos} {name} ({status})" for pos, name, status in listed[:limit])
        more = len(listed) - limit
        parts.append(f"{team} has {len(listed)} ruled out or doubtful: {named}"
                     + (f" +{more} more" if more > 0 else ""))
    if practice:
        parts.append(f"{team} " + ", ".join(f"{pos} {name}" for pos, name, _ in practice[:limit])
                     + " missed practice with an injury")
    return "; ".join(parts) or None


def _scheme(slate, team: str, opponent: str) -> str | None:
    from .recommendations import _profile_reason

    reason = _profile_reason(slate, team, opponent)
    return None if reason.startswith("Scheme profile unavailable") else reason


def spread_pick(projection, slate, injuries: dict) -> Pick | None:
    p = projection
    if p.model_margin is None or p.book_margin is None:
        return None
    gap = p.model_margin - p.book_margin
    if not MIN_SPREAD_GAP <= abs(gap) <= MAX_GAP:
        return None
    home_side = gap > 0
    team, opp = (p.home, p.away) if home_side else (p.away, p.home)
    line = -p.book_margin if home_side else p.book_margin
    fav = p.home if p.model_margin > 0 else p.away
    sentences = [f"The model makes it {fav} by {abs(p.model_margin):.1f}; DraftKings has "
                 f"{team} {_line(line)}, a {abs(gap):.1f}-point disagreement toward {team}."]
    if p.rating_margin is not None and p.efficiency_margin is not None:
        sentences.append(
            f"Power ratings make it {_by(p.rating_margin, p.home, p.away)}; this season's "
            f"efficiency makes it {_by(p.efficiency_margin, p.home, p.away)} "
            "(the model blends the two).")
    if p.qb_out:
        sentences.append(f"Starting QB out: {', '.join(p.qb_out)} - the fitted "
                         f"{abs(p.availability_margin):.1f}-point adjustment is applied.")
    support = _injury_sentence(opp, injuries)
    risk = _injury_sentence(team, injuries)
    if support:
        sentences.append(f"Injury report: {support}.")
    if risk:
        sentences.append(f"Risk: {risk}.")
    scheme = _scheme(slate, team, opp)
    if scheme:
        sentences.append(f"Matchup: {scheme}")
    prob = _phi(abs(gap) / MARGIN_SIGMA)
    tags = (["qb-out"] if p.qb_out else []) + (["injury-report"] if support else [])
    return Pick("spread", p.season, p.week, p.home, p.away, p.kickoff_utc or None,
                f"{team} {_line(line)}", "home" if home_side else "away", line,
                DEFAULT_PRICE, round(-p.model_margin if home_side else p.model_margin, 1),
                line, round(abs(gap), 1), round(prob, 3), " ".join(sentences), tags)


def total_pick(projection, slate, injuries: dict) -> Pick | None:
    p = projection
    if p.projected_total is None or p.book_total is None:
        return None
    gap = p.projected_total - p.book_total
    if not MIN_TOTAL_GAP <= abs(gap) <= MAX_GAP:
        return None
    over = gap > 0
    sentences = [f"The model projects {p.projected_total:.1f} against a DraftKings total of "
                 f"{p.book_total:.1f} ({abs(gap):.1f} {'over' if over else 'under'})."]
    if p.projected_home_score is not None and p.projected_away_score is not None:
        sentences.append(f"Scoreline: {p.away} {p.projected_away_score:.1f}, "
                         f"{p.home} {p.projected_home_score:.1f}.")
    skill = []
    for team in (p.away, p.home):
        out = [r for r in injuries.get(team, [])
               if r[2].lower() in _UNAVAILABLE and r[0] in ("QB", "WR", "RB", "TE")]
        if out:
            skill.append(f"{team} without " + ", ".join(f"{pos} {n}" for pos, n, _ in out[:3]))
    if skill:
        sentences.append("Skill-position injuries: " + "; ".join(skill) + ".")
    if p.qb_out:
        sentences.append(f"Starting QB out: {', '.join(p.qb_out)}.")
    for team, opp in ((p.away, p.home), (p.home, p.away)):
        scheme = _scheme(slate, team, opp)
        if scheme:
            sentences.append(f"{team} offense: {scheme}")
    prob = _phi(abs(gap) / TOTAL_SIGMA)
    return Pick("total", p.season, p.week, p.home, p.away, p.kickoff_utc or None,
                f"{'Over' if over else 'Under'} {p.book_total:.1f}", "over" if over else "under",
                p.book_total, DEFAULT_PRICE, round(p.projected_total, 1), p.book_total,
                round(abs(gap), 1), round(prob, 3), " ".join(sentences),
                ["qb-out"] if p.qb_out else [])


def prop_picks(slate, injuries: dict) -> list[Pick]:
    from .sources.oddsapi import normalise

    players = {normalise(pl.player_name): pl for pl in slate.player_projections
               if str(pl.injury_status or "").lower() not in _UNAVAILABLE}
    games = {(g.home, g.away): g for g in slate.projections}
    out: list[Pick] = []
    for quote in slate.player_prop_quotes:
        spec = PROP_MARKETS.get(quote.market)
        player = players.get(normalise(quote.player_name))
        if spec is None or player is None:
            continue
        metric, label = spec
        mean = player.metrics.get(metric)
        if metric == "rush_attempts" and mean is None:
            mean = player.metrics.get("carries")
        dist = prop_distributions.distribution(metric, mean)
        if not dist:
            continue
        p_over = prop_distributions.over_probability(dist, quote.line)
        if p_over is None:
            continue
        over_imp, under_imp = _implied(quote.over_price), _implied(quote.under_price)
        if over_imp and under_imp:
            total = over_imp + under_imp
            over_imp, under_imp = over_imp / total, under_imp / total
        for side, prob, implied, price in (("over", p_over, over_imp, quote.over_price),
                                           ("under", 1.0 - p_over, under_imp, quote.under_price)):
            if implied is None or prob < MIN_PROP_PROBABILITY or prob - implied < MIN_PROP_EDGE:
                continue
            home = player.team if player.home else player.opponent
            away = player.opponent if player.home else player.team
            game = games.get((home, away))
            sentences = [
                f"Projects {dist['mean']:.1f} {label} (middle 80%: {dist['p10']:g}-"
                f"{dist['p90']:g}) against {quote.line:g}; {prob:.0%} to go {side} versus "
                f"{implied:.0%} priced in."]
            sentences.append(f"Role: {player.depth_slot or player.position}, "
                             f"{player.role_continuity}; {player.role_reason}.")
            context = player.scheme_context or {}
            bits = []
            if context.get("target_multiplier"):
                bits.append(f"target share x{float(context['target_multiplier']):.2f} "
                            "against this defense")
            if context.get("pass_attempt_delta"):
                bits.append(f"{float(context['pass_attempt_delta']):+.1f} team pass attempts")
            if context.get("carry_delta"):
                bits.append(f"{float(context['carry_delta']):+.1f} team carries")
            if bits:
                sentences.append("Scheme: " + ", ".join(bits) + ".")
            if player.implied_team_points is not None:
                sentences.append(f"{player.team} implied for {player.implied_team_points:.1f}.")
            teammates = [f"{pos} {n}" for pos, n, status in injuries.get(player.team, [])
                         if status.lower() in _UNAVAILABLE
                         and pos == player.position and n != player.player_name]
            if teammates:
                sentences.append(f"Same-position teammates out: {', '.join(teammates[:3])}.")
            if player.injury_status:
                sentences.append(f"Note: {player.player_name} is listed {player.injury_status}.")
            out.append(Pick(
                "prop", player.season, player.week, home, away,
                player.kickoff_utc or (game.kickoff_utc if game else None),
                f"{player.player_name} {side} {quote.line:g} {label}", side, quote.line,
                int(price), round(dist["mean"], 1), quote.line, round(prob - implied, 3),
                round(prob, 3), " ".join(sentences),
                ["questionable"] if player.injury_status else [],
                player=player.player_name, player_id=player.player_id, team=player.team,
                metric=metric))
    return out


def build(slate) -> list[Pick]:
    injuries = injury_index(getattr(slate, "injuries", []) or [], slate.week)
    picks: list[Pick] = []
    for projection in slate.projections:
        for maker in (spread_pick, total_pick):
            pick = maker(projection, slate, injuries)
            if pick is not None:
                picks.append(pick)
    picks += prop_picks(slate, injuries)
    out: list[Pick] = []
    for family, limit in LIMITS.items():
        seen: set[str] = set()
        for pick in sorted((p for p in picks if p.family == family),
                           key=lambda p: (-p.probability, -p.edge)):
            key = pick.player_id if family == "prop" else f"{pick.away}@{pick.home}"
            if key in seen:
                continue
            seen.add(key)
            out.append(pick)
            if len(seen) >= limit:
                break
    return out
