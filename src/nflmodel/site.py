"""Self-contained static dashboard for GitHub Pages -- Chase Analytics design contract.

Visual identity is vendored byte-identical from mlb-model (`static/chase_tokens.css`
plus `static/board.css`) and the slate is rendered through the shared board kernel,
so an NFL card has the same anatomy, palette and typefaces as an MLB or WNBA one.
`tests/test_board_contract.py` fails the build if a vendored file drifts. Everything
this module adds -- ratings, unit rankings, division outlooks, methodology -- is
built from tokens only; `_PAGE_CSS` must never introduce a colour literal of its own.

**The page leads with the authority gate rather than the numbers.** This model does
not beat the closing line, and a dashboard that showed prices first and permissions
in a footnote would misrepresent exactly the thing the gate exists to prevent.
`tests/test_site.py` asserts the ordering, because the temptation to move the
pretty section up is permanent.

The sections after the board answer questions a game card cannot: which teams are
good (`ratings`), which half of each team is doing it (`units`), and what that
implies over a whole schedule (`divisions`). All three read the same
`season.Slate`, so the board, the rankings and the division odds cannot disagree.
"""
from __future__ import annotations

import html
from datetime import UTC, datetime
from pathlib import Path

from . import authority as auth
from . import divisions as divisions_mod
from . import forecast, matrix, ratings, teams, totals
from . import season as season_mod
from .board import BOARD_JS, board_html
from .board_nfl import build_board

_STATIC = Path(__file__).resolve().parent / "static"

_FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;0,9..40,800&"
    "family=Oswald:ital,wght@0,600;0,700;0,900;1,600;1,700;1,900&"
    "family=Roboto+Condensed:wght@400;500;600;700;800&display=swap');"
)

e = html.escape

# Measured on 2,383 out-of-sample games. Kept in one place so the hero tiles, the
# methodology table and the module docstrings cannot drift apart.
EVIDENCE = {
    "games": 2383,
    "seasons": "2017-2025",
    "margin_model": 10.3134,
    "margin_market": 9.8708,
    "margin_ratings_only": 10.4298,
    "ats": (1158, 1165, 60),
    "ats_rate": 0.4985,
    "ats_ci": (0.4782, 0.5188),
    "total_model": 10.8076,
    "total_market": 10.4700,
    "total_league_mean": 11.0423,
    "ou_rate": 0.4909,
    "margin_slope": 1.035,
    "total_slope": 0.907,
}


def brand_css() -> str:
    """Fonts + chase_tokens.css + the board kernel -- the shared Chase identity."""
    tokens = (_STATIC / "chase_tokens.css").read_text(encoding="utf-8")
    board = (_STATIC / "board.css").read_text(encoding="utf-8")
    return _FONT_IMPORT + tokens + board


def _fmt(value, places: int = 1, sign: bool = True) -> str:
    if value is None:
        return "&ndash;"
    if sign:
        return season_mod.signed(value, places)
    return f"{value:.{places}f}"


def _logo(abbr: str, size: int = 22) -> str:
    team = teams.get(abbr)
    return (f'<img class="tlogo" src="{e(team.logo)}" alt="" loading="lazy" '
            f'width="{size}" height="{size}">')


# ── metric grading ───────────────────────────────────────────────────────────
# The Chase 7-step gradient, applied by league percentile rather than by an
# absolute threshold. An absolute cut ("EPA over 0.10 is elite") ages badly as
# the league's scoring environment moves; a percentile is stable by construction
# and is what the tokens were designed for.
_GRADE_STEPS = (
    (0.857, "metric-elite"), (0.714, "metric-strong"), (0.571, "metric-above"),
    (0.429, "metric-neutral"), (0.286, "metric-below"), (0.143, "metric-weak"),
)


def _grade(value: float, population: list[float], *, higher_is_better: bool) -> str:
    if not population:
        return "metric-neutral"
    below = sum(1 for other in population if other < value)
    percentile = below / len(population)
    if not higher_is_better:
        percentile = 1.0 - percentile
    for threshold, name in _GRADE_STEPS:
        if percentile >= threshold:
            return name
    return "metric-very-weak"


def _chip(value: float | None, population: list[float], *, higher_is_better: bool,
          places: int = 3, percent: bool = False) -> str:
    if value is None:
        return '<td class="num dim">&ndash;</td>'
    grade = _grade(value, population, higher_is_better=higher_is_better)
    shown = f"{value * 100:.1f}%" if percent else f"{value:.{places}f}"
    return f'<td class="num"><span style="color:var(--{grade})">{shown}</span></td>'


# ── fragments ────────────────────────────────────────────────────────────────
def _nav(slate) -> str:
    return f"""
<header class="chase-header">
  <nav class="chase-nav wrap">
    <a href="https://chase-analytics.com" class="chase-logo" title="Chase Analytics">
      <svg viewBox="0 0 36 36" width="30" height="30" aria-hidden="true">
      <path d="M18 5 C21 13 24 20 33 31 L3 31 C12 20 15 13 18 5 Z" fill="#7C4DFF"/></svg>
      <span class="chase-wordmark">CHASE&nbsp;<em>ANALYTICS</em></span>
    </a>
    <div class="nav-links">
      <a class="nav-link" href="#authority">Authority</a>
      <a class="nav-link" href="#board">Board</a>
      <a class="nav-link" href="#ratings">Power Ratings</a>
      <a class="nav-link" href="#units">Offense &amp; Defense</a>
      <a class="nav-link" href="#divisions">Divisions</a>
      <a class="nav-link" href="#methodology">Method</a>
    </div>
    <div class="chase-status"><span class="product-tag">NFL MODEL</span>
    <span class="pill dim">{slate.season} &middot; WK {slate.week}</span></div>
  </nav>
</header>"""


def _hero(slate, built_at: str) -> str:
    regime = ("in-season form" if slate.in_season
              else "preseason prior &mdash; no 2026 games played yet")
    return f"""
<header class="hero">
  <div class="wrap">
    <div class="hero-eyebrow">CHASE ANALYTICS&ensp;|&ensp;MODEL LAB</div>
    <h1 class="hero-title">National Football League<br>projections, and the
    permission to use them.</h1>
    <p class="hero-sub">Opponent-adjusted power ratings, projected scores and totals for
    every game, offensive and defensive unit rankings, and a simulated race for all eight
    divisions &mdash; behind an explicit gate on what the numbers are allowed to be used for.
    This model does not beat the closing market, and says so here rather than hiding it.</p>
    <div class="hero-meta">
      <span class="pill warn">{e(slate.authority.level.value)}</span>
      <span class="pill">{len(slate.games)} games</span>
      <span class="pill">{slate.rated_teams} teams rated</span>
      <span class="pill">{regime}</span>
      <span class="pill dim">built {e(built_at)}</span>
    </div>
  </div>
</header>"""


def _tiles() -> str:
    wins, losses, pushes = EVIDENCE["ats"]
    low, high = EVIDENCE["ats_ci"]
    data = [
        (f"{EVIDENCE['margin_model']:.4f}", "Model margin MAE",
         f"points per game, {EVIDENCE['seasons']} out of sample"),
        (f"{EVIDENCE['margin_market']:.4f}", "Market margin MAE",
         "closing spread on the same games"),
        (f"{EVIDENCE['ats_rate']:.2%}", "ATS on disagreements",
         f"{wins}-{losses}-{pushes} &middot; 95% CI [{low:.1%}, {high:.1%}] "
         f"&middot; breakeven 52.38%"),
        (f"{EVIDENCE['games']:,}", "Games evaluated",
         "leave-one-season-out, point-in-time"),
    ]
    return '<div class="tiles">' + "".join(
        f'<div class="tile"><span class="tile-v">{value}</span>'
        f'<span class="tile-l">{label}</span><span class="tile-n">{note}</span></div>'
        for value, label, note in data
    ) + "</div>"


def _authority_section(authority) -> str:
    satisfied = [g for g in auth.REQUIRED_GATES if g not in authority.unmet_gates]
    rows = "".join(
        f'<tr><td class="gate-ok">&#10003;</td><td>{e(gate.replace("_", " "))}</td></tr>'
        for gate in satisfied
    ) + "".join(
        f'<tr><td class="gate-no">&#9675;</td><td>{e(gate.replace("_", " "))}</td></tr>'
        for gate in authority.unmet_gates
    )
    return f"""
<section id="authority">
  <div class="sec-head">
    <span class="kicker">Authority &middot; 01</span>
    <h2>What these numbers may be used for</h2>
    <p class="blurb">A probability and a permission are different things, and conflating
    them is how an unpromoted model becomes a bet. Measured out of sample this model is
    <b>0.44 points worse</b> than the closing spread and covers <b>49.85%</b> of its own
    disagreements &mdash; an interval that contains 50% and sits entirely below the 52.38%
    breakeven. It is therefore <b>{e(authority.level.value)}</b> and may never emit a bet.</p>
  </div>
  <div class="gate-grid">
    <div class="gate-card"><span class="gate-k">Authority</span>
      <span class="gate-v">{e(authority.level.value)}</span></div>
    <div class="gate-card"><span class="gate-k">May bet</span>
      <span class="gate-v">{"yes" if authority.may_bet else "no"}</span></div>
    <div class="gate-card"><span class="gate-k">Gates met</span>
      <span class="gate-v">{len(satisfied)} / {len(auth.REQUIRED_GATES)}</span></div>
    <div class="gate-card"><span class="gate-k">Spread &lambda;</span>
      <span class="gate-v">{forecast.SPREAD_LAMBDA:.3f}</span></div>
  </div>
  {_tiles()}
  <div class="tablewrap"><table class="gates">
    <thead><tr><th></th><th>Production gate</th></tr></thead>
    <tbody>{rows}</tbody></table></div>
  <p class="note">{e(authority.evidence)}</p>
</section>"""


def _board_section(slate) -> str:
    rows_by_key = {f"{teams.canonical(r['away_team'])}@{teams.canonical(r['home_team'])}": r
                   for r in slate.games}
    board = build_board(slate, rows_by_key=rows_by_key)
    priced = sum(1 for p in slate.projections if p.market_margin is not None)
    return f"""
<section id="board">
  <div class="sec-head">
    <span class="kicker">Slate &middot; 02</span>
    <h2>Week {slate.week} board</h2>
    <p class="blurb">Projected scores are the <b>model&rsquo;s</b>, built from its own margin
    and total &mdash; not from the published margin, which at &lambda;&nbsp;=&nbsp;0 is just the
    market wearing the model&rsquo;s label. Each card shows the market number beside it so the
    disagreement is visible; that disagreement is reported as a <b>gap</b>, never an edge.
    The <b>Picks</b> counter counts priced markets ({priced} of {len(slate.games)} games have
    one) &mdash; it is not a count of recommendations, and this authority permits none.</p>
  </div>
  {board_html(board)}
</section>"""


def _ratings_section(slate, outlooks) -> str:
    wins_by_team = {o.team: o.projected_wins for o in outlooks}
    table = ratings.rank_table(slate.table)
    span = max((abs(value) for _, _, value in table), default=1.0) or 1.0
    rows = []
    for rank, team, value in table:
        form = slate.forms.get(team)
        offense = matrix.offense_index(form) if form else None
        defense = matrix.defense_index(form) if form else None
        efficiency = matrix.efficiency_rating(form) if form else None
        meta = teams.get(team)
        width = min(abs(value) / span, 1.0) * 50.0
        offset = 50.0 if value >= 0 else 50.0 - width
        tone = "rb-pos" if value >= 0 else "rb-neg"
        projected = wins_by_team.get(team)
        rows.append(
            f'<tr><td class="rank">{rank}</td>'
            f'<td class="team"><span class="tname">{_logo(team)}'
            f'<b>{e(team)}</b><span class="dim">{e(meta.short)}</span></span></td>'
            f'<td class="dim">{e(meta.division)}</td>'
            f'<td class="num score">{value:+.2f}</td>'
            f'<td class="num">{_fmt(offense, 2)}</td>'
            f'<td class="num">{_fmt(defense, 2)}</td>'
            f'<td class="num eff">{_fmt(efficiency, 2)}</td>'
            f'<td class="num">{f"{projected:.1f}" if projected is not None else "&ndash;"}</td>'
            f'<td class="ratingbar"><div class="rb-track"><span class="rb-zero"></span>'
            f'<i class="{tone}" style="left:{offset:.1f}%;width:{width:.1f}%"></i>'
            f"</div></td></tr>")
    return f"""
<section id="ratings">
  <div class="sec-head">
    <span class="kicker">Ratings &middot; 03</span>
    <h2>Power ratings</h2>
    <p class="blurb">Two independent estimates of team strength, side by side. <b>Rating</b>
    is the opponent-adjusted scoring-margin solve: points relative to an average team on a
    neutral field, so the projected neutral margin is the difference of two ratings and the
    host adds {ratings.HOME_FIELD_POINTS:.2f}. <b>Off</b> and <b>Def</b> come from the separate
    matchup model, and <b>Eff</b> is exactly their sum. The two columns are <i>not</i> meant to
    be equal &mdash; the published margin is the 50/50 blend of them, and a team where they
    disagree sharply is a team whose record and whose per-play efficiency tell different
    stories, which is worth seeing rather than averaging away. <b>Proj W</b> is the mean of
    {divisions_mod.SIMULATIONS:,} simulated seasons over the real 2026 fixture list, not a
    league-average schedule.</p>
  </div>
  <div class="tablewrap"><table class="pr">
    <thead><tr><th>#</th><th>Team</th><th>Division</th><th class="num">Rating</th>
    <th class="num">Off</th><th class="num">Def</th><th class="num">Eff</th>
    <th class="num">Proj W</th><th>vs average</th></tr></thead>
    <tbody>{''.join(rows)}</tbody></table></div>
</section>"""


def _unit_table(slate, *, offense: bool) -> str:
    """One ranked unit table -- offence or defence -- with graded component rates."""
    prefix = "off" if offense else "def"
    index_of = matrix.offense_index if offense else matrix.defense_index
    entries = []
    for team, form in slate.forms.items():
        value = index_of(form)
        if value is not None:
            entries.append((value, team, form))
    entries.sort(reverse=True)

    populations = {
        stat: [getattr(form, f"{prefix}_{stat}") for _, _, form in entries]
        for stat in matrix.STATS
    }
    # Direction, per unit. An offence wants a high EPA and a low sack rate; a
    # defence wants the mirror, because `def_*` records what the OPPONENT did.
    better_high = {
        "epa": offense, "first_down": offense, "explosive": offense,
        "sack": not offense, "turnover": not offense,
    }
    rows = []
    for rank, (value, team, form) in enumerate(entries, start=1):
        cells = "".join(
            _chip(getattr(form, f"{prefix}_{stat}"), populations[stat],
                  higher_is_better=better_high[stat],
                  places=3, percent=stat != "epa")
            for stat in matrix.STATS
        )
        rows.append(
            f'<tr><td class="rank">{rank}</td>'
            f'<td class="team"><span class="tname">{_logo(team, 20)}<b>{e(team)}</b></span></td>'
            f'<td class="num score">{value:+.2f}</td>{cells}</tr>')
    title = "Offense" if offense else "Defense"
    # Headers are abbreviated because these two tables sit side by side; the full
    # meaning is in the section blurb and in each column's own sign convention.
    head = "EPA/pl" if offense else "EPA/pl a"
    first = "1st dn" if offense else "1st dn a"
    explosive = "Expl" if offense else "Expl a"
    sack = "Sack%" if offense else "Sack%"
    turnover = "Give%" if offense else "Take%"
    return f"""<div class="unit">
<div class="unit-h">{title} power ranking</div>
<div class="tablewrap"><table class="pr unit-tbl">
<thead><tr><th>#</th><th>Team</th><th class="num">Pts</th><th class="num">{head}</th>
<th class="num">{first}</th><th class="num">{explosive}</th><th class="num">{sack}</th>
<th class="num">{turnover}</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div></div>"""


def _units_section(slate) -> str:
    return f"""
<section id="units">
  <div class="sec-head">
    <span class="kicker">Units &middot; 04</span>
    <h2>Offensive &amp; defensive power rankings</h2>
    <p class="blurb"><b>Pts</b> is points per game above an average unit, on the same scale as
    the overall rating &mdash; an offence at +3.1 and a defence at +1.4 add to a +4.5 team, and
    they add exactly, because both come from the one matchup fit rather than from three models
    that disagree at the edges. Component rates are opponent-adjusted and coloured by league
    percentile. A defensive row records what <em>opponents</em> did, so a low EPA allowed and a
    high sack rate are both good defence; the colouring already accounts for that.</p>
  </div>
  <div class="units">{_unit_table(slate, offense=True)}{_unit_table(slate, offense=False)}</div>
</section>"""


def _division_card(name: str, members) -> str:
    rows = []
    for outlook in members:
        # Abbreviation only. The full nickname is the widest cell in the card and
        # it is what pushed the division grid past the viewport; the logo beside
        # the abbreviation already identifies the team.
        rows.append(
            f'<tr><td class="team"><span class="tname">{_logo(outlook.team, 20)}'
            f'<b>{e(outlook.team)}</b></span></td>'
            f'<td class="num">{outlook.projected_wins:.1f}&#8209;'
            f'{outlook.projected_losses:.1f}</td>'
            f'<td class="oddsbar"><div class="ob-track">'
            f'<i style="width:{outlook.win_division * 100:.1f}%"></i></div></td>'
            f'<td class="num score">{outlook.win_division:.0%}</td>'
            f'<td class="num dim">{outlook.make_playoffs:.0%}</td></tr>')
    champion = max(members, key=lambda o: o.win_division) if members else None
    crown = (f'<span class="dv-pick">{_logo(champion.team, 18)}'
             f'{e(champion.team)} &middot; {champion.win_division:.0%}</span>'
             if champion else "")
    return (f'<div class="dv"><div class="dv-h">{e(name)}{crown}</div>'
            f'<div class="dv-wrap"><table class="dv-tbl">'
            f'<thead><tr><th>Team</th><th class="num">Rec</th><th></th>'
            f'<th class="num">Div</th><th class="num">PO</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div></div>')


def _divisions_section(outlooks) -> str:
    grouped = divisions_mod.by_division(outlooks)
    cards = "".join(_division_card(name, members) for name, members in grouped.items()
                    if members)
    if not cards:
        cards = ('<div class="empty">No simulated season available &mdash; the schedule '
                 "feed returned no rateable games.</div>")
    return f"""
<section id="divisions">
  <div class="sec-head">
    <span class="kicker">Season &middot; 05</span>
    <h2>Projected division winners</h2>
    <p class="blurb">Every regular-season game on the real 2026 fixture list, replayed
    {divisions_mod.SIMULATIONS:,} times from the same margins the board publishes. Division
    ties are broken by head-to-head, then division record, then conference record, then a coin
    flip &mdash; the later NFL tiebreakers are not implemented, and ordering by power rating
    instead would quietly hand every tie to the favourite. Ratings do not change during a
    simulated season, nobody gets injured, and draws are independent, so these tails are
    thinner than reality&rsquo;s.</p>
    <p class="note"><b>Rec</b> is the mean simulated record, <b>Div</b> the chance of
    winning the division, <b>PO</b> the chance of reaching the playoffs. The bar is
    <b>Div</b> drawn to scale. Division chances sum to 100% within each division and
    playoff chances to seven per conference, by construction.</p>
  </div>
  <div class="dvs">{cards}</div>
</section>"""


def _method_section() -> str:
    coefficients = "".join(
        f"<tr><td>{e(matrix.FEATURE_LABELS.get(k, k).replace('_', ' '))}</td>"
        f'<td class="num">{v:+.3f}</td></tr>'
        for k, v in matrix.COEFFICIENTS.items() if k in matrix.STATS)
    offense = "".join(f"<tr><td>{e(k.replace('_', ' '))}</td>"
                      f'<td class="num">{v:.4f}</td></tr>'
                      for k, v in matrix.OFFENSE_WEIGHTS.items())
    defense = "".join(f"<tr><td>{e(k.replace('_', ' '))}</td>"
                      f'<td class="num">{v:.4f}</td></tr>'
                      for k, v in matrix.DEFENSE_WEIGHTS.items())
    return f"""
<section id="methodology">
  <div class="sec-head">
    <span class="kicker">Method &middot; 06</span>
    <h2>How the numbers are built</h2>
    <p class="blurb">Every constant below was measured on {EVIDENCE['games']:,}
    out-of-sample games rather than assumed, and the sweeps that came back flat are labelled
    as flat. Reproduce with <code>python scripts/fit_matrix.py</code>.</p>
  </div>
  <div class="mth">
    <div class="mth-card"><div class="mth-h">The matchup model</div>
      <p>One fit produces everything: margin, total, scoreline and both unit rankings.
      It predicts the points <em>one</em> team scores, from that team&rsquo;s offence and its
      opponent&rsquo;s defence.</p>
      <pre><code>points(A vs B) = intercept
   + &Sigma; g&#8339; &middot; ( off&#8339;(A) + def&#8339;(B) )
   + home_field</code></pre>
      <p>Offence and defence share a coefficient because they measure the same per-play
      quantity from opposite sides. Given them free coefficients instead and
      <code>def_epa</code> fits <b>positive</b> &mdash; allowing more EPA per play
      &ldquo;improves&rdquo; the margin &mdash; because EPA and first-down rate correlate at
      r&nbsp;=&nbsp;0.83 and the pair splits the effect with opposite signs. The symmetric form
      costs 0.016 points of MAE and every sign it produces is the one football says it should
      be.</p>
      <table class="mth-tbl">{coefficients}
      <tr><td>Home field (fitted)</td>
      <td class="num">{matrix.COEFFICIENTS['home_field']:+.2f}</td></tr></table></div>

    <div class="mth-card"><div class="mth-h">Offense weights</div>
      <p>Standardised importance &mdash; |coefficient| &times; the spread of that rate across
      teams &mdash; normalised within the group.</p>
      <table class="mth-tbl">{offense}</table></div>

    <div class="mth-card"><div class="mth-h">Defense weights</div>
      <p>Not the offence table relabelled. The coefficients are shared; the spreads are not.
      Takeaways separate defences (0.222) more than giveaways separate offences (0.182), and
      EPA separates offences more than defences &mdash; the fitted version of an old scouting
      claim.</p>
      <table class="mth-tbl">{defense}</table></div>

    <div class="mth-card"><div class="mth-h">Power ratings</div>
      <p>Opponent-adjusted scoring margin, solved iteratively, home field removed before
      rating so a soft home schedule earns nothing.</p>
      <p>Margins pass through <code>cap &middot; tanh(margin / cap)</code>, but the cap barely
      matters here: 7.0% of NFL games are decided by 28+ against 36% in FBS, and the sweep moves
      MAE by 0.004 points from cap 17 to no cap at all. It is kept to bound a freak result, not
      because it contributes.</p>
      <table class="mth-tbl">
      <tr><td>Home field, rating path</td>
      <td class="num">{ratings.HOME_FIELD_POINTS:.2f}</td></tr>
      <tr><td>Blowout cap</td><td class="num">{ratings.BLOWOUT_CAP:.0f}</td></tr>
      <tr><td>Recency half-life</td>
      <td class="num">{ratings.RECENCY_HALFLIFE_WEEKS:.0f} wks</td></tr>
      <tr><td>Margin residual SD</td><td class="num">{ratings.MARGIN_SD:.2f}</td></tr></table>
      <p class="fine">Measured mean home margin is +2.06 (2021&ndash;2025). The rating path
      publishes 1.20 because the published margin is a 50/50 blend with the efficiency path,
      which carries its own fitted +1.74 &mdash; an effective
      <b>{(ratings.HOME_FIELD_POINTS + matrix.COEFFICIENTS['home_field']) / 2:.2f}</b>. Using
      the measured 2.06 here would double-count.</p></div>

    <div class="mth-card"><div class="mth-h">Totals &amp; scorelines</div>
      <p>The raw model total is over-dispersed &mdash; regressing actual on predicted gives a
      slope near 0.63 &mdash; so it is shrunk {1 - totals.TOTAL_SHRINK:.0%} toward the league
      mean, which lifts the slope to {EVIDENCE['total_slope']:.2f}.</p>
      <table class="mth-tbl">
      <tr><td>League mean total</td><td class="num">{EVIDENCE['total_league_mean']:.4f}</td></tr>
      <tr><td>Model total</td><td class="num">{EVIDENCE['total_model']:.4f}</td></tr>
      <tr><td>Market total</td><td class="num">{EVIDENCE['total_market']:.4f}</td></tr>
      <tr><td>O/U on disagreements</td><td class="num">{EVIDENCE['ou_rate']:.2%}</td></tr>
      <tr><td>Total residual SD</td><td class="num">{totals.TOTAL_SD:.2f}</td></tr></table>
      <p class="fine">Scores are algebra: home = (total + margin) / 2. They inherit the error
      of <em>both</em> models, so read a scoreline as a centre of mass. A total marked
      <b>*</b> is the league mean, not a modelled figure.</p></div>

    <div class="mth-card"><div class="mth-h">Point-in-time</div>
      <p>A game in week <em>W</em> is forecast only from completed games before <em>W</em>,
      blended with earlier seasons on a shrinkage of
      {int(__import__("nflmodel.preseason", fromlist=["x"]).BLEND_K)} games. Coefficients are
      fitted leave-one-season-out, so no game is ever scored by a model that saw it &mdash;
      the only fit whose MAE is comparable with the market&rsquo;s, because the market never
      saw the answer either.</p></div>

    <div class="mth-card"><div class="mth-h">Where this is weakest</div>
      <p>The model is 0.44 points behind the closing line overall, and the gap is
      <em>widest</em> late (weeks 10&ndash;18: +0.55) rather than early (weeks 1&ndash;4:
      +0.39). That is the opposite of the college case, and the reason is availability: by
      November the market is pricing injuries and roster news this repo does not model at all,
      while in September there is less of it to know.</p>
      <p>Three families the inherited research prior asserted &mdash; early-down pass
      efficiency, red-zone conversion and special-teams field position &mdash; are not
      derivable from the weekly team box score and are <b>not</b> in the model. They were
      dropped rather than shipped as decorative weights.</p></div>
  </div>
</section>"""


def _footer(built_at: str) -> str:
    return f"""
<footer>
  <div class="wrap">
    <p><b>Chase Analytics &mdash; NFL Model</b> is analysis infrastructure, not betting advice.
    The model does not beat the closing market, its authority is RESEARCH_ONLY, and nothing
    here is a recommendation to wager. If gambling stops being fun, call 1-800-GAMBLER.</p>
    <p class="foot-links">
      <a href="https://github.com/Alphakiller1/nfl-model">Source</a><span>&middot;</span>
      <a href="https://alphakiller1.github.io/mlb-model/">MLB Model</a><span>&middot;</span>
      <a href="https://alphakiller1.github.io/wnba-edge-model/">WNBA Model</a><span>&middot;</span>
      <a href="https://alphakiller1.github.io/cfb-model/">CFB Model</a><span>&middot;</span>
      <a href="https://chase-analytics.com/">chase-analytics.com</a>
    </p>
    <p class="dim">Generated {e(built_at)} &middot; data from nflverse &middot;
    ratings, projections and division odds all derived from one fit.</p>
  </div>
</footer>"""


# ── page ─────────────────────────────────────────────────────────────────────
def render(slate, outlooks) -> str:
    built_at = datetime.now(UTC).strftime("%b %d %Y · %H:%M UTC")
    body = (f"{_nav(slate)}{_hero(slate, built_at)}"
            f'<main class="wrap">'
            f"{_authority_section(slate.authority)}"
            f"{_board_section(slate)}"
            f"{_ratings_section(slate, outlooks)}"
            f"{_units_section(slate)}"
            f"{_divisions_section(outlooks)}"
            f"{_method_section()}"
            f"</main>{_footer(built_at)}")
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>NFL Model — Chase Analytics</title>\n"
        '<meta name="description" content="NFL research dashboard: opponent-adjusted power '
        'ratings, projected scores and totals, offensive and defensive unit rankings, '
        'simulated division races, and an explicit authority gate.">\n'
        f"<style>{brand_css()}{_PAGE_CSS}</style>\n</head>\n<body>\n{body}\n"
        f"<script>{BOARD_JS}</script>\n</body>\n</html>\n"
    )


def build_site(out: Path, season: int | None = None, week: int | None = None,
               simulations: int = divisions_mod.SIMULATIONS) -> Path:
    """Assemble, simulate and write the dashboard."""
    slate = season_mod.assemble(season, week)
    games = divisions_mod.build_games(
        [row for row in slate.schedule if row["season"] == slate.season],
        slate.table,
        margin_of=season_mod.margin_for(slate.table, slate.forms),
    )
    outlooks = divisions_mod.simulate(games, slate.table, simulations=simulations)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(slate, outlooks), encoding="utf-8")
    return out


# Page chrome only. Every colour and typeface comes from chase_tokens.css above;
# this block must never introduce a literal of its own.
_PAGE_CSS = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth;scroll-padding-top:76px}
body{background:var(--bg);color:var(--text);font:15px/1.55 var(--font-primary);
background-image:radial-gradient(1100px 480px at 78% -12%,rgba(124,77,255,.16),transparent 62%),
radial-gradient(900px 420px at 8% 4%,rgba(91,43,224,.10),transparent 55%)}
a{color:var(--v-light);text-decoration:none}a:hover{text-decoration:underline}
.wrap{max-width:1240px;margin:0 auto;padding:0 24px}
.chase-header{position:sticky;top:0;z-index:50;background:rgba(8,9,15,.86);
backdrop-filter:blur(14px);border-bottom:1px solid var(--border-soft)}
.chase-nav{display:flex;align-items:center;gap:22px;height:64px}
.chase-logo{display:flex;align-items:center;gap:10px;min-width:0;overflow:hidden}
.chase-wordmark{font-family:var(--font-wordmark);font-weight:700;font-size:19px;
letter-spacing:.04em;color:var(--text);font-style:italic;white-space:nowrap}
.chase-wordmark em{font-style:italic;background:var(--v-grad);-webkit-background-clip:text;
background-clip:text;color:transparent}
.nav-links{display:flex;gap:2px;flex:1;justify-content:center;flex-wrap:wrap}
.nav-link{padding:7px 11px;border-radius:9px;color:var(--text-2);font-size:13px;font-weight:600}
.nav-link:hover{color:var(--text);background:var(--ca-brand-dim);text-decoration:none}
.chase-status{display:flex;gap:7px;align-items:center}
.product-tag{font-family:var(--font-display);font-weight:700;font-size:11px;
letter-spacing:.14em;color:var(--v-light);border:1px solid var(--ca-brand-border);
border-radius:999px;padding:5px 11px;white-space:nowrap}
.hero{padding:52px 0 32px;border-bottom:1px solid var(--border-soft)}
.hero-eyebrow{font-family:var(--font-display);font-size:11px;font-weight:700;
letter-spacing:var(--ca-editorial-caps);text-transform:uppercase;color:var(--gold);
margin-bottom:14px}
.hero-title{font-family:var(--font-display);font-size:var(--mm-text-hero);font-weight:700;
line-height:1.08;letter-spacing:-.01em}
.hero-sub{color:var(--text-2);max-width:720px;margin-top:12px;font-size:var(--mm-text-md)}
.hero-meta{display:flex;gap:8px;margin-top:18px;flex-wrap:wrap}
.pill{border:1px solid var(--border-soft);border-radius:999px;padding:4px 13px;
font-size:var(--mm-text-xs);background:var(--bg-3);color:var(--text-2);white-space:nowrap}
.pill.warn{color:var(--gold);border-color:var(--ca-brand-border)}
.pill.dim{color:var(--text-3)}
main{padding:14px 0 40px}
section{margin-top:46px;scroll-margin-top:80px}
.sec-head{margin-bottom:16px}
.kicker{font-family:var(--font-display);font-size:11px;font-weight:700;
letter-spacing:var(--ca-editorial-caps);text-transform:uppercase;color:var(--gold)}
h2{font-family:var(--font-display);font-size:var(--mm-text-2xl);margin-top:5px;font-weight:700}
.blurb{color:var(--text-2);max-width:860px;margin-top:8px;font-size:var(--mm-text-sm)}
.blurb b{color:var(--text)}
.note{color:var(--text-3);font-size:var(--mm-text-xs);margin-top:11px;max-width:900px}
.fine{color:var(--text-3);font-size:var(--mm-text-xs);margin-top:10px}
.dim{color:var(--text-3)}
/* Every grid child carries min-width:0. A grid item defaults to min-width:auto,
   which is its content's min-content size, so one wide table stretched the whole
   page and clipped body copy off the right edge at 430px. */
.tiles,.gate-grid,.units,.dvs,.mth{min-width:0}
.tiles>*,.gate-grid>*,.units>*,.dvs>*,.mth>*{min-width:0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;
margin:14px 0}
.tile{background:var(--ca-panel-glass);border:1px solid var(--border-soft);
border-radius:var(--ca-card-radius);padding:15px;display:flex;flex-direction:column;gap:3px}
.tile-v{font-family:var(--font-display);font-size:var(--mm-text-2xl);font-weight:800;
color:var(--v-light);font-variant-numeric:tabular-nums}
.tile-l{font-family:var(--font-display);font-size:var(--mm-text-2xs);font-weight:600;
letter-spacing:var(--ca-editorial-caps);text-transform:uppercase;color:var(--text-2)}
.tile-n{color:var(--text-3);font-size:var(--mm-text-2xs);line-height:1.4}
/* 170px, not 150: the longest value these cards ever hold is "RESEARCH_ONLY",
   which needs 137px at this type size and was being clipped by 2px in the
   four-column layout. The minimum is set by the content, not by eye. */
.gate-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
gap:10px;margin:14px 0}
.gate-card{background:var(--ca-panel-glass);border:1px solid var(--border-soft);
border-radius:var(--ca-card-radius);padding:14px;display:flex;flex-direction:column;gap:3px}
.gate-k{font-family:var(--font-display);font-size:var(--mm-text-2xs);font-weight:600;
letter-spacing:var(--ca-editorial-caps);text-transform:uppercase;color:var(--text-3)}
.gate-v{font-family:var(--font-display);font-size:var(--mm-text-lg);font-weight:700;
color:var(--text);font-variant-numeric:tabular-nums}
.tablewrap{overflow-x:auto;border:1px solid var(--border-soft);
border-radius:var(--ca-card-radius);background:var(--bg-2)}
table{width:100%;border-collapse:collapse;font-size:var(--mm-text-sm)}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--border-soft);
white-space:nowrap}
th{font-family:var(--font-display);font-size:var(--mm-text-2xs);color:var(--text-3);
text-transform:uppercase;letter-spacing:.08em;position:sticky;top:0;background:var(--bg-2)}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover{background:rgba(124,77,255,.05)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
/* Gate names are prose, not figures. The global nowrap clipped the longest of
   them inside its scroll container, which reads as truncated data. */
.gates td{white-space:normal}
.gate-ok{color:var(--ca-green);width:28px}
.gate-no{color:var(--text-4);width:28px}
.empty{padding:16px;border:1px dashed var(--border-2);border-radius:var(--ca-card-radius);
color:var(--text-3);font-size:var(--mm-text-sm)}
/* Power ratings & unit tables */
.tlogo{vertical-align:middle;border-radius:4px}
.tname{display:inline-flex;align-items:center;gap:8px}
.tname b{font-family:var(--font-display);font-weight:700;letter-spacing:.03em}
.tname .dim{font-size:var(--mm-text-xs)}
.pr .rank{color:var(--text-4);width:34px;font-variant-numeric:tabular-nums}
.pr .score{font-family:var(--font-display);font-weight:800;color:var(--v-light)}
.pr .ratingbar{width:20%;min-width:120px}
.pr .eff{color:var(--text-2)}
.rb-track{position:relative;height:8px;border-radius:99px;background:var(--bg-4)}
.rb-zero{position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--border-2)}
.rb-track i{position:absolute;top:0;bottom:0;border-radius:99px}
.rb-pos{background:var(--v-grad)}
.rb-neg{background:var(--ca-red);opacity:.72}
.units{display:grid;grid-template-columns:1fr;gap:16px}
@media(min-width:1080px){.units{grid-template-columns:1fr 1fr}}
.unit-h{font-family:var(--font-display);font-size:var(--mm-text-md);font-weight:700;
margin-bottom:9px;color:var(--text)}
.unit-tbl th,.unit-tbl td{padding:6px 9px;font-size:var(--mm-text-xs)}
/* Division outlook cards */
.dvs{display:grid;grid-template-columns:repeat(auto-fit,minmax(272px,1fr));gap:14px}
.dv{background:var(--ca-panel-glass);border:1px solid var(--border-soft);
border-radius:var(--ca-card-radius);padding:14px 15px 6px;box-shadow:var(--ca-card-shadow)}
.dv-h{display:flex;align-items:center;justify-content:space-between;gap:10px;
font-family:var(--font-display);font-weight:700;font-size:var(--mm-text-md);
letter-spacing:.03em;padding-bottom:9px;border-bottom:1px solid var(--border-soft)}
.dv-pick{display:inline-flex;align-items:center;gap:6px;font-size:var(--mm-text-2xs);
font-weight:600;color:var(--v-light);border:1px solid var(--ca-brand-border);
border-radius:999px;padding:3px 9px;letter-spacing:.02em}
.dv-wrap{overflow-x:auto}
.dv-tbl{font-size:var(--mm-text-xs);width:100%}
.dv-tbl th{background:transparent;position:static;padding:7px 4px;
white-space:normal}
.dv-tbl td{padding:7px 4px;border-bottom:1px solid var(--border-soft)}
.dv-tbl .score{font-family:var(--font-display);font-weight:800;color:var(--v-light)}
.oddsbar{width:30%;min-width:44px}
.ob-track{position:relative;height:6px;border-radius:99px;background:var(--bg-4)}
.ob-track i{position:absolute;left:0;top:0;bottom:0;border-radius:99px;background:var(--v-grad)}
/* Methodology grid */
/* The kernel dims every unpriced tile to opacity .5 so a projection can never
   read as a pick. On an explanatory shelf that is the whole shelf, and the
   result is a legibility problem rather than a signal. A non-market group is
   exactly the one the kernel renders WITHOUT a market count, so it can be
   selected precisely without touching the vendored stylesheet. */
.bd-group:not(:has(.bd-group__count)) .bd-tile.is-idle{opacity:.88}
.bd-group:not(:has(.bd-group__count)) .bd-tile.is-idle .bd-tile__value{color:var(--text-2)}
.mth{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
.mth-card{background:var(--ca-panel-glass);border:1px solid var(--border-soft);
border-radius:var(--ca-card-radius);padding:16px;font-size:var(--mm-text-sm);
color:var(--text-2)}
.mth-h{font-family:var(--font-display);font-size:var(--mm-text-md);font-weight:700;
color:var(--text);margin-bottom:8px}
.mth-card p+p{margin-top:9px}
.mth-card b{color:var(--text)}
.mth-tbl{margin-top:10px;font-size:var(--mm-text-xs)}
.mth-tbl td{padding:5px 0;border-bottom:1px solid var(--border-soft);white-space:normal}
.mth-tbl tr:last-child td{border-bottom:none}
pre{overflow-x:auto;margin:10px 0;padding:11px 13px;background:var(--bg-3);
border:1px solid var(--border-soft);border-radius:9px}
code{font-family:var(--font-display);font-size:var(--mm-text-xs);color:var(--v-light)}
footer{border-top:1px solid var(--border-soft);margin-top:52px;padding:24px 0 44px;
font-size:var(--mm-text-xs);color:var(--text-3)}
footer b{color:var(--text-2)}
.foot-links{margin-top:9px;display:flex;gap:9px;flex-wrap:wrap}
@media(max-width:820px){.hero-title{font-size:var(--mm-text-2xl)}.nav-links{display:none}}
/* At 375px the product tag and the season pill together pushed the document 49px
   wider than the viewport and clipped every paragraph on the page. The wordmark
   is also allowed to clip inside its own box rather than push the row, so a font
   fallback that renders wider cannot bring the overflow back. */
@media(max-width:640px){
.wrap{padding:0 16px}
.chase-nav{gap:8px}
.chase-wordmark{font-size:14px}
.product-tag{font-size:9px;padding:4px 7px;letter-spacing:.08em}
.chase-status{gap:5px}
.chase-status .pill{font-size:10px;padding:3px 8px}
.tile-v{font-size:var(--mm-text-xl)}
}
"""
