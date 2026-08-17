"""
Self-contained static dashboard for GitHub Pages — Chase Analytics design contract.

Visual identity is vendored byte-identical from mlb-model (`static/chase_tokens.css` +
`static/board.css`), so this product, wnba-edge-model and mlb-model read as one brand.
`tests/test_board_contract.py` fails the build if a vendored file drifts.

The page leads with the authority gate rather than the numbers. This model matches the
market and does not beat it; a dashboard that showed prices first and permissions in a
footnote would misrepresent exactly the thing the authority gate exists to prevent.
"""
# ruff: noqa: E501
from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path

from . import authority as auth
from . import ratings
from .board import BOARD_JS, board_html
from .board_nfl import build_board
from .forecast import DEFAULT_LAMBDA, forecast_slate
from .projections import (
    division_winners,
    outlook,
    outlook_note,
    schedule_source,
    team_logo_url,
    week_one_projections,
)

_STATIC = Path(__file__).resolve().parent / "static"

_FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;0,9..40,800&"
    "family=Oswald:ital,wght@0,600;0,700;0,900;1,600;1,700;1,900&"
    "family=Roboto+Condensed:wght@400;500;600;700;800&display=swap');"
)

e = html.escape


def brand_css() -> str:
    """Fonts + chase_tokens.css + the board kernel — the shared Chase identity."""
    tokens = (_STATIC / "chase_tokens.css").read_text(encoding="utf-8")
    board = (_STATIC / "board.css").read_text(encoding="utf-8")
    return _FONT_IMPORT + tokens + board


def _load_games(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["games"] if isinstance(data, dict) else data


def _gate_section(payload: dict) -> str:
    unmet = payload.get("unmet_gates") or []
    satisfied = [gate for gate in auth.REQUIRED_GATES if gate not in unmet]
    rows = "".join(
        f'<tr><td class="gate-ok">✓</td><td>{e(gate)}</td></tr>' for gate in satisfied
    ) + "".join(
        f'<tr><td class="gate-no">✗</td><td>{e(gate)}</td></tr>' for gate in unmet
    )
    level = e(str(payload.get("authority") or ""))
    may_bet = "yes" if payload.get("may_bet") else "no"
    return f"""
<section id="authority">
  <div class="sec-head">
    <span class="kicker">Authority · 1/5</span>
    <h2>What these numbers may be used for</h2>
    <p class="blurb">A probability and a permission are different things. This model
    <b>matches</b> the paired no-vig closing market and does not beat it, so it is
    <b>{level}</b> and may never emit a bet. Matching a benchmark is not an edge.</p>
  </div>
  <div class="gate-grid">
    <div class="gate-card">
      <span class="gate-k">Authority</span><span class="gate-v">{level}</span>
    </div>
    <div class="gate-card">
      <span class="gate-k">May bet</span><span class="gate-v">{may_bet}</span>
    </div>
    <div class="gate-card">
      <span class="gate-k">Gates met</span>
      <span class="gate-v">{len(satisfied)} / {len(auth.REQUIRED_GATES)}</span>
    </div>
    <div class="gate-card">
      <span class="gate-k">Lambda</span>
      <span class="gate-v">{float(payload.get("lam", 0.0)):.3f}</span>
    </div>
  </div>
  <div class="tablewrap"><table>
    <thead><tr><th></th><th>Production gate</th></tr></thead>
    <tbody>{rows}</tbody></table></div>
  <p class="note">{e(str(payload.get("evidence") or ""))}</p>
</section>"""


def _method_section(payload: dict) -> str:
    note = str(payload.get("note") or "")
    return f"""
<section id="methodology">
  <div class="sec-head">
    <span class="kicker">Methodology · 4/4</span>
    <h2>How the forecast is built</h2>
  </div>
  <div class="prose">
    <p>The forecast is market-anchored:</p>
    <pre><code>logit(p) = logit(market_fair)
         + lam * (logit(structural) - logit(market_fair))</code></pre>
    <p><code>lam</code> is selected out-of-sample in
    <a href="https://github.com/Alphakiller1/nfl-genesis">nfl-genesis</a>, not here. It
    selected <b>0.000 in all five folds from 2021 onward</b> — the estimator stating that the
    structural component carries no incremental information over a paired no-vig closing
    line.{f" {e(note)}" if note else ""}</p>
    <p><b>Paired de-vigging only.</b> One-sided de-vigging against an assumed overround is
    guesswork and a common source of phantom edge; an implausible overround raises rather
    than returning a "fair" price. Games without a paired price are reported as
    <code>AVOID</code> rather than dropped — a silently shorter list is indistinguishable
    from a slate with no games.</p>
  </div>
</section>"""


def _ratings_section() -> str:
    """Preseason power ratings — the model's content when no market exists yet."""
    payload = ratings.load()
    entries = ratings.teams()
    head = (
        '<section id="ratings">'
        '<div class="sec-head">'
        '<span class="kicker">Preseason · 4/5</span>'
        "<h2>Power ratings</h2>"
        '<p class="blurb">Points relative to an average team on a neutral field. This is a '
        "prior, not an edge: it is what the model believes about team strength before any "
        "line exists, and it is not anchored to a market because out of season there is no "
        "market to anchor to.</p></div>"
    )
    if not entries:
        return (
            f"{head}<div class=\"empty\">No ratings built yet. Run "
            f"<code>scripts/build_power_ratings.py</code> in nfl-genesis and commit the "
            f"emitted <code>power_ratings.json</code>.</div></section>"
        )

    seasons = payload.get("seasons") or []
    span = f"{min(seasons)}–{max(seasons)}" if seasons else "recent seasons"
    hfa = float(payload.get("home_field_points") or 0.0)

    rows = []
    for entry in entries:
        last = entry.get("last_season") or {}
        record = (
            f"{last.get('w', 0)}&#8209;{last.get('l', 0)}"
            + (f"&#8209;{last['t']}" if last.get("t") else "")
            if last
            else "&ndash;"
        )
        diff = last.get("diff")
        diff_tone = "pos" if (diff or 0) > 0 else "neg" if (diff or 0) < 0 else "dim"
        diff_cell = (
            f'<span class="{diff_tone}">{diff:+.0f}</span>'
            if isinstance(diff, (int, float))
            else "&ndash;"
        )
        rating = float(entry.get("rating", 0.0))
        # Bar is centred on zero: above-average right, below-average left.
        width = min(abs(rating) / 7.0, 1.0) * 50.0
        offset = 50.0 if rating >= 0 else 50.0 - width
        tone = "pos" if rating >= 0 else "neg"
        team = str(entry.get("team", ""))
        wins = ratings.projected_wins(team)
        wins_cell = f"{wins:.1f}" if wins is not None else "&ndash;"
        rows.append(
            f'<tr><td class="rank">{entry.get("rank", "")}</td>'
            f'<td class="team"><span class="rating-team">{_team_mark(team)}'
            f'<b>{e(team)}</b></span></td>'
            f'<td class="ratingbar"><div class="rb-track"><span class="rb-zero"></span>'
            f'<i class="rb-{tone}" style="left:{offset:.1f}%;width:{width:.1f}%"></i></div></td>'
            f'<td class="num score">{rating:+.2f}</td>'
            f'<td class="num wins">{wins_cell}</td>'
            f'<td class="num">{record}</td>'
            f'<td class="num">{last.get("ppg", "&ndash;")}</td>'
            f'<td class="num">{last.get("papg", "&ndash;")}</td>'
            f'<td class="num">{diff_cell}</td></tr>'
        )

    return (
        f"{head}"
        f'<p class="note">{e(str(payload.get("method") or ""))} '
        f'Home field is worth <b>{hfa:.2f}</b> points, measured over the same window '
        f'({e(span)}, {payload.get("games_used", 0)} regular-season games). '
        f'<b>Proj W</b> is expected wins against a league-average schedule, not a '
        f'schedule-aware total — a schedule-aware season simulation has not yet been published '
        f'yet, and inventing opponents would make the number look more precise than the '
        f'inputs support. The 32 projections sum to '
        f'{sum(ratings.projected_wins(t["team"]) or 0 for t in entries):.0f} wins against the '
        f'{ratings.GAMES_PER_SEASON * len(entries) // 2} a full season contains.</p>'
        f'<div class="tablewrap"><table class="pr">'
        f"<thead><tr><th></th><th>Team</th><th>Rating</th><th class=\"num\">Pts</th>"
        f'<th class="num">Proj W</th>'
        f'<th class="num">Last yr</th><th class="num">PF/g</th><th class="num">PA/g</th>'
        f'<th class="num">Diff</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div></section>"
    )


def _team_mark(team: str) -> str:
    """A compact, accessible logo; its abbreviation remains visible alongside it."""
    return (
        f'<img class="team-mark" src="{e(team_logo_url(team))}" alt="{e(team)} logo" '
        'loading="lazy" width="36" height="36">'
    )


def _outlook_overview() -> str:
    payload = outlook()
    generated = str(payload.get("generated_at_utc", "")).replace("T", " · ").replace(
        "+00:00", " UTC"
    )
    return (
        '<section class="outlook-overview" aria-label="Genesis outlook summary">'
        '<div class="outlook-title"><span class="kicker">Genesis outlook · 2026</span>'
        '<h2>Preseason command center</h2><p>One research-only source powers the Week 1 board, '
        'division leaders, and team ratings.</p></div>'
        '<div class="outlook-metrics">'
        '<div><span>Week 1</span><b>16 games</b></div>'
        '<div><span>Division board</span><b>8 leaders</b></div>'
        '<div><span>Authority</span><b>Research only</b></div>'
        f'<div><span>Genesis publish</span><b>{e(generated)}</b></div>'
        '</div></section>'
    )


def _week_one_section() -> str:
    cards = []
    for game in week_one_projections():
        away = str(game["away_team"])
        home = str(game["home_team"])
        home_probability = float(game["home_win_probability"])
        away_probability = float(game["away_win_probability"])
        margin = float(game["home_margin"])
        favored, favored_probability = (
            (home, home_probability) if home_probability >= 0.5 else (away, away_probability)
        )
        margin_text = f"{home} by {margin:+.1f}" if margin >= 0 else f"{away} by {abs(margin):.1f}"
        cards.append(
            '<article class="projection-card">'
            f'<div class="projection-start">{e(str(game["start_text"]))}</div>'
            '<div class="projection-matchup">'
            f'<div class="projection-team">{_team_mark(away)}<b>{e(away)}</b>'
            f'<span>{away_probability:.0%}</span></div>'
            '<span class="projection-vs">@</span>'
            f'<div class="projection-team">{_team_mark(home)}<b>{e(home)}</b>'
            f'<span>{home_probability:.0%}</span></div></div>'
            f'<p><b>{e(favored)}</b> {favored_probability:.0%} win probability · '
            f'{e(margin_text)} projected</p>'
            '</article>'
        )
    return (
        '<section id="projections">'
        '<div class="sec-head"><span class="kicker">Preseason · 2/5</span>'
        '<h2>Week 1 projections</h2>'
        '<p class="blurb">Rating-based win probabilities for every Week 1 matchup. These are '
        'preseason strength projections, not market prices or betting recommendations.</p></div>'
        f'<p class="note">Published by <a href="https://github.com/Alphakiller1/nfl-genesis">NFL Genesis</a>. '
        f'Matchups and kickoff windows: <a href="{e(schedule_source())}">NFL 2026 Week 1 schedule</a>. '
        f'{e(outlook_note())}</p>'
        f'<div class="projection-grid">{"".join(cards)}</div></section>'
    )


def _division_section() -> str:
    cards = []
    for winner in division_winners():
        team = str(winner["team"])
        cards.append(
            '<article class="division-card">'
            f'<span class="division-name">{e(str(winner["division"]))}</span>'
            f'<div class="division-winner">{_team_mark(team)}<b>{e(team)}</b></div>'
            '<div class="division-stats">'
            f'<span><b>{float(winner["projected_wins"]):.1f}</b> proj W</span>'
            f'<span><b>{float(winner["rating"]):+.2f}</b> rating</span></div>'
            f'<p>Over {e(str(winner["runner_up"]))} by {float(winner["rating_gap"]):.2f} pts</p>'
            '</article>'
        )
    return (
        '<section id="divisions"><div class="sec-head">'
        '<span class="kicker">Preseason · 3/5</span><h2>Projected division winners</h2>'
        '<p class="blurb">The highest-rated team in each division. Projected wins remain '
        'league-average-schedule estimates, so this is a transparent strength-based division '
        'projection rather than a schedule simulation.</p></div>'
        f'<div class="division-grid">{"".join(cards)}</div></section>'
    )


def build_site(out: Path, games_path: Path | None = None,
               lam: float = DEFAULT_LAMBDA) -> Path:
    payload = forecast_slate(_load_games(games_path), lam)
    board = build_board(payload)
    built_at = datetime.now(UTC).strftime("%b %d · %H:%M UTC")
    level = e(str(payload.get("authority") or ""))

    body = f"""
<header class="chase-header">
  <nav class="chase-nav wrap">
    <a href="https://chase-analytics.com" class="chase-logo" title="Chase Analytics">
      <svg viewBox="0 0 36 36" width="30" height="30" aria-hidden="true">
      <path d="M18 5 C21 13 24 20 33 31 L3 31 C12 20 15 13 18 5 Z" fill="#7C4DFF"/></svg>
      <span class="chase-wordmark">CHASE&nbsp;<em>ANALYTICS</em></span>
    </a>
    <div class="nav-links">
      <a class="nav-link" href="#authority">Authority</a>
      <a class="nav-link" href="#projections">Week 1</a>
      <a class="nav-link" href="#divisions">Divisions</a>
      <a class="nav-link" href="#ratings">Ratings</a>
      <a class="nav-link" href="#board">Board</a>
      <a class="nav-link" href="#methodology">Methodology</a>
    </div>
    <div class="chase-status"><span class="product-tag">NFL MODEL</span></div>
  </nav>
</header>
<header class="hero">
  <div class="wrap">
    <div class="hero-eyebrow">CHASE ANALYTICS&ensp;|&ensp;NFL INTELLIGENCE</div>
    <h1 class="hero-title">A calibrated price,<br>and the permission to use it.</h1>
    <p class="hero-sub">Market-anchored NFL forecasts behind an explicit authority gate.
    This model matches the closing market and does not beat it &mdash; and says so here
    rather than hiding it. Research software, not betting advice.</p>
    <div class="hero-meta">
      <span class="pill warn">{level}</span>
      <span class="pill dim">built {e(built_at)}</span>
    </div>
  </div>
</header>
<main class="wrap">
  {_gate_section(payload)}
  {_outlook_overview()}
  {_week_one_section()}
  {_division_section()}
  {_ratings_section()}
  <section id="board">
    <div class="sec-head">
      <span class="kicker">Market slate · 5/5</span>
      <h2>Forecast board</h2>
      <p class="blurb">Every card carries the action its authority permits. While this model
      is research-only that action is <code>MONITOR</code> at best &mdash; never
      <code>BET</code>.</p>
    </div>
    {board_html(board)}
  </section>
  {_method_section(payload)}
</main>
<footer>
  <div class="wrap">
    <p><b>Chase Analytics &mdash; NFL Model</b> is research and analytics software. It does
    not provide betting advice, does not guarantee outcomes, and no output is a wager
    instruction. If you or someone you know has a gambling problem, call 1-800-GAMBLER.</p>
    <p class="foot-links">
      <a href="https://github.com/Alphakiller1/nfl-model">Source</a><span>&middot;</span>
      <a href="https://alphakiller1.github.io/mlb-model/">MLB Model</a><span>&middot;</span>
      <a href="https://alphakiller1.github.io/wnba-edge-model/">WNBA Model</a><span>&middot;</span>
      <a href="https://chase-analytics.com/">chase-analytics.com</a>
    </p>
  </div>
</footer>"""

    document = (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>NFL Model — Chase Analytics</title>\n"
        '<meta name="description" content="Market-anchored NFL forecasts behind an explicit '
        'authority gate: what the model says, and what it is allowed to be used for.">\n'
        f"<style>{brand_css()}{_PAGE_CSS}</style>\n</head>\n<body>\n{body}\n"
        f"<script>{BOARD_JS}</script>\n</body>\n</html>\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8")
    return out


# Page chrome only. Every colour and typeface comes from chase_tokens.css above; this block
# must never introduce a literal of its own.
_PAGE_CSS = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth;scroll-padding-top:76px}
body{background:var(--bg);color:var(--text);font:15px/1.55 var(--font-primary);
background-image:radial-gradient(1100px 480px at 78% -12%,rgba(124,77,255,.16),transparent 62%),
radial-gradient(900px 420px at 8% 4%,rgba(91,43,224,.10),transparent 55%)}
a{color:var(--v-light);text-decoration:none}a:hover{text-decoration:underline}
.wrap{max-width:1180px;margin:0 auto;padding:0 24px}
.chase-header{position:sticky;top:0;z-index:50;background:rgba(8,9,15,.82);
backdrop-filter:blur(14px);border-bottom:1px solid var(--border-soft)}
.chase-nav{display:flex;align-items:center;gap:26px;height:64px}
.chase-logo{display:flex;align-items:center;gap:10px}
.chase-wordmark{font-family:var(--font-wordmark);font-weight:700;font-size:19px;
letter-spacing:.04em;color:var(--text);font-style:italic}
.chase-wordmark em{font-style:italic;background:var(--v-grad);-webkit-background-clip:text;
background-clip:text;color:transparent}
.nav-links{display:flex;gap:4px;flex:1;justify-content:center}
.nav-link{padding:7px 13px;border-radius:9px;color:var(--text-2);font-size:13.5px;font-weight:600}
.nav-link:hover{color:var(--text);background:rgba(124,77,255,.12);text-decoration:none}
.product-tag{font-family:var(--font-display);font-weight:700;font-size:11px;
letter-spacing:.14em;color:var(--v-light);border:1px solid var(--ca-brand-border);
border-radius:999px;padding:5px 11px}
.hero{padding:54px 0 34px;border-bottom:1px solid var(--border-soft)}
.hero-eyebrow{font-family:var(--font-display);font-size:11px;font-weight:700;
letter-spacing:var(--ca-editorial-caps);text-transform:uppercase;color:var(--gold);margin-bottom:14px}
.hero-title{font-family:var(--font-display);font-size:var(--mm-text-hero);font-weight:700;
line-height:1.08;letter-spacing:-.01em}
.hero-sub{color:var(--text-2);max-width:640px;margin-top:12px;font-size:var(--mm-text-md)}
.hero-meta{display:flex;gap:8px;margin-top:18px;flex-wrap:wrap}
.pill{border:1px solid var(--border-soft);border-radius:999px;padding:4px 13px;
font-size:var(--mm-text-xs);background:var(--bg-3);color:var(--text-2)}
.pill.warn{color:var(--gold);border-color:var(--ca-brand-border)}
.pill.dim{color:var(--text-3)}
main{padding:14px 0 40px}
section{margin-top:40px}
.sec-head{margin-bottom:14px}
.kicker{font-family:var(--font-display);font-size:11px;font-weight:700;
letter-spacing:var(--ca-editorial-caps);text-transform:uppercase;color:var(--gold)}
h2{font-family:var(--font-display);font-size:var(--mm-text-2xl);margin-top:5px;font-weight:700}
.blurb{color:var(--text-2);max-width:720px;margin-top:7px;font-size:var(--mm-text-sm)}
.note{color:var(--text-3);font-size:var(--mm-text-xs);margin-top:11px;max-width:820px}
.gate-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
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
th,td{padding:9px 14px;text-align:left;border-bottom:1px solid var(--border-soft)}
th{font-family:var(--font-display);font-size:var(--mm-text-2xs);color:var(--text-3);
text-transform:uppercase;letter-spacing:.08em}
tbody tr:last-child td{border-bottom:none}
.gate-ok{color:var(--ca-green);width:28px}
.gate-no{color:var(--ca-red);width:28px}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.dim{color:var(--text-3)}
.pos{color:var(--ca-green)}
.neg{color:var(--ca-red)}
.empty{padding:16px;border:1px dashed var(--border-2);border-radius:var(--ca-card-radius);
color:var(--text-3);font-size:var(--mm-text-sm);background:rgba(255,255,255,.012)}
.outlook-overview{border:1px solid var(--ca-brand-border);border-radius:var(--ca-card-radius);overflow:hidden;position:relative;background:linear-gradient(125deg,rgba(124,77,255,.18),rgba(255,255,255,.02) 58%);padding:22px}.outlook-overview:after{content:"";position:absolute;right:-60px;top:-100px;width:230px;height:230px;border:1px solid rgba(154,107,255,.28);border-radius:50%;box-shadow:0 0 0 36px rgba(154,107,255,.05),0 0 0 72px rgba(154,107,255,.025)}.outlook-title{position:relative;z-index:1}.outlook-title h2{margin-top:4px}.outlook-title p{color:var(--text-2);font-size:var(--mm-text-sm);max-width:530px;margin-top:5px}.outlook-metrics{display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:8px;margin-top:18px;position:relative;z-index:1}.outlook-metrics div{background:rgba(8,9,15,.48);border:1px solid var(--border-soft);border-radius:10px;padding:10px 12px}.outlook-metrics span{display:block;color:var(--text-3);font-family:var(--font-display);font-size:var(--mm-text-2xs);letter-spacing:.08em;text-transform:uppercase}.outlook-metrics b{display:block;color:var(--text);font-family:var(--font-display);font-size:var(--mm-text-sm);margin-top:2px}
.projection-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(255px,1fr));gap:10px}
.projection-card{background:var(--ca-panel-glass);border:1px solid var(--border-soft);border-radius:var(--ca-card-radius);padding:13px;box-shadow:0 12px 32px rgba(0,0,0,.10);transition:border-color .18s ease,transform .18s ease}.projection-card:hover{border-color:var(--ca-brand-border);transform:translateY(-2px)}
.projection-start{color:var(--text-3);font-family:var(--font-display);font-size:var(--mm-text-2xs);letter-spacing:.06em;text-transform:uppercase}
.projection-matchup{align-items:center;display:grid;grid-template-columns:1fr 20px 1fr;gap:5px;margin:8px 0}
.projection-team{align-items:center;display:grid;grid-template-columns:36px auto 1fr;gap:7px;min-width:0}.projection-team b{font-family:var(--font-display);letter-spacing:.04em}.projection-team span{text-align:right;color:var(--v-light);font-family:var(--font-display);font-weight:800;font-size:var(--mm-text-md)}
.projection-vs{color:var(--text-4);font-family:var(--font-display);font-weight:700;text-align:center}.projection-card p{color:var(--text-2);font-size:var(--mm-text-xs)}.projection-card p b{color:var(--text)}
.team-mark{display:block;height:36px;object-fit:contain;width:36px}.division-team{align-items:center;display:flex;gap:9px}.division-team .team-mark{height:28px;width:28px}
.division-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.division-card{background:var(--ca-panel-glass);border:1px solid var(--border-soft);border-radius:var(--ca-card-radius);min-height:190px;padding:15px;position:relative;overflow:hidden}.division-card:before{background:var(--v-grad);content:"";height:3px;left:0;position:absolute;right:0;top:0}.division-name{color:var(--gold);font-family:var(--font-display);font-size:var(--mm-text-2xs);letter-spacing:.09em;text-transform:uppercase}.division-winner{align-items:center;display:flex;gap:9px;margin:13px 0 11px}.division-winner .team-mark{height:46px;width:46px}.division-winner b{font-family:var(--font-display);font-size:var(--mm-text-xl);letter-spacing:.05em}.division-stats{border-top:1px solid var(--border-soft);display:flex;gap:15px;padding-top:9px}.division-stats span{color:var(--text-3);font-size:var(--mm-text-2xs)}.division-stats b{color:var(--v-light);display:block;font-family:var(--font-display);font-size:var(--mm-text-md)}.division-card p{color:var(--text-3);font-size:var(--mm-text-xs);margin-top:9px}.rating-team{align-items:center;display:flex;gap:8px}.rating-team .team-mark{height:25px;width:25px}
/* Power ratings: a zero-centred bar, so above- and below-average teams are told apart at a
   glance rather than by reading the sign off a number. */
.pr .rank{color:var(--text-4);width:34px;font-variant-numeric:tabular-nums}
.pr .team{font-family:var(--font-display);font-weight:700;letter-spacing:.03em;width:62px}
.pr .score{font-family:var(--font-display);font-weight:800;color:var(--v-light)}
.pr .ratingbar{width:38%;min-width:150px}
.rb-track{position:relative;height:8px;border-radius:99px;background:var(--bg-4)}
.rb-zero{position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--border-2)}
.rb-track i{position:absolute;top:0;bottom:0;border-radius:99px}
.rb-pos{background:var(--v-grad)}
.rb-neg{background:var(--ca-red);opacity:.75}
.prose{background:var(--ca-panel-glass);border:1px solid var(--border-soft);
border-radius:var(--ca-card-radius);padding:18px;font-size:var(--mm-text-sm);color:var(--text-2)}
.prose p+p{margin-top:11px}
.prose b{color:var(--text)}
pre{overflow-x:auto;margin:10px 0;padding:11px 13px;background:var(--bg-3);
border:1px solid var(--border-soft);border-radius:9px}
code{font-family:var(--font-display);font-size:var(--mm-text-xs);color:var(--v-light)}
footer{border-top:1px solid var(--border-soft);margin-top:48px;padding:24px 0 40px;
font-size:var(--mm-text-xs);color:var(--text-3)}
.foot-links{margin-top:9px;display:flex;gap:9px;flex-wrap:wrap}
@media(max-width:760px){.outlook-metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.division-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:640px){.hero-title{font-size:var(--mm-text-3xl)}.nav-links{display:none}.division-grid{grid-template-columns:1fr}}
"""
