"""
Self-contained static dashboard for GitHub Pages — Chase Analytics design contract.

Visual identity is vendored byte-identical from mlb-model (`static/chase_tokens.css` +
`static/board.css`), so this product, wnba-edge-model and mlb-model read as one brand.
`tests/test_board_contract.py` fails the build if a vendored file drifts.

The page leads with the authority gate rather than the numbers. This model matches the
market and does not beat it; a dashboard that showed prices first and permissions in a
footnote would misrepresent exactly the thing the authority gate exists to prevent.
"""
from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path

from . import authority as auth
from .board import BOARD_JS, board_html
from .board_nfl import build_board
from .forecast import DEFAULT_LAMBDA, forecast_slate

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
    <span class="kicker">Authority · 1/3</span>
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
    <span class="kicker">Methodology · 3/3</span>
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
  <section id="board">
    <div class="sec-head">
      <span class="kicker">Slate · 2/3</span>
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
@media(max-width:640px){.hero-title{font-size:var(--mm-text-3xl)}.nav-links{display:none}}
"""
