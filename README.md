# nfl-model

**Live dashboard:** https://alphakiller1.github.io/nfl-model/ — rebuilt by
`.github/workflows/deploy-pages.yml` on every push to `main` and on a Tuesday cron.

Sibling boards on the same shared kernel: [MLB](https://alphakiller1.github.io/mlb-model/)
· [WNBA](https://alphakiller1.github.io/wnba-edge-model/)
· [CFB](https://alphakiller1.github.io/cfb-model/).

Opponent-adjusted NFL power ratings, projected scores and totals, offensive and
defensive unit rankings, and a simulated race for all eight divisions — behind an
explicit **authority gate** on what any of it may be used for.

> Analysis infrastructure, not betting advice.

## The honest answer

The model does **not** beat the closing market, and this repo says so in its output
rather than hiding it. Measured leave-one-season-out on **2,383** regular-season
games, 2017–2025, strictly point-in-time:

| | Model | Market | Gap |
| --- | ---: | ---: | ---: |
| Margin MAE | 10.3134 | 9.8708 | +0.4427 |
| Total MAE | 10.8076 | 10.4700 | +0.3376 |

ATS on games where the model disagrees with the line: **1158–1165–60 = 49.85%**,
95% CI [47.82%, 51.88%], against a 52.38% breakeven. The interval contains 50% and
sits entirely below breakeven — unproven and slightly negative, not a signal.

So `SPREAD_LAMBDA = 0`: the **published** margin is the market. The model's own
number is shown beside it, and their difference is published as a **gap**, never
an edge. Full evidence and every parameter sweep: [`reports/BASELINE_2016_2025.md`](reports/BASELINE_2016_2025.md).

## What it publishes

| Section | What it answers |
| --- | --- |
| Authority | what the numbers may be used for, and which of the twelve gates are unmet |
| Board | projected score, total, spread and moneyline per game, plus a factor-by-factor breakdown of *why* |
| Disagreements | all sixteen games ranked by how far the model sits from the closing line |
| Power ratings | opponent-adjusted points vs an average team, split into offence, defence and their sum |
| Offense & defense | both units ranked, with opponent-adjusted component rates graded by league percentile |
| Divisions | 20,000 simulated seasons over the real fixture list — division and playoff odds |
| Playoffs | the projected seven-team field per conference, with the cut line drawn |
| Method | every fitted constant, and which sweeps came back flat |

Each game card carries twelve tiles: three market projections, three shelves
explaining which unit drives the number, and the **exact** decomposition of the
margin into its five efficiency families plus home field. Those addends sum to
the efficiency margin to within 1e-14 — a breakdown whose parts do not add up to
the number above it is worse than no breakdown, so `tests/test_matrix.py` pins
the reconciliation.

## The model

One fit produces all of it. It predicts the points **one** team scores, from that
team's offence and its opponent's defence:

```text
points(A vs B) = intercept + Σ gₓ · ( offₓ(A) + defₓ(B) ) + home_field
```

Margin is the difference of the two sides, total is their sum, and the unit
rankings are the same coefficients applied to one team at a time:

```text
offence_index(A) =  Σ gₓ · ( offₓ(A) − meanₓ )     points generated
defence_index(A) = −Σ gₓ · ( defₓ(A) − meanₓ )     points prevented
efficiency rating = offence + defence
```

Because they come from one fit rather than three, the board, the rankings and the
division odds reconcile exactly instead of disagreeing at the edges.

The published margin is a 50/50 blend of that efficiency margin and an
opponent-adjusted **scoring-margin** rating (`ratings.py`). Neither dominates; the
blend is worth ~0.11 points of MAE over either alone.

### Why offence and defence share a coefficient

They measure the same per-play quantity in the same units — what A's offence
produced, what B's defence allowed — so a matchup is their *sum*.

Fitted with free coefficients instead, `def_epa` came out **positive**: allowing
more EPA per play "improved" the margin, because `epa` and `first_down` correlate
at r = 0.83 and the collinear pair split the effect with opposite signs. That
model predicted acceptably and described football backwards, and a defensive
ranking built on it would have looked entirely plausible while rewarding defences
for giving up yards. The symmetric form costs 0.016 points of MAE — noise — and
every coefficient it produces has the sign football says it should.

### What was dropped

The inherited prior in [`nfl-genesis`](https://github.com/Alphakiller1/nfl-genesis)
`src/genesis/logic_matrix.py` asserted seven offensive families with hand-assigned
weights and was labelled `CHALLENGER/UNPROMOTED`. Three of the seven — early-down
pass efficiency, red-zone conversion, special-teams field position — are not
derivable from the weekly team box score. They were dropped rather than shipped as
decorative weights on a live board.

## The authority gate

A probability and a permission are different things. Conflating them is how an
unpromoted model becomes a bet.

| Level | Meaning |
| --- | --- |
| `RESEARCH_ONLY` | numbers only; never sized, never staked — **current state** |
| `SHADOW` | logged as if traded, no capital |
| `PROMOTED` | may emit `BET` |

Actions an authority may permit: `AVOID` (no usable price), `MONITOR` (priced and
modelled but unpromoted), `REVIEW` (edge implausible enough to suspect the inputs),
`BET` (promoted only). Nine of twelve production gates are unmet — run
`nfl-model status` to see which.

`promote()` takes the satisfied gate set as an **argument** rather than reading a
config flag, so promotion is always an auditable claim about evidence.

## Use

```bash
pip install -e ".[dev]"

nfl-model status                        # authority and unmet gates
nfl-model board                         # this week's slate, scores and totals
nfl-model ratings                       # power ratings, offence/defence split
nfl-model units                         # offensive and defensive rankings
nfl-model divisions                     # simulated division and playoff odds
nfl-model export --out board.json       # the JSON contract
nfl-model build-site --out docs/index.html
```

Every command defaults to the current season and the first week with an unplayed
game, so a scheduled build needs no arguments.

Data comes from [nflverse](https://github.com/nflverse) — `games.csv` for the
schedule and market, `stats_team_week_<season>.csv` for box scores. **No API key**,
and the package has **no third-party dependencies**: stdlib `urllib` plus `csv`,
with completed seasons cached forever and volatile files on a 6-hour TTL.

The moneyline probability contract (`forecast_slate`) is unchanged and still
market-anchored per `nfl-genesis`; `profit-priority` consumes it through
`feeds/nfl_model.py` and refuses to treat a `RESEARCH_ONLY` forecast as a value edge.

## Refitting

```bash
python scripts/fit_matrix.py            # full fit + parameter sweeps
python scripts/fit_matrix.py --no-sweeps
```

The script may use numpy; the package may not. Fitted values are copied into
`matrix.py`, `totals.py`, `ratings.py` and `preseason.py` as constants, exactly as
cfb-model does.

## The dashboard

`docs/index.html` and `docs/board.json` are a **committed snapshot** of the last
local build, for review and as a fallback. GitHub Pages does **not** serve from
`docs/` — `deploy-pages.yml` builds fresh from nflverse into `_site/` and uploads
that as the Pages artifact, so a code change ships without anyone remembering to
regenerate the snapshot.

`nfl-model build-site` renders a self-contained static page. It leads with the
**authority gate** and only then shows the board, because a dashboard that put
prices first and permissions in a footnote would misrepresent the exact thing the
gate exists to prevent. `tests/test_site.py` asserts that ordering.

The page is built from the **shared Chase Analytics board kernel**:
`src/nflmodel/board.py`, `static/board.css` and `static/chase_tokens.css` are
vendored **byte-identical** from [`mlb-model`](https://github.com/Alphakiller1/mlb-model)
and [`wnba-edge-model`](https://github.com/Alphakiller1/wnba-edge-model), so an NFL
card has the same anatomy, palette and typefaces as an MLB or WNBA one.
`BOARD_CONTRACT.sha256` pins their hashes and `tests/test_board_contract.py` fails
the build if any copy drifts — change a shared file in all three repos and
regenerate the manifest, never edit one in isolation. Everything this repo adds on
top uses tokens only; a test asserts `_PAGE_CSS` contains no colour literal.

What differs per sport is only the *adapter* (`board_nfl.py`):

| Slot | MLB | WNBA | NFL |
| --- | --- | --- | --- |
| `principals` | starting pitchers | usage leaders | starting QBs, or head coaches until nflverse publishes them |
| `groups` | Full Game, First 5 | Full Game | Full Game + an unpriced matchup shelf |
| side `score` | expected runs | projected points | projected points |

## Design notes

- **Point-in-time by construction.** `season.assemble` takes the week being
  forecast and only ever reads games strictly before it, so no code path can hand
  a forecaster the result it is predicting.
- **Absences are explicit.** A game the model cannot rate is emitted with null
  fields and a reason, never dropped — a silently shorter list is
  indistinguishable from a shorter slate.
- **One margin, everywhere.** The season simulation and the game board are driven
  by the same function. Two estimates of one quantity is how division odds end up
  disagreeing with the spreads on the same page.
- **Relocations fold forward.** nflverse keeps the abbreviation a team carried at
  the time, so `OAK`/`LV` and `SD`/`LAC` would otherwise be rated as four
  franchises and all four would be wrong while the table still looked fine.
- **Paired de-vigging only.** One-sided de-vigging against an assumed overround is
  guesswork and a common source of phantom edge. An implausible overround raises
  rather than returning a "fair" price.
- **Flat sweeps are reported as flat.** The blowout cap, home field and the recency
  half-life all moved MAE by less than 0.01 points. They are published at their
  argmin and described as unidentified rather than dressed up as tuning.
