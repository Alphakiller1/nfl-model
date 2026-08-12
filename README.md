# nfl-model

**Live dashboard:** https://alphakiller1.github.io/nfl-model/ — rebuilt by
`.github/workflows/deploy-pages.yml` on every push to `main`. It leads with the authority
gate rather than the numbers, because at `lam = 0` the price below *is* the market.

Sibling boards on the same shared kernel: [MLB](https://alphakiller1.github.io/mlb-model/)
· [WNBA](https://alphakiller1.github.io/wnba-edge-model/).

Deployable NFL forecasts with an explicit **authority gate**, sitting downstream of
[`nfl-genesis`](https://github.com/Alphakiller1/nfl-genesis) the way `mlb-model` sits downstream
of `mlbma-pipeline`.

`nfl-genesis` is the research foundation: acquisition, point-in-time features, walk-forward
competitions, promotion gates. This repo is the thin, dependency-free layer that turns its
published evidence into prices other systems can consume — and, just as importantly, into a
**permission** that travels with those prices.

> Analysis infrastructure, not betting advice.

## The current honest answer

The model does **not** beat the market, and this repo says so in its output rather than hiding it.

From `nfl-genesis` `reports/EMPIRICAL_BASELINE_2016_2025.md`, on 1,865 untouched regular-season
games (2019–2025):

| Candidate | Log loss | Brier | vs market |
| --- | ---: | ---: | ---: |
| Paired no-vig closing market | **0.608393** | **0.210547** | — |
| Free-coefficient stacker | 0.611319 | 0.211378 | −0.002926 |
| Market-anchored stacker | 0.608567 | 0.210583 | −0.000173 |

The forecast is market-anchored:

```text
logit(p) = logit(market_fair) + lam * (logit(structural) - logit(market_fair))
```

`lam` is selected out-of-sample in the research repo, and it selected **0.000 in all five folds
from 2021 onward**. That is the estimator stating the structural component carries no
incremental information over a paired no-vig closing line.

**So at `lam = 0` this forecast equals the market by construction.** That is a calibrated price,
not an edge. Shipping it as an edge would be the exact failure the anchoring was built to prevent.

## Why a near-trivial module is the point

`forecast.py` looks thin today, and it should. Its value is that the shape is correct and
auditable: when a challenger in `nfl-genesis` finally moves `lam` off zero, only
`DEFAULT_LAMBDA` and the evidence pointer change — and every consumer inherits the improvement
**together with the promotion state**, rather than silently gaining a signal nobody gated.

## The authority gate

A probability and a permission are different things. Conflating them is how an unpromoted model
becomes a bet.

| Level | Meaning |
| --- | --- |
| `RESEARCH_ONLY` | numbers only; never sized, never staked — **current state** |
| `SHADOW` | logged as if traded, no capital |
| `PROMOTED` | may emit `BET` |

Actions an authority may permit: `AVOID` (no usable price), `MONITOR` (priced and modelled but
unpromoted), `REVIEW` (edge implausible enough to suspect the inputs), `BET` (promoted only).

`promote()` takes the satisfied gate set as an **argument** rather than reading a config flag,
so promotion is always an auditable claim about evidence. Nine of twelve production gates are
currently unmet — run `nfl-model status` to see exactly which.

## Use

```bash
pip install -e ".[dev]"
nfl-model status
nfl-model forecast --games slate.json
nfl-model build-site --games data/slate_example.json --out _site/index.html
```

`slate.json`:

```json
[{"home_team": "KC", "away_team": "LAC", "home_american": -175, "away_american": 150}]
```

Output carries the price, both American lines, the permitted action, and the unmet gates.
Games without a paired price are reported as `AVOID` in `skipped` rather than dropped — a
silently shorter list is indistinguishable from a slate with no games.

## The dashboard

`nfl-model build-site` renders a self-contained static page for GitHub Pages. It leads with the
**authority gate** and only then shows the board: a dashboard that put prices first and
permissions in a footnote would misrepresent the exact thing the gate exists to prevent. Every
card carries the action its authority permits — `MONITOR` at best today, never `BET` — and the
board reports `0 gems`, because a gem asserts an actionable edge and this authority permits none.

The page is built from the **shared Chase Analytics board kernel**: `src/nflmodel/board.py`,
`static/board.css` and `static/chase_tokens.css` are vendored **byte-identical** from
[`mlb-model`](https://github.com/Alphakiller1/mlb-model) and
[`wnba-edge-model`](https://github.com/Alphakiller1/wnba-edge-model), so an NFL card has the same
anatomy, palette and typefaces as an MLB or WNBA one. `BOARD_CONTRACT.sha256` pins their hashes
and `tests/test_board_contract.py` fails the build if any copy drifts — change a shared file in
all three repos and regenerate the manifest, never edit one in isolation.

What differs per sport is only the *adapter* (`board_nfl.py`): which slots the sport fills.

| Slot | MLB | WNBA | NFL |
| --- | --- | --- | --- |
| `principals` | starting pitchers | usage leaders | *(none — no player feed)* |
| `groups` | Full Game, First 5 | Full Game | Full Game |
| side `score` | expected runs | projected points | fair win probability |

## Consumers

`profit-priority` reads the emitted contract through `feeds/nfl_model.py` and refuses to treat a
`RESEARCH_ONLY` forecast as a value edge.

## Design notes

- **Paired de-vigging only.** One-sided de-vigging against an assumed overround is guesswork and
  a common source of phantom edge. An implausible overround raises rather than returning a
  "fair" price.
- **No dependencies.** The deployable layer must price a quote without the research stack
  installed.
- **`lam` is not tuned here.** Tune it in the harness that has an out-of-sample gate to catch it.
