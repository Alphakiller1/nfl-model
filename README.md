# nfl-model

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
```

`slate.json`:

```json
[{"home_team": "KC", "away_team": "LAC", "home_american": -175, "away_american": 150}]
```

Output carries the price, both American lines, the permitted action, and the unmet gates.
Games without a paired price are reported as `AVOID` in `skipped` rather than dropped — a
silently shorter list is indistinguishable from a slate with no games.

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
