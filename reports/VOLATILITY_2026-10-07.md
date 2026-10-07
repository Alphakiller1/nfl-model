# Team volatility rankings — October 7, 2026

## The question

For each team and each market (spread, moneyline, over, under): when the
advanced metrics project this team's game, how reliably does the result land
there, and how often does it stray in the direction that loses that bet? Rank 1
conforms most. The question is about conformity to **our** metrics, so every
game is scored against the model's own pre-game numbers, never the market's.

| Market | Per-game score, from the team's side |
| --- | --- |
| Spread | `|actual margin − model margin|` |
| Moneyline | `|won − p| − 2p(1−p)`, `p = Φ(model margin / 13.18)` — surprise beyond what `p` itself implied |
| Over | `max(model total − actual total, 0)` — the shortfall that loses an over |
| Under | `max(actual total − model total, 0)` — the overshoot that loses an under |

The module, `volatility.py`, is the same file as cfb-model's.

## Is it a team trait?

Data: the time-forward replay in `scripts/audit_regimes.py`. Every margin and
total comes from coefficients fit only on earlier seasons and from ratings and
form built only from earlier weeks. That gives **1,615 games, 2020–2025, 192
team-seasons**.

Each team's predicted volatility is its observed mean shrunk toward the league,
`w = n_eff / (n_eff + k)`. `k` and last season's weight are chosen to predict the
second half of a team's season from its first half plus last season, and each
season is scored with parameters chosen on the others. A market counts as a
trait only if held-out skill against "every team is average" is positive
overall **and** in a majority of seasons.

| Market | Held-out skill | Seasons better | First → second half r | Season → next r | Verdict |
| --- | ---: | :---: | ---: | ---: | --- |
| Spread | −3.05% | 0 / 6 | 0.03 | −0.01 | noise |
| Moneyline | +2.81% | 3 / 6 | 0.00 | −0.08 | noise |
| Over | −0.40% | 0 / 6 | 0.13 | 0.07 | noise |
| Under | −5.14% | 0 / 6 | 0.07 | 0.18 | noise |

**No NFL market passes.** Under-side volatility does correlate across seasons
(r = 0.18), as it does in college. But with 32 teams a fold is small, and the
shrinkage chosen on five seasons did worse than the league mean on the sixth more
often than not. Moneyline's pooled +2.8% comes mostly from 2022 (+12.6%), with
three of six seasons worse.

Extending the replay back to 2018 (8 seasons, still time-forward) gave the
same answer: spread −3.7%, moneyline +1.6%, over −3.7%, under −1.5%.

The contrast with college is expected. NFL teams are far more homogeneous in
tempo and style, and parity compresses exactly the program-level differences that
make CFB totals volatility persistent.

## How production uses it

`volatility.build` runs on every site build, from this season's graded shadow-
ledger games (the last snapshot recorded before kickoff). Because every market is
noise, every market is **descriptive**: it ranks on this season's observed misses,
carries no steady/volatile grade, and the page says it is what happened, not a
forecast. The section is "Team Volatility Rankings". The payload is published as
`volatility.json`.

If a refit later finds a trait, that market switches to the shrunk, graded
**predictive** ranking automatically. No code change is needed.

## Refitting

    python scripts/fit_volatility.py    # needs numpy, like the other fit scripts

This writes the evidence to `reports/volatility_fit.json` and the shipped
parameters (plus 2025 per-team aggregates as the prior season) to
`src/nflmodel/volatility_fit.py`.

Research context, not a betting signal. Authority remains RESEARCH_ONLY.
