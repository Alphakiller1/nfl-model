# Team volatility rankings — October 7, 2026

## The question

For each team and each market (spread, moneyline, over, under): when the
advanced metrics project this team's game, how reliably does the result land
there, and how often does it stray in the direction that loses that bet? Rank 1
conforms most. Every game is scored against the model's own pre-game numbers,
never the market's.

| Market | Per-game miss, from the team's side |
| --- | --- |
| Spread | `|actual margin − model margin|` |
| Moneyline | `|won − p| − 2p(1−p)`, `p = Φ(model margin / 13.18)`: surprise beyond what `p` itself implied |
| Over | `max(model total − actual total, 0)`: the shortfall that loses an over |
| Under | `max(actual total − model total, 0)`: the overshoot that loses an under |

`volatility.py` is the same file as cfb-model's. The method, and why each piece
exists, is in cfb-model's `reports/VOLATILITY_2026-10-07.md`. In brief:

* **Noise cancellation.** Each game's own first-down rate, explosive-play rate,
  sack rate and plays give a stats-implied margin and total. EPA and turnover
  rate are left out because interceptions and lost fumbles are the luck being
  stripped. The **noise-cancelled score (NC)** is a team's miss against that
  stats-implied outcome instead of the final score.
* **Consistency prior.** Shrink toward what a team's game-to-game SD of
  first-down and explosive rates predicts, rather than toward the league
  average.
* **The bar.** Plain shrinkage, plus noise cancellation, plus consistency, and
  both are each scored leave-one-season-out *and* time-forward. A market ships
  as predictive only if its variant beats "every team is average" overall and
  in most seasons under both schemes.

Data: the time-forward replay in `scripts/audit_regimes.py`, 1,615 games,
2020–2025, 192 team-seasons. Coefficients are fit only on earlier seasons, and
ratings and form are built only from earlier weeks.

## Results

The stats-implied outcome explains **49% of NFL margins and 52% of totals**
without turnovers (luck SD 10.1 points), against 74% of margins in college. That
gap is itself a measure of how much NFL results ride on turnovers.

Held-out skill, LOSO · time-forward (seasons better):

| Market | plain | + noise cancellation | + consistency prior | Ships |
| --- | --- | --- | --- | --- |
| Spread | −1.04% (0/6) · 0.00% (0/4) | same | −0.08% (4/6) · −4.40% (1/4) | descriptive |
| Moneyline | +3.15% (3/6) · +1.19% (2/4) | same | +1.45% (2/6) · −1.94% (2/4) | descriptive |
| Over | −0.40% (0/6) · −4.10% (0/4) | same | −2.27% (3/6) · −8.06% (0/4) | descriptive |
| Under | −5.14% (0/6) · −0.79% (1/4) | −6.83% · −4.27% | −10.32% · −12.51% | descriptive |

**No NFL market passes**, with or without the additions. Under-side volatility
does correlate across seasons (r = 0.18), as it does in college. But with 32
teams a fold is small, and every way of acting on that correlation did worse
than the league mean on the held-out season more often than not. Extending the
plain replay back to 2018 gave the same answer. NFL teams are far more alike
in tempo and style than college programs, and parity compresses exactly the
program-level differences that make CFB totals volatility persist.

## How production uses it

`volatility.build` runs on every site build, from this season's graded shadow-
ledger games (the last snapshot recorded before kickoff). Each game is joined
to both sides' process lines from the nflverse weekly file the form model
already reads. Every market is **descriptive**: it ranks on this season's
observed misses, carries no grade, and says it is what happened, not a
forecast. Every team carries its NC score per market, with `luck` (raw minus
process) in `volatility.json`. If a refit finds a trait, that market switches
to the predictive ranking automatically.

## Audit, October 7

The production matrix was re-derived with independent code against the live
2026 ledger. Every per-game miss, pool and rank reproduced exactly, and the
replay's team alignment was verified on all 1,615 games. The audit also found
that the deploy had failed since October 5. Two release checks demanded all 32
teams, and week 5 has byes. That is fixed separately: the checks now count the
teams actually playing.

## Refitting

    python scripts/fit_volatility.py    # needs numpy, like the other fit scripts

This prints the ablation and writes the evidence to
`reports/volatility_fit.json` and the shipped parameters (plus 2025 per-team
aggregates as the prior season) to `src/nflmodel/volatility_fit.py`.

Research context, not a betting signal. Authority remains RESEARCH_ONLY.
