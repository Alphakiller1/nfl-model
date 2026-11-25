# Projection bias audit — November 25, 2026

Eight biases in the player projection layer, each confirmed on a 2024-25
training sample **and** on a 2026 holdout with the same sign. Corrections were
fitted on 2024-25 only; no 2026 outcome chose a constant.

Measured on the structural layer (`audit_player_props.py` runs with
`intelligence_enabled=False`), so these are the projections before
`prop_intelligence.revise`. That is also the layer the prop pricing calibration
is fitted on, so the two are consistent with each other — but production serves
structural + revised, and the revision is measured separately below.

`ratio` is mean actual / mean projected: above 1.0 the model projects too little.

| position : metric | n train | ratio train | t | n hold | ratio hold | t |
|---|---:|---:|---:|---:|---:|---:|
| QB : rush_attempts | 978 | 0.9342 | 3.7 | 292 | 0.8734 | 3.7 |
| RB : anytime_td_probability | 2,645 | 1.1111 | -3.3 | 751 | 1.1252 | -2.0 |
| TE : anytime_td_probability | 1,824 | 1.6060 | -7.6 | 519 | 1.3755 | -3.1 |
| TE : targets | 1,824 | 1.2030 | -11.6 | 519 | 1.0584 | -2.1 |
| WR : anytime_td_probability | 3,922 | 1.3921 | -9.3 | 1,094 | 1.3528 | -4.5 |
| WR : receiving_yards | 3,922 | 1.0439 | -3.1 | 1,094 | 1.0938 | -3.5 |
| WR : receptions | 3,922 | 1.0525 | -4.5 | 1,094 | 1.0784 | -3.6 |
| WR : targets | 3,922 | 1.0543 | -5.4 | 1,094 | 1.1046 | -5.4 |

## Three causes found and fixed

**Quarterback carry prior.** 4.4 per game against a measured 3.44 (2024-25) and
3.20 (2026). A prior a full carry above every quarterback's mean pulls the
position up through the Bayes posterior, and the backfield pool is sized as
`team carries - quarterback carries`, so it also starved the running backs.

**Receiving touchdown rate priors**, roughly half the real rates: WR 0.028 vs
0.0508, TE 0.031 vs 0.0524, RB 0.012 vs 0.0315. The anytime-TD lambda is
`carries x rush_td_rate + targets x receive_td_rate`, so a prior at half
strength carries straight into the probability.

**Target-rate clamp ceiling** 0.98 -> 1.00. A pass attempt has at most one
target, so 1.0 is the physical maximum. This buys nothing measurable and is
kept only because a clamp should not be tighter than the quantity it bounds.

## The WR target shortfall is NOT the clamp

This was the hypothesis and it is wrong, which is worth recording so it is not
re-investigated:

* The clamp never binds. The estimated rate comes out at 0.951-0.959 per club,
  never within 0.04 of either bound.
* The projected receiver pool sums to 0.915 of projected attempts against a
  league-actual 0.954, but that 4-point gap is `GROUP_RESERVE` working as
  intended: a fifth receiver, a fullback and a third tight end draw real targets
  and are not projected, so their share must not be handed to the players who
  are.
* Rewriting the rate as a shrunk ratio rather than a ratio of shrunk halves
  moved the holdout bias by 0.0001. Reverted.

WR targets, receptions and receiving yards remain 5-10% light on both samples,
t between -3.5 and -5.4, cause unknown. It is the largest open defect in the
projection layer.

## Holdout validation

2026, corrections fitted on 2024-25 only, RMSE change after the three fixes and
the refitted intelligence layer:

| metric | v3.0.0 | v3.1.0 structural | v3.1.0 revised | net |
|---|---:|---:|---:|---:|
| QB : rush_attempts | 2.309 | 2.271 | 2.271 | +1.67% |
| QB : rushing_yards | 16.057 | 15.883 | 15.883 | +1.08% |
| QB : passing_yards | 75.774 | 75.774 | 75.436 | +0.45% |
| WR : anytime_td_probability | 0.376 | 0.373 | 0.373 | +0.78% |
| TE : anytime_td_probability | 0.365 | 0.363 | 0.363 | +0.69% |
| RB : targets | 1.740 | 1.740 | 1.727 | +0.74% |
| RB : receptions | 1.504 | 1.504 | 1.494 | +0.67% |
| RB : receiving_yards | 15.816 | 15.816 | 15.758 | +0.36% |
| RB : anytime_td_probability | 0.387 | 0.386 | 0.386 | +0.15% |
| RB : carries | 4.457 | 4.467 | 4.467 | -0.23% |
| RB : rushing_yards | 26.157 | 26.237 | 26.237 | -0.31% |

Ten improve, seven are unchanged, two regress slightly. The regressions are the
backfield pool redistributing once the quarterback stops taking carries he never
had.

## The intelligence layer shrank

Refitting the corrections on the new structural layer **rejected
TE:anytime_td_probability at +0.25%**, which v3.0.0 accepted at +2.171%. That
correction existed to paper over the TD-rate prior now fixed at source; keeping
both would double-count it. The accepted set is down to two families,
QB:passing_yards (+0.35%) and RB:targets (+0.38%).

WR:targets was rejected at -14.64% — the same correction family that would, if
accepted, have masked the open WR bias above rather than explained it.

## Reproduce

    PYTHONPATH=src python scripts/audit_player_props.py --cache data/cache \
        --out <replay> --seasons 2024,2025,2026
    PYTHONPATH=src python scripts/fit_player_intelligence.py --rows <replay> \
        --baseline <old replay> --out <report> --cache data/cache
    PYTHONPATH=src python scripts/fit_prop_pricing.py --replay <replay> --season 2026
