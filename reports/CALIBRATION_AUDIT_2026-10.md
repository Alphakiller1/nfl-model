# Model-wide calibration and consistency audit (2026-10-04)

Triggered by GB @ TB, week 4 2026: the board published "Over 38.5, 76%" on a
47.8 model total while Baker Mayfield was out and every scheme note on the game
pointed to less offense. Three classes of mistake were behind it, and the audit
looked for every other instance of each.

## 1. Probabilities that treated the model as the truth

| market | measured skill vs the close (time-forward) | was priced as | now |
|---|---|---|---|
| spread | blend weight -0.035; ATS 49.1% at 2+ pt gaps (792), 51.6% at 6+ (95); MAE 10.15 vs 9.76 | Phi(gap / 13.2): a 9-pt gap showed 75% | `calibration.SKILL["spread"].weight = 0`: no spread picks |
| total | blend weight 0.017; 52-52 at 6+ pt gaps | Phi(gap / 12.9) | weight 0.02: no total picks |
| props | replay leans >= 53% went 450-388 held out | calibrated (prop_pricing) | unchanged; now also gated by the slip evidence rules |
| board win probability | read off the market margin | correct | unchanged |

Every published probability now comes from `calibration.py` (game markets) or
`prop_pricing.py` (props), both priced on measured skill. A market whose weight
is ~0 publishes nothing however large the gap looks.

## 2. An adjustment applied to one output but not its siblings

* **QB out moved the margin, not the total.** Fitted: -3.13 total points per
  missing starter (se 0.82), team points -3.65 (t -6.1). Applied (#27).
* **Checked, not needed:** top target earner out -0.50 team points (t -0.6),
  lead back out -0.43 (t -0.5): noise at the game level; the prop layer
  already redistributes their volume.
* **Best bets vs slips:** best-bet props published an RB carries under (a
  24-26 market), a depth-3 under and a 0.5 line that the slip rules refuse.
  Best bets now use the slip rules (`slips.candidate_legs`).

## 3. Explanations that contradicted the number

* The gap tables' "Scheme explanation" column implied the scheme drove the gap;
  it is now "Matchup context (not a model input)", and game-pick angles say so.

## The guard

`scripts/audit_consistency.py` runs on every deploy after `verify_build.py` and
fails the build when the board contradicts itself: a pick priced above its
market's measured skill or below the publish floor, a pick whose side
disagrees with its own numbers, a scoreline that does not add up to its total
or margin, a win probability favouring the other side, a QB-out game without
both adjustments, a projected player listed Out, team targets above team
attempts, a negative projection, a prop pick or slip leg outside the evidence
rules. Unit tests cover each check (`tests/test_audit_consistency.py`).

Reproduce the measurements: `research/props_model/model_audit.py` and
`qb_out_totals.py`. Refit `calibration.SKILL` after each season.
