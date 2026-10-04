# Weekly RB, QB, WR and kicking prop scouting

Contract: `nfl-weekly-schematic-props/1.0.0`

Every scheduled dashboard build produces four separate top tens: RB, QB, WR and
kicking. Each list contains ten **distinct players**, one selected market per
player, when at least ten eligible players and meaningful thresholds exist.
Started games, undated kickoffs, Out/Doubtful/inactive players and immaterial
depth roles cannot fill the lists. A real shortfall is published explicitly.

The section is **Weekly Props** on the dashboard. The identical report is in
`board.json` at `weekly_report.weekly_position_props`, and in `weekly-props.json`.
The existing Tuesday and NFL game-day Pages schedules rebuild it automatically
for the season/week selected by `season.assemble`.

```powershell
python -m nflmodel.cli weekly-props --season 2026 --week 4 --out weekly-props.json
python -m nflmodel.cli build-site --season 2026 --week 4 --out _site/index.html
```

## Data flow and cutoff

1. Keep the existing roster/depth/injury-aware projection matrix and reconciled
   team pass, target and carry pools as the structural baseline.
2. Read the MLBMA public NFL snapshot at
   `https://chase-analytics.com/data/public/nfl/slate.json`. This is a read-only
   consumer of the pipeline powering Chase Analytics; no second scraper or
   circular forecast ingestion is introduced.
3. Accept only the requested season/week, finite/range-checked measures, a
   snapshot and scouting timestamp no later than assembly, and an unstarted
   fixture with a matching kickoff. Both timestamps have a 72-hour age limit.
4. Retain current and combined windows, declared profile source seasons,
   participation seasons, individual split seasons and sample counts. Missing
   current shells may use dated prior/combined evidence without acquiring a
   current-season label. Profile-level dates are not exact per-field charting
   dates; the source contract cannot establish the latter for every field.
5. If the source is unavailable/stale/wrong-week, publish its status and use
   the local pre-week scheme profiles with wider stress scenarios. Do not
   silently backfill advanced individual/trench/red-zone evidence.

Set `NFL_CHASE_CONTEXT_PATH` to a saved snapshot for deterministic local work.
Historical replay needs the snapshot that actually existed at its forecast
cutoff; today's snapshot must not be substituted for it.

## Football evidence for every entry

The expandable dossier covers these families and identifies missing fields:

| Family | Evidence and football implication |
| --- | --- |
| Role | Current slot, continuity, games of history, designation and projected opportunities; backups cannot inherit starter volume independently |
| Intent / tempo | Neutral passing and no-huddle; script changes the late-game allocation of carries and attempts |
| Formation | Shotgun, under center and pistol; presentation is distinct from a blocking assignment |
| Personnel | 11, 12, 21, 13 and 22; receiver/blocker availability and package responses |
| Concepts | Motion, play action, RPO and screens; test the offense's execution of its presumed answer |
| Coverage | Man/zone, Cover 0/1/2/3/4/6/2-Man, single/two-high and available individual responses |
| Pressure / fronts | Blitz versus actual pressure, light/heavy boxes, base/nickel/dime/sub-packages; pressure may remove attempts before creating checkdowns |
| Response | Offensive and defensive EPA/success against the available looks; frequency is not quality |
| Player splits | Counts, season and completion/catch/yardage/TD/interception response by coverage, pressure/clean, blitz/no-blitz, safety, package and box looks; small samples are shrunk. Next Gen time to throw, time to LOS, boxes faced and rushing over expected retain their dates and denominators |
| Run points | Individual left/middle/right and guard/tackle/end usage mapped to opponent yield; these remain proxies |
| Trenches | Stuff and explosive rates, line yards, yards before contact, sacks and hits; available ranks and source dates retained |
| Drive finishing | Scoring-range access, TD/scoring conversion, positional and individual red-zone use; TDs remove FG chances while creating PATs |
| Availability | Own and opponent reports, OL/skill absences, replacement and inactive-list checks |
| Regime | Staff continuity, reaction window, recent flags and bounded reaction weight |
| Environment | Implied points, stadium/roof, surface, rest and travel where observed |
| Missing charting | Individual routes/alignment, press/brackets/shadows/rotations, blocking family/assignments, front techniques/run fits, fourth-down policy, kicker distance and dated wind |

The thesis translates the line into an opportunity hurdle: required yards per
attempt/carry/target or completion/catch rate at projected volume. It names
supporting and opposing evidence, then the concrete ways the case can fail.
Receptions use catch-rate response, not yards-per-target response. Pass/carry
volume propositions do not receive a yardage-efficiency signal.

Player response is weighted by opponent look frequency, shrunk to the player's
all-look baseline with 24 receiver targets or 48 QB attempts of prior weight.
Uncovered looks retain baseline rather than being erased by renormalization.
Look response is compared only with a baseline from the same source season.
Pressure, blitz, shell and package axes overlap: their diagnostics are reported
separately and are never summed as independent mean adjustments. Selected-market
diagnostics retain their own response metric (catch rate for receptions, for
example). Tracking through the forecast week or a future season is excluded;
missing tracking dates remain explicit. Time to throw and boxes faced describe
operating style, not a quality grade or an automatic favorable matchup.
Run-point response uses same-season player usage and a 24-carry opponent-lane
prior. Neither diagnostic is applied to the matrix mean a second time.

Kicking diagnostics compare scoring-range access and conversion with a
club-deduplicated observed league pool (minimum eight clubs), using 16-trip and
6-game combined-window shrinkage. Low TD conversion helps FGs only when
scoring conversion is retained; low TD conversion plus empty drives is not
automatic FG support. Outside-red-zone kicks and fourth-down choices remain
unmeasured. Conflicting drive evidence widens the stress envelope.

QB rushing requires separate individual rushing evidence: RB front tendencies
cannot establish designed QB runs, scrambles or kneels. Those standalone QB
markets currently stay out of this ranking when that charting is absent.

## Likelihood and ordering

The report evaluates all supported markets, then chooses one per player.
Quoted calibrated markets come first, quoted research extensions second,
research milestones third. Within each tier, descending **worst-case hit
likelihood across the disclosed stress scenarios** determines rank. Base hit
likelihood and opposing evidence break ties. It does not rank by dollars of
yardage, a generic scheme bonus, or expected monetary value.

For ordinary quoted passing/rushing/receiving markets, use the existing
projection-error distributions and `prop_pricing` line calibration. A discrete
push is separated from wins/losses, with calibration conditional on no push.
Empirical continuous ladders use half-unit boundaries for lattice outcomes.

FG/PAT counts use a **Poisson assumption**, explicitly unvalidated for kicking.
Combined rush/receiving yards, pass/rushing yards and kicking points use
Frechet/union bounds on the marginal models. No independent-component or
correlation fit is invented. The displayed interval is a dependence bound,
not a confidence interval. Integer combined book lines are withheld because
their push mass is unmeasured.

The base mean stress is ±15%, or ±25% for kickers. Add five percentage points
for limited/new-role history, five for missing dated advanced context, five
for opposing schematic evidence, and ten for the player's injury designation;
cap at ±40%. Price the same selection at reduced, base and expanded means and
rank on its lowest hit probability. These are disclosed sensitivity assumptions,
**not learned penalties or statistical confidence intervals**.

If no eligible book line exists, choose from a fixed standard milestone grid
nearest 65% raw/bounded over probability, restricted to 45–85%. A research
milestone has `line=null`, no book/price, and no line-calibration claim. It does
not masquerade as a posted alternate market. Missing prices on ESPN lines are
also null; nominal -110 values are never published as executable prices here.

All rows retain `RESEARCH_ONLY` and `may_bet=false`. The new rankings, stress
envelopes, kicking and combined-market extensions need forward validation;
the existing pricing evidence is reported as its own lineage, not as evidence
that this new forty-entry selection policy is already profitable.

## Provider coverage and validation

ESPN's observed full-game names now include passing attempts, rushing plus
receiving yards, passing plus rushing yards, field goals, kicking points and
extra points. Period and milestone markets remain excluded. Identity uses
the roster's ESPN-to-GSIS join; fixture identity is checked again at ranking.

The Odds API request adds its documented five combined/kicking markets to
the prior seven: 12 markets, approximately 192 credits for sixteen games in
one region when the paid provider is used. Its existing 30-hour lead window,
20-hour cache and provider fallback remain in effect.

Regression coverage verifies four unique-player top tens, explicit shortfalls,
source date/range gates, same-season conditional/selected-market responses,
tracking cutoff/sample lineage, correct reception response, run-point shrinkage,
opposite FG/PAT conversion implications, quote/fixture identity, injury and
kickoff exclusions, push conservation, dependence bounds, HTML escaping and
one shared report across consumers. Production verification includes the
existing dashboard smoke, build and consistency audits plus desktop and 375px
mobile checks of all four groups, evidence expansion and the JSON export endpoint.
