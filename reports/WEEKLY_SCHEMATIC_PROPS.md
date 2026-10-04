# Weekly RB, QB, WR and kicking prop scouting

Contract: `nfl-weekly-schematic-props/2.0.0`

Every scheduled dashboard build produces four separate top tens: RB, QB, WR and
kicking. Each list contains ten **distinct players**, one selected market per
player, when at least ten eligible players have supported PrizePicks projections.
Started games, undated kickoffs, Out/Doubtful/inactive players and immaterial
depth roles cannot fill the lists. A real shortfall is published explicitly.
These four lists use **PrizePicks lines only**. Sportsbook lines and generated
milestone thresholds cannot fill a missing PrizePicks slot.

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

## PrizePicks line source

Read the actual projection cards on PrizePicks' public first-party NFL player
research pages at `https://www.prizepicks.com/research/nfl/players/`. The direct
API is not required (it returned HTTP 403 during implementation). Every line
retains its projection ID, original stat label, source URL and observation time.
The adapter verifies the published player name, club/position, opponent, next-game
ISO kickoff and each card's Eastern date/time against the model's fixture.
Missing IDs, other leagues, partial-game periods and unsupported stats fail closed.

Four workers fetch eligible player pages with a 15-minute cache and bounded
responses/timeouts. HTTP `Age` is retained in the observation time. Lines over
one hour old, undated or observed after assembly are excluded. Failed pages
are reported; there is no cross-provider fallback. Set
`NFL_PRIZEPICKS_SNAPSHOT_PATH` to a saved `nfl-prizepicks-lines/1` snapshot for
deterministic/offline builds. A saved snapshot receives the same provider,
identity, fixture, source attribution and freshness gates at ranking.

Public cards show More/Less labels but do **not** identify standard/Goblin/Demon
status, per-contest availability or payout. Every row labels these as unverified
and requires checking the current offered side/variant in the app. Alternate
thresholds remain their actual observed values; none is guessed to be the
standard line. Rankings measure modeled stat-threshold likelihood, not contest
return, executable prices or a complete lineup's payout. A source carrying only
one published side can never produce a pick on the other side.

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

The report evaluates supported observed PrizePicks lines, then chooses one per
player. Descending **worst-case hit likelihood across the disclosed stress
scenarios** determines rank. Base hit
likelihood and opposing evidence break ties. It does not rank by dollars of
yardage, a generic scheme bonus, or expected monetary value.

For ordinary passing/rushing/receiving markets, use the existing fitted
projection-error distributions. The DraftKings closing-line calibration has no
PrizePicks validation record and is **not applied** to these selections. All
PrizePicks probabilities are labeled uncalibrated. A discrete tie is separated
from wins/losses; injury/DNP/Reboot settlement is not modeled as a stat win.
Empirical continuous ladders use half-unit boundaries for lattice outcomes.

FG/PAT counts use a **Poisson assumption**, explicitly unvalidated for kicking.
Combined rush/receiving yards, pass/rushing yards and kicking points use
Frechet/union bounds on the marginal models. No independent-component or
correlation fit is invented. The displayed interval is a dependence bound,
not a confidence interval. Integer combined PrizePicks lines are withheld because
their tie mass is unmeasured.

The base mean stress is ±15%, or ±25% for kickers. Add five percentage points
for limited/new-role history, five for missing dated advanced context, five
for opposing schematic evidence, and ten for the player's injury designation;
cap at ±40%. Price the same selection at reduced, base and expanded means and
rank on its lowest hit probability. These are disclosed sensitivity assumptions,
**not learned penalties or statistical confidence intervals**.

If no eligible PrizePicks projection exists, publish an empty slot. Every row
has an actual non-null line and PrizePicks attribution. Price is null; no
bookmaker odds, nominal -110 prices or contest payouts are invented.

All rows retain `RESEARCH_ONLY` and `may_bet=false`. The new rankings, stress
envelopes, kicking and combined-market extensions need forward validation;
the sportsbook pricing evidence is not presented as PrizePicks calibration
or as evidence that this selection policy is profitable.

## Provider coverage and validation

The current PrizePicks cards include passing yards/attempts, pass/rush yards,
rushing yards/attempts, receiving yards, rush/receiving yards, receptions,
field goals, PATs and kicking points.
Other allowlisted supported labels are accepted only when present on a real
card; a model metric alone cannot create a projection. QB standalone rushing
still requires its separate scheme evidence. WR rush/receiving totals are
withheld when the model lacks an independently modeled WR rushing component.

Existing game-line and legacy sportsbook exports retain their separate source
paths. The dedicated `prizepicks_quotes` field feeds these four lists, so a
DraftKings/ESPN/Kalshi quote cannot leak into them. The five additional paid
Odds API requests originally proposed for this feature were removed; the
legacy request retains its existing seven-market credit footprint.

Regression coverage verifies PrizePicks-only four unique-player top tens, no
milestone/sportsbook fallback, actual card IDs/fixture dates, observed-side
restrictions, uncalibrated probabilities, explicit shortfalls,
source date/range gates, same-season conditional/selected-market responses,
tracking cutoff/sample lineage, correct reception response, run-point shrinkage,
opposite FG/PAT conversion implications, quote/fixture identity, injury and
kickoff exclusions, push conservation, dependence bounds, HTML escaping and
one shared report across consumers. Production verification includes the
existing dashboard smoke, build and consistency audits plus desktop and 375px
mobile checks of all four groups, evidence expansion and the JSON export endpoint.
