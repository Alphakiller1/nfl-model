# Player props: game script, pace, tendencies and pricing

Study and rebuild, 2026-10-03. Projection layer `nfl-player-projections/2.0.0`
(`src/nflmodel/player_props.py`); pricing `src/nflmodel/prop_pricing.py`;
DraftKings lines `src/nflmodel/sources/espn_props.py`. Every number here can
be reproduced from `research/props_model/` (see the README there).

## The question

Can the stats we have (nflverse play-by-play 2021-2026, weekly player stats,
snap counts, depth charts, injuries, FTN/participation scheme data, and the
market's spread and total) project player volume and efficiency well enough
to beat DraftKings' player lines? And which of those stats carry predictive
signal, as opposed to describing what already happened?

Short answer: **volume and role carry the signal; matchup quality does not,
because the book already prices it.** The projection layer got measurably
better, but out of sample it does not beat the line yet. Props stay research
only until the ledger, which now records the line on every projection, says
otherwise.

## Method

The models are scored only on games after the data they were fitted on.

* **Reliability (signal vs noise).** For a per-game stat, the season
  split-half correlation r (odd vs even weeks) gives the shrinkage constant
  `k = n_half * (1 - r) / r`. That is the number of games (or attempts) at
  which a team's or player's own average earns half weight against the prior.
  Big k means mostly noise.
* **Recency.** The estimate is an exponentially weighted mean with half-life h,
  shrunk toward a prior worth k pseudo-games. For team volume the prior is
  `league + c * (last season - league)`, where c is the share of last season's
  deviation that survives the offseason. h, k and c are grid-searched to
  minimise next-game error, walking forward through 2022-2025.
* **Projection replay.** `research/props_model/replay.py` rebuilds every 2025
  week (2-18) exactly as the board would have: the roster, depth chart and
  injury report as of that week, and history only up to the week. That is
  5,098 player-weeks. Because the published number is a **mean**,
  improvements are judged on RMSE and bias. MAE rewards projecting the median,
  which is how the v1 layer drifted low (see C3).
* **Market test.** ESPN's core API serves DraftKings' player over/unders for
  every 2026 game, including finished ones. Those lines were joined to the
  ledger's latest pre-kickoff projection and the box score: 2,003 lines from
  2026 weeks 1-4.

## A. Team volume and game script

### A1. What is signal

| per team-game | split-half r | k (games) | year-over-year r |
|---|---|---|---|
| neutral pass rate over expected (PROE) | 0.57 | 6.3 | 0.33 |
| neutral seconds per play (pace) | 0.58 | 6.1 | 0.43 |
| pass attempts | 0.59 | 5.9 | 0.43 |
| designed runs | 0.35 | 15.7 | 0.39 |
| offensive plays | 0.29 | 20.4 | 0.15 |
| *defense:* pass attempts allowed | 0.28 | 21.5 | |
| *defense:* plays allowed | 0.11 | 71.5 | |
| *defense:* pace allowed | 0.12 | 63.5 | |

Offensive tendencies are real and stabilise in about 6 games. What a
defense "allows" in volume is 3-10x noisier. v1 weighted the opponent's
allowed volume at 42% with 8 pseudo-games, so it over-trusted defense.

### A2. How fast tendencies move

Best forward fit, 2022-2025:

| | half-life | prior games k | carry-over c |
|---|---|---|---|
| offense pass attempts | 4 | 4 | 0.5 |
| offense designed runs | 4 | 6 | 0.3 |
| offense PROE | 4 | 4 | 0.3 |
| offense pace | 8 | 4 | 0.3 |
| defense pass attempts / runs allowed | 4 | 10 | 0.3 |

Tendencies shift within a season: a 4-game half-life beats every longer
memory, including v1's 12 weeks. Only 30-50% of last season's deviation
survives the offseason.

### A3/A4. The game-script mapping

Market-only slopes per point of the team's spread (favourite > 0) and per
point of game total, 2022-2025 team-games:

| outcome | per spread point | per total point |
|---|---|---|
| offensive plays | +0.143 | +0.093 |
| dropback rate | -0.002 (-4 pts across ±10) | +0.003 |
| **pass attempts** | **-0.013** | **+0.250** |
| **designed runs** | **+0.203** | -0.126 |
| mean score differential at the snap | +0.591 | |

The mechanism: a favourite leads (≈0.6 points of in-game lead per point of
spread), so it drops back less often, but it sustains drives and runs more
plays. The two effects cancel for passing. **The spread moves runs, not
passes. The total moves passes.** v1 applied -0.16 attempts and +0.13
carries per point of margin and ignored the total, so it projected underdog
passers about 1.6 attempts too high per 10 points of spread.

Pre-game volume model (deviation form, fitted on 2022-2025, all priors
point-in-time):

```
attempts = L + 0.905 (off_prior - L) + 0.585 (def_prior - L) + 0.021 spread + 0.175 (total - 44.5)
carries  = L + 0.924 (off_prior - L) + 0.738 (def_prior - L) + 0.103 spread - 0.102 (total - 44.5)
```

Time-forward MAE, pass attempts: v1 6.04 / 6.05 (2024 / 2025) vs 5.98 / 5.92.
Designed runs: 5.41 / 5.51 vs 5.42 / 5.47. Adding pbp pace, PROE and the
opponent's pace on top changed nothing (5.88-6.03), so the shipped model
runs on the weekly player stats the pipeline already loads. The ceiling is
low: the game-to-game standard deviation of team pass attempts is 7.8, and
roughly 6 of it cannot be predicted before kickoff.

## B. Player usage and efficiency

### B1. Usage shares

Next-game target and carry share (2023-2025, players who played) is best
predicted by a **5-8 game half-life** with almost no shrinkage toward the
positional mean (k ≈ 0.5 games). Usage is the most persistent thing in the
data.

### B2. Snap share

Adding last game's snap share to the trailing share cut share MAE by 1-2%.
Shipped as a trend multiplier, `share × (last snap % / weighted snap %)^β`
with β = 0.3 for targets and 0.4 for carries. On the 2025 replay it is
worth about -0.3% RMSE on targets and receptions and is neutral on yards.
Snap counts come from nflverse (`snap_counts_{season}.csv`) and join through
the roster's `pfr_id`.

### B3. Injury redistribution

When a regular with at least 12% target share misses a game (541
team-games), his teammates' shares move as:

```
share = trailing × (1.026 - 0.227 × missing_share_all + 0.522 × missing_share_same_position)
```

Same-position teammates absorb the volume; other positions slightly lose it.
Fully proportional redistribution overshoots by +1.7 share points and half
proportional is about right. In the replay, separate pools for WR, TE and RB
lost to one shared WR+TE pool, while a separate RB pool won for RB targets
and TDs. The shipped allocation is team targets -> {WR+TE, RB} pools ->
players, each pool keeping a reserve for players off the projected depth
slots (WR 7.6%, TE 12.8%, RB 4.5% of the group; 3.5% of carries).

### B4. Efficiency is mostly noise

Split-half pseudo-counts (opportunities at which a player's own rate earns
half weight):

| rate | k | v1 prior |
|---|---|---|
| WR yards / target | 119 targets | 38 |
| TE yards / target | 141 | 34 |
| RB yards / carry | 318 carries | 45 |
| WR catch rate | 85 | 34 |
| receiving TD rate | 570 | 55 |
| rushing TD rate (RB) | 390 | 55 |
| QB completion % | 440 attempts | 75 |
| QB yards / attempt | 190 | 75 |
| QB TD rate | 540 | 90 |
| QB INT rate | no signal (r = -0.03) | 100 |
| **WR aDOT (air yards / target)** | **22** | n/a |

Only the QB values shipped. For receivers and backs, the split-half values
lost to v1's lighter priors on the replay: a multi-season weighted history
holds more talent signal than one season's halves. Yards per target is now
shrunk toward a prior read off the player's own aDOT (stable, k = 22).
That is principled but nearly neutral in the replay (≤0.2%). QB priors are
centred on the recency-weighted league rate rather than a constant. A fixed
6.95 yards per attempt sat 0.1-0.2 below every season since 2023.

### B5. Red-zone share does not help TDs

Anytime-TD Brier on 2023-2025 (fitted 2021-22): volume + TD rate 0.1310;
adding red-zone and inside-10 usage 0.1310; red-zone usage alone 0.1334.
Once a player's volume is known, his red-zone share adds nothing. Not
shipped.

## C. The market

### C1. Projection vs DraftKings closing line (2026 weeks 1-3, v1 projections)

| market | n | MAE line | MAE projection | blend weight w [90% CI] |
|---|---|---|---|---|
| receiving yards | 588 | 20.3 | 21.4 | 0.37 [0.16, 0.60] |
| receptions | 581 | 1.54 | 1.59 | 0.32 [0.15, 0.55] |
| rushing yards | 276 | 16.7 | 17.8 | 0.21 [-0.08, 0.52] |
| passing yards | 93 | 60.8 | 62.3 | 0.37 [-0.09, 0.81] |
| carries | 132 | 3.73 | 4.15 | 0.04 [-0.20, 0.27] |

The line beats the projection on every market. The projection still adds
something for receiving: `line + w (projection - line)` beats the line, with
CIs from a game-clustered bootstrap. v1 priced props by reading the raw
projection through a skewed outcome ladder. Its P(over) had a **worse
Brier than a coin** in every market except passing, and its ≥55% plays hit
42-55%.

### C2. Matchup metrics against the line

Correlation with (actual - line) on the receiving markets (n = 1,169):

| signal | r |
|---|---|
| the model's own usage gap | **+0.18** (receiving yards), +0.12 (receptions) |
| opponent pass EPA allowed | -0.00 |
| coverage-based target multiplier | +0.01 |
| scheme pass-efficiency delta | -0.01 |
| opponent rush EPA allowed | +0.04 |

Defense quality, coverage shell and scheme rates do not predict where a
player lands **relative to the line**: DraftKings has already priced them.
Combined with the 2026-09-29 result (adding the matchup page's pbp families
to the game model moved MAE by noise), this is why the rebuild spent its
effort on volume and role.

### C3. Mean vs median

v1 was tuned on MAE. MAE is minimised by the median, which for skewed
yardage sits below the mean, so changes that pushed projections down "won"
(`ACTIVE_RATE` cut MAE 5-7% that way). The pricing layer then treated the
number as a mean and applied a skewed ladder on top, counting the skew twice
and leaning under (P(over) on carries averaged 0.39 against a 0.52 hit
rate). v2 treats the projection as an unbiased mean and is judged on RMSE
and bias.

### C4. Line-anchored pricing

```
fair   = line + w_family × (projection - line)
P(over) = sigmoid(a_family + b_family × (projection - line) / sqrt(line))
```

Refitted with `scripts/fit_prop_pricing.py` (leave-one-week-out) on 2,003
lines from 2026 weeks 1-4.

**Correction, same day.** The first fit had a free intercept a. Held out it
went 238-251 (48.7%), and most of those plays were the intercept, not the
gap: receiving overs hit 46% early in the season, so the model priced 269
receiving unders and 0 overs in week 1. That is a constant applied to every
player, not a read on any of them, and without prices it may only be the
book's vig shading. With a fixed at 0 (pricing through 50%), the slopes are
receiving -0.008, rushing 0.016 and passing 0.239. Held out, 69 plays went
32-37, and passing alone went 28-24. **Only passing shows a per-player
signal, and on far too few plays to trust.**

This pricing was replaced by pricing from the matrix's own outcome distribution
(section E), which publishes leans every week with their measured record.

## D. What shipped (v2.0.0) and what it did

2025 replay, v1 -> v2 (RMSE, bias):

| stat | RMSE | bias |
|---|---|---|
| QB pass attempts | -3.5% | +1.09 -> +0.57 |
| QB completions | -3.6% | +0.87 -> +0.58 |
| QB passing yards | -3.0% | +8.0 -> +4.3 |
| QB passing TDs | -3.0% | +0.04 -> -0.00 |
| RB carries | -2.8% | -1.05 -> +0.17 |
| RB rushing yards | -0.8% | -4.7 -> +0.6 |
| RB targets | -1.4% | -0.05 -> +0.03 |
| WR targets / yards | -0.1% / +0.2% | -0.20 -> +0.16 / -0.8 -> +1.8 |
| TE targets / yards | -0.4% / -1.2% | -0.35 -> 0.00 / -2.5 -> +0.1 |
| anytime TD Brier | 0.1505 -> 0.1502 | |

Changes:

1. Team volume: the A3 model with 4-week half-life priors. The spread moves
   carries and the total moves attempts.
2. `ACTIVE_RATE` scales a slot's claim before the pool is divided, so an
   unlikely-active slot's volume goes to teammates instead of vanishing (v1
   deleted ~3 carries a game from every backfield).
3. Targets: team -> {WR+TE, RB} pools with measured reserves. The starter's
   share of team attempts is 0.954, not 0.97.
4. Usage: 6-week half-life, with observed usage weighted by same-team
   evidence (`W / (W + 1)` usage-weighted games).
5. QB rates: split-half pseudo-counts, centred on the weighted league.
6. Snap-share trend (B2), fail-soft when snap counts are missing.
7. DraftKings player lines from ESPN when the Odds API returns none, matched
   by ESPN athlete id -> gsis. The board now publishes `player_prop_source`.
8. Line-anchored pricing (C4) through 50% (no base-rate intercept). No prop
   picks are published until the held-out record clears break-even.
9. The ledger stores the line, open line and price on every player
   projection, grades each against the line, and summarises `vs_line` by
   family.

## E. The prop matrix (v2.1.0)

The projection is a fitted matrix: for each stat, a log-link (Poisson)
regression of the outcome on a structural baseline plus schematic factors,

```
log E[stat] = log(team volume x usage share  [x efficiency prior]) + sum_k beta_k x_k
```

fitted 2022-23 and 2022-24 and scored on 2024 and 2025 (players who played,
point-in-time features: `research/props_model/m0_features.py`,
`m1_matrix.py`). Factor families and their held-out effect:

| family | factors | held-out deviance |
|---|---|---|
| usage | share regression to position mean, snap trend, games on team | **-5 to -10%** (every stat) |
| box | opponent heavy-box (8+) rate on runs, FTN | carries and rush yards better in 2024 and 2025 |
| script | spread, total, team pass EPA | small, mixed |
| opponent | position-group target share, YPT, catch rate allowed, pass EPA allowed | ~0; dropping it is often better |
| pressure | sack+hit rate matchup, x aDOT, x RB | ~0 |
| coverage | player man/zone target-rate tilt x opponent man rate (prior season) | ~0 |
| blitz | opponent blitz rate x RB / TE | ~0 or worse |

The schematic coefficients have sensible signs (a man-beater against a
man-heavy defense gains targets, pressure shortens deep roles, a defense that
funnels targets to tight ends feeds them), but against out-of-sample noise
they do not predict, so they are not applied. Shipped: usage shrinkage for
carries (`USAGE_SHRINK`), the box factor (`BOX_EFFECT`: per 10 points of
heavy-box rate, carries x0.90, yards x0.87).

**Depth slots are filled after injuries.** The largest single gain in this
round was not a factor at all: the depth reader dropped Out players without
promoting the next man, so a team with three receivers out projected two
receivers to split the whole WR+TE pool (12.7 targets for a 5-target
player). Filling vacated slots in chart order: WR targets -1.9% RMSE, TE
-2.3%, RB carries -2.0%, WR yards bias +1.8 -> +0.7, and 361 more
player-weeks projected in 2025.

### Pricing from the matrix

`prop_distributions` is now fitted on the projection layer's own errors
(2025 replay, players who played; `dist_fit.py`). Held-out second half of
2025: 10-90 coverage 0.82-0.86 for receiving and rushing, 0.73-0.78 for QBs.

`prop_pricing.price`: raw = P(over) from that distribution at the matrix
projection; p_over = 0.5 + s (raw - c), s = 0.30, c = 0.505 (raw 70% hit 57%
against the line, so it is shrunk toward even). Tested on 2026 weeks 1-3,
the matrix rebuilt point-in-time and priced against 1,920 DraftKings closing
lines (`scripts/fit_prop_pricing.py --rows`):

| test | record |
|---|---|
| plays >= 53%, calibration fitted leave-one-week-out | **450-388 (53.7%)**, Brier 0.2499 vs coin 0.2500 |
| fit-free ranking: top third over, bottom third under, per week x market | **688-568 (54.8%)** |
| over plays vs the over base rate (receiving) | 53.8% vs 48.7% |
| under plays vs the under base rate (receiving) | 54.8% vs 51.3% |
| by week | 51.0%, 55.2%, 55.5% |

That beats the -110 break-even and a coin at a one-sided 90% bound, but
only just, on three weeks. Leans are published every week (one side per
line, `lean` below 53%), each with this record in its text, and every line
and price is graded in the ledger so the record updates itself.

**Why the earlier pricing was withdrawn.** C4's logistic carried a free
intercept: the season's over/under base rate (receiving overs 46%). It
priced 269 receiving unders and 0 overs in week 1. That was a constant, not
a read on any player; through 50% its slopes were ~0. Pricing now starts
from the projection's own distribution, with no base-rate term.

## H. Weekly pick'em slips (`slips.py`)

The strategy used for week 4 of 2026, as code. Every build publishes a slip
plan (`board.json` `prop_slips`, the "Pick'em slips" section) and the ledger
grades it (`summary.prop_slips`).

**Which legs.** A leg must clear every rule; each dropped leg is counted by
reason on the board.

| rule | why |
|---|---|
| (position, market, side) record >= 56% on >= 15 replay plays | RB catch unders 33-16, TE catch unders 36-25, TE yardage unders 25-14, QB rush and passing-TD unders; RB carries unders (24-26) and WR catch unders (43-38) never qualify |
| calibrated lean >= 53% | |
| starter (depth 1-2), not Questionable/Doubtful, >= 2 games this season | backup unders 44-39 in the replay; no read on players without history |
| no role expansion: a recent regular (>= 15% target share, or >= 8% at the same position, or >= 30% carry share) missing from this week's projections, or a new starting QB (also blocks QB rushing unders when a target earner or lead back is missing) | the week-4 audit's misses: Keenan Allen out (Downs), Jefferson out (Aaron Jones), Caleb Williams out (Swift) |
| no line moved against the pick by a unit (or 10%) | |
| no 0.5 lines | juiced and non-standard on pick'em |
| no recent form against the pick (cleared the line in 2 of the last 3) | judgment guard; measured by the ledger |

**Which format.** The plan's target is the owner's: six $25 entries in promo
credits, cash payout above $50. For each format (2-pick and 3-pick Power, 3-
and 4-pick Flex) the strongest legs are snake-drafted into slips (one leg per
player, never two legs from one game in a slip) and simulated: each leg's
edge trusted at the replay's measured rate (53%), a shared game shock (rho
0.15) and a shared week shock (0.05). The format with the highest chance of
clearing the target wins; for week 4 that is 6 x 2-pick Power, ~85%.

**Fresh injuries.** The nflverse report lagged game-day news (Keenan Allen
Questionable there, Out on ESPN), so ESPN's game-summary designations,
including IR, are merged over it before projecting
(`sources/espn_injuries.py`). That also corrects the projections: Josh
Downs' week-4 targets went from 6.6 to 8.3.

**Measuring the rules.** Legs a rule drops are stored with the plan and
graded as if played; `summary.prop_slips.dropped_by_rule` reports their
record. A rule earns its place when what it drops loses.

## F. What did not work, kept so nobody re-runs it blind

* Matchup metrics (defensive EPA, coverage, scheme) against the line: r ≈ 0.
* Pace and PROE on top of the volume priors: no gain.
* Red-zone share for anytime TDs: no gain once volume is known.
* Split-half efficiency priors for skill positions: lost to lighter priors.
* Separate WR and TE target pools: lost to one shared pool.
* Opponent, pressure, coverage and blitz factors in the prop matrix (E).
* A free intercept in prop pricing (a base rate, not a prediction).
* Centring each week's gaps per market: no change (49.3% either way).
* Fully proportional injury redistribution: overshoots.

## G. Next

* Let the ledger accumulate v2 projections with their lines, and refit
  pricing weekly (`scripts/fit_prop_pricing.py`). The gate opens itself if
  the held-out record earns it.
* The pricing fit so far is on v1 projections; v2's gap to the line will
  differ.
* ESPN publishes lines but not prices. Shading (for example -130 unders on
  receiving yards) is invisible, so a 50/50 de-vig overstates any under edge.
  The receiving intercept (-0.18) is the measured over/under lean, not a
  price.
* Route participation would sharpen usage beyond snaps. Public
  participation data for 2026 does not exist yet.
