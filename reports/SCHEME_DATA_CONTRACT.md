# NFL scheme matrix data contract

Version: `nfl-scheme-matrix/1.0.0`

The scheme layer is a point-in-time research challenger for offensive player and
kicker projections. It does not modify the published spread model. Every play is
filtered to strictly before the forecast week, weighted with a 16-week half-life,
shrunk toward the league distribution, and discounted to 55% carryover after a
head-coach change (80% when staff continuity cannot be verified).

## Observed features

| Family | Public field(s) | Use |
| --- | --- | --- |
| Play calling | `pass`, `rush`, `down`, `qtr`, score/WP | neutral early-down pass rate |
| Formation | `offense_formation`, `shotgun` | shotgun, under center, pistol |
| Personnel | `offense_personnel`, `defense_personnel` | 11/12/21/13; base/nickel/dime |
| Coverage | `defense_man_zone_type`, `defense_coverage_type` | man/zone and Cover 0/1/2/3/4/6/2-Man |
| Front/pressure | `defenders_in_box`, `n_blitzers`, `was_pressure` | average/stacked box, blitz and pressure rate |
| Concepts | FTN motion/play-action/RPO/screen flags | concept frequency and defensive response |
| Outcomes | `epa`, `success`, `receiver_player_id` | response EPA and RB/WR/TE target allocation |

The exported artifact reports source seasons separately for play-by-play,
participation and FTN charting. That matters because nflverse documents that
participation data from 2023 onward is published after the postseason; a current
play-calling profile can therefore coexist with prior-season coverage/personnel.

## Proxy-only and unavailable fields

`run_location` and nflverse `run_gap` (`guard`, `tackle`, `end`) are run-point
descriptions. They are labelled proxies and are not treated as line blocking
scheme. The public files do not contain offensive-line zone/gap/power/man family
or individual blocking assignments. Both are exported as unavailable.

## Projection coupling

For each offense/defense pairing, the matrix estimates the opponent coverage mix,
blends offensive and defensive response EPA, and calculates position target
response. Adjustments are bounded:

- pass attempts: ±2.0
- carries: ±1.6
- pass efficiency delta: ±0.35
- rush efficiency delta: ±0.25
- RB/WR/TE target multiplier: 0.88–1.12

Target multipliers alter allocation before targets are re-normalized to the same
team pool, preventing the matrix from manufacturing aggregate opportunities.
Every prospective player ledger snapshot stores the exact scheme context used.

## Source and attribution

- nflverse play-by-play and participation releases:
  https://github.com/nflverse/nflverse-data
- nflreadr participation documentation and timing:
  https://github.com/nflverse/nflreadr/blob/main/R/load_participation.R
- nflreadr FTN charting documentation:
  https://github.com/nflverse/nflreadr

Participation data from 2023 onward is FTN Data via nflverse, CC-BY-SA 4.0.
