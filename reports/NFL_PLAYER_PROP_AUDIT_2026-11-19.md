# NFL player-projection audit — November 19, 2026

This revision fixes injury ingestion and current-role allocation, adds measured process and tendency context, and estimates uncertainty separately for each stat. Historical point-forecast gains are small and mixed. This audit does not establish a profitable betting edge.

## Injury defect and correction

The November 19 04:16 UTC published snapshot gave Woody Marks 10.45 carries despite his Out designation, leaving David Montgomery at 9.64 carries and 37.79 rushing yards. ESPN's game-summary widget contained only five Houston injury entries and omitted both players. A successful response did not establish complete injury coverage.

Ingestion now starts with the complete league feed, resolves roster IDs from athlete links, checks season/team coverage and dated reporting-week evidence, and marks the partial summary fallback as degraded. A current official Out designation cannot be overwritten by a weaker secondary status. Unavailable previous-week reports are not reused as current reports.

The refreshed local Week 11 build excludes Marks and allocates Montgomery 17.38 carries and 68.10 rushing yards; Demercado receives 5.35 carries. Opportunity is redistributed within the team budget. Montgomery's recent workload gap is separately flagged as uncertain. Questionable/doubtful teammates produce explicit availability assumptions without invented probabilities of playing.

The [Texans' official Week 11 report](https://www.houstontexans.com/news/week-11-injury-report-texans-vs-colts) lists Marks Out and Montgomery with full participation and no final game designation. The [complete ESPN source](https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries?limit=1000) supplements official reports.

## Structural and forecasting changes

- Current-team-stint usage determines opportunity. Old-team shares do not become current-team role evidence. Historical skill can transfer.
- Each club's depth evidence is bounded by its own kickoff, allowing legitimate Sunday updates after Thursday's first game.
- Unknown air yards remain missing. Current/future-week plays do not enter pregame features. Spikes, sacks, duplicate plays and kneels are excluded where appropriate.
- Forecast components separate structural opportunity, volume changes, rate changes and final output. Targets and carries reconcile to team pools, including quarterback rushing changes.
- Pre-week process context includes expected completions/receiving yards, explosive dependence, scoring-area opportunity, current-stint workload trend/variability, and man/zone/blitz/play-action responses with small-sample shrinkage. Missing fields remain missing.
- Participation context uses prior-season data because of provider release timing. Current-season charting is used only where released. Aggregate tendencies do not reveal the next defensive call or exact route assignment.
- Regression/progression labels compare the forecast with the last three recorded appearances. They are candidates affected by role and matchup, not causal proof of luck. Provider expected-outcome models are estimates too.
- Every stat has a predictability label, median and central 80% outcome range. Labels reflect relative dispersion, role variability, sample support and unresolved availability; they are not wager win probabilities.
- Count distributions retain their upper tail; yardage distributions permit negative outcomes. Anytime TD uses a Bernoulli outcome. Combined-stat markets retain dependence bounds instead of assuming independence.
- Every matched posted line remains visible beside its forecast. Line-only sources do not acquire invented prices. Player-prop lines never determine these forecasts.

## Research protocol

Replayed 14,242 projected player-weeks: 5,915 in 2024, 5,459 in 2025 and 2,868 in 2026 Weeks 2–10. The 2024 halves choose regularization; 2025 accepts or rejects 21 predeclared correction families. Acceptance requires at least 100 observations, at least 0.25% MSE improvement and neither half worsening by more than 1%. Accepted coefficients are refitted through 2025. No 2026 outcome chooses features, coefficients, acceptance thresholds or uncertainty parameters.

Only QB passing yards, RB targets and TE anytime TD passed selection (2025 MSE improvements 0.354%, 0.377% and 2.171%). Eighteen corrections remain disabled; their process features can still supply context. This evidence does not justify forcing schematic mean adjustments into every player forecast.

This is a retrospective time-forward audit using currently downloaded historical inputs, not a replay of exact historical source vintages or timestamped sportsbook quotes. Point errors include projected players with zero recorded production. Outcome intervals below are conditional on recorded appearance; true DNP/void handling is a separate settlement question.

## Calibration and publication

The forecast version is `nfl-player-projections/3.0.0`. V2.1.0 betting calibration cannot qualify v3 player bets or slips. Quote-specific v3 calibration remains pending. Raw forecasts, available quotes, PrizePicks outcome estimates and complete weekly projection tracking remain available. No historical entry quote, wager or CLV is reconstructed from today's data.

Full builds run daily during September–January, with Sunday kickoff-wave and late Sunday/Monday/Thursday refreshes. Current-season caches expire hourly. Tracking collects twice hourly. Both workflows retain pending runs in one serialized publication queue. GitHub schedules can be delayed; this is not an instantaneous injury wire. Open dashboards check for a newer full generation every five minutes and on focus.

Fitted parameters ship with the package. Assembly rejects missing parameters, mismatched forecast versions or a training cutoff overlapping the forecast season. The board exposes lineage, fingerprint, data cutoff, process samples, assumptions and per-stat components.

## Verification and remaining limits

316 tests plus 6 subtests and Ruff pass. Tests cover complete-feed injuries, official Out precedence, backfield redistribution with conserved pools, time cutoffs, player transfers, missing values, count tails, negative yardage, calibration isolation, artifact lineage and unfabricated quotes. Release smoke, coverage and consistency checks run before publication. Wheel inspection confirms the fitted parameters and revision engine are packaged. Local research rows and QA ledgers are not published.

`scripts/audit_player_props.py` produces structural/process replays. `scripts/fit_player_intelligence.py` applies selection and generates aggregate results. Research uses NumPy/SciPy; production uses standard-library Python.

Historical improvements are modest and some forecasts/ranges worsen. Predictability describes estimated stability, not certainty. The uncertainty model does not estimate probability of being active. Complete sources can lag official announcements; the partial fallback remains incomplete and visibly degraded. Live weather, exact routes, protection calls and substitutions are not known inputs. Further forward validation is required before treating these outputs as qualified betting probabilities.

References: [nflverse data schedule](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html), [play-by-play dictionary](https://nflreadr.nflverse.com/articles/dictionary_pbp.html), [GitHub pending workflow queues](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency).

Exact aggregate metrics, selection decisions and uncertainty results are in [the accompanying JSON](NFL_PLAYER_PROP_AUDIT_2026-11-19.json). The following tables compare the published structure with the revised model; positive RMSE improvement is better, and lower interval scores are better.

## 2026 point forecasts

| Position / stat | N | V2 RMSE | V3 RMSE | Improvement |
| --- | ---: | ---: | ---: | ---: |
| QB:pass_attempts | 268 | 9.386 | 9.386 | +0.00% |
| QB:rush_attempts | 268 | 2.296 | 2.296 | +0.00% |
| QB:rushing_yards | 268 | 16.209 | 16.209 | +0.00% |
| QB:passing_yards | 268 | 77.463 | 77.152 | +0.40% |
| QB:completions | 268 | 6.393 | 6.393 | +0.00% |
| QB:passing_tds | 268 | 1.112 | 1.112 | +0.00% |
| QB:interceptions | 268 | 0.809 | 0.809 | +0.00% |
| RB:targets | 733 | 1.733 | 1.723 | +0.59% |
| RB:carries | 733 | 4.511 | 4.515 | -0.10% |
| RB:rushing_yards | 733 | 26.434 | 26.493 | -0.22% |
| RB:receiving_yards | 733 | 15.803 | 15.776 | +0.17% |
| RB:receptions | 733 | 1.508 | 1.500 | +0.48% |
| RB:anytime_td_probability | 733 | 0.390 | 0.390 | +0.03% |
| WR:targets | 1065 | 2.722 | 2.734 | -0.47% |
| WR:receiving_yards | 1065 | 30.315 | 30.390 | -0.25% |
| WR:receptions | 1065 | 1.913 | 1.920 | -0.36% |
| WR:anytime_td_probability | 1065 | 0.376 | 0.376 | -0.08% |
| TE:targets | 536 | 2.195 | 2.182 | +0.57% |
| TE:receiving_yards | 536 | 22.607 | 22.572 | +0.15% |
| TE:receptions | 536 | 1.817 | 1.813 | +0.22% |
| TE:anytime_td_probability | 536 | 0.370 | 0.369 | +0.32% |

For binary TD outcomes MSE is the Brier score; this table reports its square root.

## 2026 outcome intervals

| Position / stat | N | V2 coverage | V3 coverage | V2 interval score | V3 interval score |
| --- | ---: | ---: | ---: | ---: | ---: |
| QB:pass_attempts | 266 | 82.3% | 87.2% | 32.088 | 32.654 |
| QB:completions | 266 | 76.3% | 86.8% | 22.388 | 22.109 |
| QB:passing_yards | 266 | 77.1% | 79.3% | 269.654 | 265.158 |
| QB:passing_tds | 266 | 96.2% | 96.2% | 3.508 | 3.508 |
| QB:interceptions | 266 | 97.7% | 97.7% | 2.293 | 2.301 |
| QB:rush_attempts | 266 | 82.3% | 86.8% | 7.576 | 7.786 |
| QB:rushing_yards | 266 | 86.1% | 84.6% | 50.305 | 47.994 |
| RB:rushing_yards | 684 | 82.3% | 81.6% | 88.772 | 88.437 |
| RB:carries | 684 | 81.7% | 83.9% | 15.345 | 16.434 |
| RB:targets | 684 | 85.8% | 90.8% | 5.642 | 5.602 |
| RB:receptions | 684 | 88.0% | 89.8% | 4.889 | 4.797 |
| RB:receiving_yards | 684 | 87.3% | 78.8% | 47.483 | 48.454 |
| WR:targets | 996 | 82.2% | 85.8% | 9.198 | 9.452 |
| WR:receptions | 996 | 88.2% | 89.7% | 6.336 | 6.311 |
| WR:receiving_yards | 996 | 81.7% | 79.9% | 99.371 | 99.116 |
| TE:targets | 473 | 86.9% | 91.8% | 7.412 | 7.247 |
| TE:receptions | 473 | 90.9% | 91.1% | 5.901 | 5.924 |
| TE:receiving_yards | 473 | 86.0% | 76.3% | 77.051 | 75.812 |
| K:fg_attempts | 266 | — | 91.7% | — | 4.617 |
| K:fg_made | 266 | — | 93.2% | — | 4.139 |
| K:pat_made | 266 | — | 95.5% | — | 4.308 |
| K:kicking_points | 266 | — | 82.7% | — | 12.209 |

Central 80% coverage targets approximately 80%. QB passing-yard coverage improves from 77.1% to 79.3%; TE receiving yards under-cover at 76.3%. QB attempts and RB carries have worse interval scores. These weaknesses are reported rather than tuned away on the held-out outcomes. Legacy kicker intervals were unavailable for this comparison.

Parameter SHA-256: `0e504c9265464c37dc4d8ea1d03bb5f534cec8d398076e385454aa5aca7d1792`.
