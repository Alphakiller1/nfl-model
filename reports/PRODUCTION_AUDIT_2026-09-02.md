# NFL production-readiness audit — 2026-09-02

## Outcome

The NFL board now publishes exact DraftKings spreads, totals, and paired
moneylines from a fresh provider snapshot; proves input freshness in a machine-
readable manifest; and starts an immutable pre-kickoff shadow record. The model
remains **RESEARCH_ONLY** because its independent number does not beat the
closing-market benchmark.

The audit covered model specification, matrix signs and reconciliation,
historical validation, favorite/underdog behavior, live price identity, quote
timestamps, schedule matching, cache fallback, deployment failure modes, JSON
contracts, grading, dashboard hierarchy, and responsive source.

## Critical findings and disposition

| Severity | Finding | Disposition |
| --- | --- | --- |
| P0 | The public board had no sportsbook client or repository secret, so it could not publish live book lines. | Added a quota-aware The Odds API client locked to DraftKings and installed the Actions secret without committing it. |
| P0 | The workflow could deploy a board without proving sportsbook identity, freshness, or coverage. | Added `build.json` verification and fail-closed release gates. |
| P1 | The headline backtest was leave-one-season-out, which excludes the scored season but can train an early fold on later seasons. | Added an expanding-season coefficient audit: every fold fits on earlier seasons only. The frozen specification was selected by the legacy LOSO study, so prospective proof remains outstanding. |
| P1 | Model disagreements overwhelmingly selected underdogs. | Measured the effect (72.32%) and tested global, weekly, regime, and component recalibrations. All worsened time-forward error; none was adopted cosmetically. |
| P1 | nflverse stale fallback was silent. | Every source now reports live/cache/bounded-snapshot/error state, age, row count, and error text. |
| P1 | The board mixed model-derived values and market references on one shelf. | DraftKings prices now occupy the first shelf; independent projections and explanatory factors are separate. |
| P2 | No forward NFL record existed. | Added an immutable quote-vintage ledger and deterministic grader, persisted across Actions runs. |
| P2 | HTML and JSON were assembled in separate commands. | One build now emits `index.html`, `board.json`, `build.json`, and `record.json` from the same slate. |

## Predictive audit

Protocol: 1,615 regular-season games, 2020–2025. Each season's point-model coefficients are
fit only on earlier seasons. Pregame features use only games before the week being forecast. The
feature specification and hyperparameters are frozen at their legacy LOSO-selected 2026 values;
this is not a fully nested or locked prospective trial.

| Candidate | Margin MAE | ATS | Underdog-side share |
| --- | ---: | ---: | ---: |
| Rating only | 10.3374 | 766–816–33 | 56.41% |
| Efficiency only | 10.3256 | 804–778–33 | 82.48% |
| **Existing 50/50 blend** | **10.2274** | **784–798–33** | **72.32%** |
| Global affine correction | 10.2399 | 780–802–33 | 70.46% |
| Regime affine correction | 10.2890 | 779–803–33 | 68.30% |
| Week-specific affine correction | 10.3819 | 755–827–33 | 70.90% |
| Regime component refit | 10.3115 | 765–817–33 | 68.30% |
| Closing-market benchmark | **9.7644** | — | — |

The favorite/underdog skew is real, but it is not valid to correct a model until
the correction improves future outcomes. The retained blend was best on MAE.
Its ATS rate is 49.56% with a normal-approximation 95% interval of
[47.09%, 52.02%], still below the 52.38% -110 breakeven at the upper bound.

| Candidate | Total MAE | O/U |
| --- | ---: | ---: |
| Past-only league mean | 10.9702 | — |
| **70% shrunk total** | **10.6598** | **794–805–16** |
| Global affine correction | 10.7988 | 792–807–16 |
| Regime affine correction | 10.8610 | 764–835–16 |
| Closing-market benchmark | **10.2833** | — |

The time-forward residual scales now drive displayed uncertainty: 13.18 points
for margin and 13.48 for total. The point coefficients themselves did not
change; no tested alternative earned lower future-season error.

## Live data and deployment

- Production requests exactly `americanfootball_nfl`, bookmaker
  `draftkings`, markets `h2h,spreads,totals`.
- Team identity is exact after normalization. Ambiguous city-only matches are
  forbidden; historical relocation aliases are explicit.
- Provider and nflverse kickoffs may differ by no more than six hours.
- Live odds are cached for 15 minutes; the free sports endpoint checks quota
  before the paid three-market request.
- Release verification requires every slate game to have a spread, total, and paired moneyline,
  and rejects a sportsbook snapshot older than 20 minutes.
- Scheduled builds run at the Tuesday line-open and noon ET on Thursday,
  Sunday, and Monday during September–January.
- Actions restores and saves source snapshots plus the shadow ledger. A failed
  build cannot replace the last deployed page.

## Live verification

The 2026 Week 1 local production build matched **16/16** scheduled games to
DraftKings. All matched rows included spread, total, and paired moneyline. The
event feed contained 272 NFL events with zero unmatched identities. The
provider reported **94 credits remaining** after the validated snapshot.

The rendered board contained 16 cards and all four JSON/HTML artifacts passed
their structural checks. Browser interaction remains pending because the Codex
browser extension was not connected; no alternate browser surface was silently
substituted.

## Remaining risks

1. The independent model is 0.4630 margin-MAE points and 0.3766 total-MAE points
   behind the closing market. No betting promotion is justified.
2. The public matrix still lacks point-in-time quarterback, injury, and snap-
   displacement inputs. Genesis owns those challengers; raw status or recent
   usage changes may not be treated as persistent without a causal mechanism.
3. Ninety shadow days and 300+ executable observations remain unmet production
   gates.
4. The remaining Odds API quota is adequate for the current refresh, not an
   unlimited season. The quota floor prevents accidental exhaustion; scheduled
   cadence may need adjustment if other sports share the same plan.

## Primary external references checked

- [nflverse-data release inventory and update cadence](https://github.com/nflverse/nflverse-data/blob/main/README.Rmd)
- [nflverse roster, depth-chart, and injury pipeline](https://github.com/nflverse/nflverse-rosters)
- [nflreadr weekly-roster source contract](https://github.com/nflverse/nflreadr/blob/main/R/load_rosters_weekly.R)
- [The Odds API v4 guide](https://the-odds-api.com/liveapi/guides/v4/)
