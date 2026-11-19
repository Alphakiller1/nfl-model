# Projection and wager tracking

Every production build runs `scripts/track_projections.py` against the complete
model board, including every player projection. The archive retains each
observed model vintage and its original timestamp, model version and raw
projection. It has no top-N or season-retention truncation. Projections observed
after kickoff remain archived but cannot create a wager or enter pregame
accuracy statistics. Existing legacy prediction ledgers remain intact.

`tracking/` on the model dashboard contains the weekly record, featured cohort,
CSV of every selection, and a compressed archive of every projection, quote
history, result and source receipt. NFL and CFB use the model's official season
and football week. NBA and CBB use the game's Monday–Sunday Eastern week.

The featured cohort locks up to 20 selections before kickoff, ranked within
each incoming batch by relative model disagreement; later batches fill empty
slots. It never withdraws a losing selection or replaces it after a result.
Every other selection remains in the uncapped all-selection record. Tracking
does not change the model's authority, calibrate a new betting edge, or place
real wagers. A ranking is not a claim of a demonstrated advantage.

The first available DraftKings quote locks the selected side, line and actual
price. Spreads and totals follow the model's side of the quoted line;
moneylines use the model's own probability when available; props follow the
model mean against an observed named-book prop line. Missing prices are never
changed to a nominal -110. A result without a price contributes to W–L–P, but
is excluded from units and ROI. A wager risks one paper unit at its entry
price. A verified DNP prop or canceled game is void; missing statistics remain
pending. Pushes return the stake.

Positive CLV means the entry beat the close for the selected side. Spread and
under CLV equal entry line minus closing line; over CLV reverses that sign.
Moneyline CLV is closing minus entry implied probability in percentage points,
and is kept separate from line-market point CLV. A provider's explicit pregame
`close` is preferred. An observed prestart quote is usable only within 60
minutes of kickoff and is labeled `last_observed_prestart`, with its age. A
Tuesday quote is never silently reported as Sunday's close. In-game current
quotes are excluded. Missing close stays `missing_close`, with CLV coverage
reported by market.

The independent `Collect projection results and CLV` workflow runs twice an
hour in season. It restores the full latest model build, recovers the latest
published tracking manifest, collects free ESPN/DraftKings lines and final
boxes, and republishes the site. It does not retrain the model or spend paid
Odds API credits. Full build artifacts last 31 days so a weekly football build
remains available between updates. A failed archive recovery or hash mismatch
stops publication instead of restarting the history. Source/network gaps are
reported explicitly. GitHub's scheduler can be delayed; the close-quality and
last-updated fields expose the actual coverage.

The collector and its contract tests are identical in the four model repos.
Changes must be applied and tested together. Run:

```
python -m unittest discover -s tests -p test_projection_tracking.py -v
```

The weekly archive also retains every complete published board, including season outlooks, futures probabilities and new forecast fields without a sportsbook quote. These remain in the raw record even when they cannot produce a priced wager or CLV observation.
