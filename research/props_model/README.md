# Props model research harness

Reproduces `reports/PROPS_MODEL.md`. These scripts need `pandas`, `pyarrow`
and `scikit-learn`; the package itself stays dependency-free.

## Data (play-by-play study, parts A and B)

Download into this directory:

```
for s in 2021 2022 2023 2024 2025 2026; do
  curl -fsSLO https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_$s.parquet
  curl -fsSLO https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_$s.parquet
done
curl -fsSLO https://github.com/nflverse/nflverse-data/releases/download/players/players.parquet
```

The scripts read `pbp_{season}.parquet` and `snaps_{season}.parquet`, so rename
the files (`play_by_play_` -> `pbp_`, `snap_counts_` -> `snaps_`).

| script | what it answers |
|---|---|
| `build_tables.py` | team-game table (volume, pace, PROE, market lines) |
| `a1_reliability.py` | split-half reliability and shrinkage k per tendency |
| `a2_halflife.py` | half-life, prior weight and offseason carry-over |
| `a3_volume.py` | pre-game pass-attempt / designed-run models vs v1 |
| `a4_script.py` | how spread and total map to plays, dropback rate, volume |
| `a5_lean_volume.py` | the shipped volume model on player-stat inputs only |
| `b0_players.py` | player-game usage table with snaps |
| `b1_shares.py` | half-life and shrinkage for target and carry share |
| `b2_snaps.py`, `b6_snapform.py` | does snap share add; the shipped trend form |
| `b3_redistribution.py` | where an absent regular's volume goes |
| `b4_eff_td.py` | efficiency pseudo-counts; red-zone share for TDs |
| `b5_qb.py` | QB rate pseudo-counts |

## Projection replay and market test (parts C and D)

```
python replay.py 2025 2,3,...,18 replay_v2.json  # point-in-time rebuild of a season
python compare.py replay_v1.json replay_v2.json  # RMSE / MAE / bias by position
python collect_espn_props.py 2026 1,2,3,4        # DraftKings lines from ESPN
python join.py                                   # join lines to the published ledger
python c1_vs_line.py                             # line vs projection, blend weight
python c2_blend_calibration.py                   # bootstrap CIs, P(over) calibration
python c3_matchup_vs_line.py                     # matchup metrics vs (actual - line)
```

`replay.py` honours `NFLMODEL_SRC` (which package to replay) and `PATCH`
(Python run after import, for ablations: `PATCH="player_props.USAGE_PSEUDO_GAMES=3"`).
`join.py` expects the published `ledger.json` in this directory.
