import math
import statistics

import pytest

from nflmodel import volatility as v


def _game(week, home, away, model, actual, model_total=50.0, actual_total=50.0, season=2026):
    return v.GradedGame(season=season, week=week, home=home, away=away,
                        model_margin=model, actual_margin=actual,
                        model_total=model_total, actual_total=actual_total)


def test_losses_are_scored_from_each_side():
    rows = v.team_games([_game(1, "A", "B", 7.0, -3.0, 50.0, 60.0)], margin_sd=14.0)
    home, away = rows
    assert home.team == "A" and away.team == "B"
    assert home.losses["spread"] == away.losses["spread"] == pytest.approx(10.0)
    # A total that runs 10 over the projection hurts the under, not the over.
    assert home.losses["under"] == pytest.approx(10.0)
    assert home.losses["over"] == 0.0
    # The model favoured A and A lost: an upset from both sides.
    assert home.win_probability > 0.5 > away.win_probability
    assert home.won == 0.0 and away.won == 1.0


def test_moneyline_surprise_has_zero_expectation_when_calibrated():
    # A team at p=0.7 that wins 7 of 10 has no excess surprise.
    p = 0.7
    margin = 14.0 * math.sqrt(2.0) * _erfinv(2 * p - 1)
    games = ([_game(w, "A", "B", margin, 3.0) for w in range(7)]
             + [_game(w, "A", "B", margin, -3.0) for w in range(7, 10)])
    surprise = [r.losses["moneyline"] for r in v.team_games(games, margin_sd=14.0)
                if r.team == "A"]
    assert sum(surprise) / len(surprise) == pytest.approx(0.0, abs=1e-9)


def _erfinv(y):
    lo, hi = -5.0, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if math.erf(mid) < y else (lo, mid)
    return (lo + hi) / 2


def _params(trait=True, k=10.0, decay=0.5, prior=None):
    market = {"trait": trait, "k": k if trait else None, "decay": decay if trait else 0.0,
              "held_out_skill": 0.03 if trait else -0.01}
    return {"markets": {m: dict(market) for m in v.MARKETS}, "seasons": [2021, 2025],
            "prior_season": prior}


def test_predictive_market_shrinks_toward_the_pool_and_ranks_steady_first():
    games = [_game(w, "Steady", "Wild", 0.0, 2.0 if w % 2 else -2.0) for w in range(1, 6)]
    games += [_game(w, "Calm", "Chaos", 0.0, 1.0) for w in range(1, 6)]
    out = v.rank(games, season=2026, params=_params(k=5.0), margin_sd=14.0)
    by = {t["team"]: t["markets"]["spread"] for t in out["teams"]}
    # Every team here missed by the same amount as its opponent, so check the
    # mechanics: reliability is n / (n + k) and expected sits between pool and observed.
    entry = by["Steady"]
    assert entry["reliability"] == pytest.approx(5 / 10, abs=1e-3)
    pool = out["markets"]["spread"]["pool"]
    assert min(pool, entry["observed"]) <= entry["expected"] <= max(pool, entry["observed"])
    ranks = sorted((e["rank"], t) for t, e in by.items())
    assert {t for _, t in ranks[:2]} == {"Calm", "Chaos"}


def test_noise_market_ranks_on_observed_and_is_labelled_descriptive():
    games = [_game(w, "A", "B", 0.0, 1.0) for w in range(1, 4)]
    games += [_game(w, "C", "D", 0.0, 20.0) for w in range(1, 4)]
    out = v.rank(games, season=2026, params=_params(trait=False), margin_sd=14.0)
    assert out["markets"]["spread"]["basis"] == "descriptive"
    by = {t["team"]: t["markets"]["spread"] for t in out["teams"]}
    # No shrinkage: expected is the pool for everyone, so the order must come
    # from what happened, and nobody is graded.
    assert by["A"]["expected"] == by["C"]["expected"]
    assert by["A"]["rank"] < by["C"]["rank"]
    assert all(e["grade"] is None for e in by.values())


def test_stored_prior_season_carries_when_the_ledger_has_none():
    prior = {"season": 2025, "teams": {"A": {m: [10, 100.0] for m in v.MARKETS}}}
    games = [_game(1, "A", "B", 0.0, 0.0)]
    out = v.rank(games, season=2026, params=_params(prior=prior), margin_sd=14.0)
    a = next(t for t in out["teams"] if t["team"] == "A")
    assert a["prior_games"] == 10
    # 1 current game + 0.5 * 10 prior games of effective sample.
    assert a["markets"]["spread"]["reliability"] == pytest.approx(6 / 16, abs=1e-3)
    # A stale prior (two seasons back) is ignored.
    stale = dict(prior, season=2024)
    out = v.rank(games, season=2026, params=_params(prior=stale), margin_sd=14.0)
    assert next(t for t in out["teams"] if t["team"] == "A")["prior_games"] == 0


def test_teams_filter_limits_ranking_not_the_pool():
    games = [_game(1, "A", "FCS", 0.0, 30.0), _game(1, "B", "C", 0.0, 0.0)]
    out = v.rank(games, season=2026, params=_params(trait=False), margin_sd=14.0,
                 teams={"A", "B", "C"})
    assert {t["team"] for t in out["teams"]} == {"A", "B", "C"}
    assert out["markets"]["spread"]["pool"] == pytest.approx(15.0)


def test_ledger_keeps_the_last_pre_kickoff_snapshot_per_game():
    base = {"season": 2026, "week": 3, "home": "A", "away": "B", "status": "graded",
            "kickoff": "2026-09-19T19:30:00Z", "actual_margin": 7.0, "actual_total": 51.0}
    snaps = [
        {**base, "recorded_at": "2026-09-17T12:00:00Z", "model_margin": 1.0, "model_total": 40.0},
        {**base, "recorded_at": "2026-09-19T12:00:00+00:00", "model_margin": 3.0,
         "model_total": 45.0},
        {**base, "recorded_at": "2026-09-19T21:00:00Z", "model_margin": 9.0, "model_total": 60.0},
        {**base, "week": 4, "status": "pending", "recorded_at": "2026-09-24T12:00:00Z",
         "model_margin": 2.0},
    ]
    games = v.graded_from_ledger(snaps)
    assert len(games) == 1
    assert games[0].model_margin == 3.0 and games[0].model_total == 45.0


def test_fit_finds_no_trait_in_pure_noise():
    import random
    rnd = random.Random(7)
    teams = [f"T{i}" for i in range(30)]
    games = []
    for season in (2021, 2022, 2023, 2024):
        for week in range(1, 11):
            order = teams[:]
            rnd.shuffle(order)
            for home, away in zip(order[::2], order[1::2]):
                games.append(_game(week, home, away, 0.0, rnd.gauss(0, 14), 50.0,
                                   50.0 + rnd.gauss(0, 14), season=season))
    fit = v.fit(games, margin_sd=14.0, source="test")
    assert not fit["markets"]["spread"]["trait"]
    assert fit["prior_season"]["season"] == 2024


def test_render_section_handles_empty_and_full_payloads():
    assert "No graded games" in v.render_section(None)
    games = [_game(w, "A", "B", 3.0, 7.0, 50.0, 41.0) for w in range(1, 4)]
    payload = v.rank(games, season=2026, params=_params(), margin_sd=14.0)
    out = v.render_section(payload, logo=lambda team: None)
    assert out.count('class="vol-card"') == 4
    assert "predictive" in out and 'id="volatility"' in out


def test_shipped_fit_is_loadable():
    params = v.params()
    assert set(params["markets"]) == set(v.MARKETS)
    assert params["prior_season"]["teams"]


# -- noise cancellation, consistency prior, and the two-scheme bar ----------------
def _process_game(week, home, away, rate_h, rate_a, luck=0.0, season=2026):
    """A game whose margin is exactly 100 * (rate gap) plus `luck`."""
    margin = 100.0 * (rate_h - rate_a) + luck
    return v.GradedGame(season=season, week=week, home=home, away=away,
                        model_margin=0.0, actual_margin=margin,
                        model_total=50.0, actual_total=50.0 + 50.0 * (rate_h + rate_a - 0.8),
                        home_process={"sr": rate_h}, away_process={"sr": rate_a})


def test_outcome_model_recovers_the_process_relationship():
    import random
    rnd = random.Random(3)
    games = [_process_game(w, "A", "B", rnd.uniform(0.3, 0.5), rnd.uniform(0.3, 0.5),
                           luck=rnd.gauss(0, 3)) for w in range(60)]
    model = v.fit_outcome_model(games, ("sr",))
    assert model.margin_coef[1] == pytest.approx(100.0, rel=0.15)
    assert model.total_coef[1] == pytest.approx(50.0, rel=0.05)
    assert model.luck_sd == pytest.approx(3.0, rel=0.3)
    assert model.margin_r2 > 0.5


def test_process_loss_strips_the_luck():
    model = v.OutcomeModel(("sr",), (0.0, 100.0), (-40.0, 50.0), 5.0, 0.8, 0.8)
    # Process says a 10-point win; the final was a 3-point loss (a turnover swing).
    game = _process_game(1, "A", "B", 0.5, 0.4, luck=-13.0)
    home = v.team_games([game], margin_sd=14.0, outcome=model)[0]
    assert home.losses["spread"] == pytest.approx(3.0)
    assert home.process["spread"] == pytest.approx(10.0)
    assert home.line == {"sr": 0.5} and home.against == {"sr": 0.4}


def test_consistency_is_game_to_game_spread_on_both_sides():
    games = [_process_game(w, "A", "B", r, 0.4) for w, r in enumerate((0.3, 0.5, 0.4))]
    rows = [r for r in v.team_games(games, margin_sd=14.0) if r.team == "A"]
    c = v.consistency(rows, ("sr",))
    assert c["off_sd_sr"] == pytest.approx(statistics.pstdev([0.3, 0.5, 0.4]))
    assert c["def_sd_sr"] == pytest.approx(0.0)
    assert v.consistency(rows[:2], ("sr",))["off_sd_sr"] is None


def test_consistency_prior_moves_a_team_off_the_pool():
    games = []
    for w in range(1, 5):
        games.append(_process_game(w, "Swingy", "X", 0.3 if w % 2 else 0.6, 0.4))
        games.append(_process_game(w, "Steady", "Y", 0.45, 0.4))
    params = {"consistency_stats": ["sr"], "markets": {
        "spread": {"trait": True, "k": None, "decay": 0.0, "alpha": 0.0,
                   "beta": {"off_sd_sr": -1.0, "def_sd_sr": 0.0}}}}
    out = v.rank(games, season=2026, params=params, margin_sd=14.0)
    by = {t["team"]: t["markets"]["spread"] for t in out["teams"]}
    # Negative beta on offensive inconsistency: the swingy team is forecast to
    # miss less, and with k=None its own miss record carries no weight.
    assert by["Swingy"]["consistency_shift"] < 0 < by["Steady"]["consistency_shift"]
    assert by["Swingy"]["rank"] < by["Steady"]["rank"]
    assert by["Swingy"]["reliability"] == 0.0
    html = v.render_section(out)
    assert "gives a team&rsquo;s own miss record no weight" in html


def test_a_trait_must_pass_both_validation_schemes():
    import random
    rnd = random.Random(11)
    teams = [f"T{i}" for i in range(24)]
    # Each team has a persistent total-overshoot tendency: a real trait.
    lean = {t: rnd.uniform(0, 12) for t in teams}
    games = []
    for season in range(2019, 2025):
        for week in range(1, 11):
            order = teams[:]
            rnd.shuffle(order)
            for h, a in zip(order[::2], order[1::2]):
                over = abs(rnd.gauss(0, 4)) + lean[h] + lean[a]
                games.append(_game(week, h, a, 0.0, rnd.gauss(0, 10), 50.0, 50.0 + over,
                                   season=season))
    fit = v.fit(games, margin_sd=14.0, source="test")
    under = fit["markets"]["under"]
    assert under["trait"] and under["variant"] == "plain"
    assert under["variants"]["plain"]["loso"]["passes"]
    assert under["variants"]["plain"]["forward"]["passes"]
    assert not fit["markets"]["spread"]["trait"]
