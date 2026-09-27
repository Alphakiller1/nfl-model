from nflmodel import availability, forecast, totals


def _qb(team, pid, week, attempts, season=2026):
    return {"season": season, "week": week, "season_type": "REG", "position": "QB",
            "team": team, "player_id": pid, "player_display_name": pid, "attempts": attempts}


ROWS = [
    _qb("KC", "starter", 1, 35), _qb("KC", "starter", 2, 30), _qb("KC", "backup", 2, 4),
    _qb("BUF", "allen", 1, 33), _qb("BUF", "allen", 2, 31),
    _qb("BUF", "late", 3, 40),  # week 3 is the week being forecast: must not count
]
ROSTER = [
    {"team": "KC", "gsis_id": "starter", "status": "ACT"},
    {"team": "BUF", "gsis_id": "allen", "status": "ACT"},
]


def test_usual_starter_ignores_the_forecast_week():
    starters = availability.usual_starters(ROWS, 2026, 3)
    assert starters["KC"][0] == "starter"
    assert starters["BUF"][0] == "allen"


def test_week_one_falls_back_to_prior_season():
    rows = [_qb("KC", "old", 17, 30, season=2025)]
    assert availability.usual_starters(rows, 2026, 1)["KC"][0] == "old"


def test_out_and_doubtful_flag_questionable_does_not():
    starters = availability.usual_starters(ROWS, 2026, 3)
    for tag, expected in (("Out", False), ("Doubtful", False), ("Questionable", True)):
        injuries = [{"team": "KC", "gsis_id": "starter", "report_status": tag}]
        status = availability.quarterback_status(starters, injuries, ROSTER)
        assert status["KC"].available is expected, tag
        assert status["BUF"].available


def test_reserve_list_and_departure_flag_but_gameday_inactive_does_not():
    starters = availability.usual_starters(ROWS, 2026, 3)
    for roster_row, expected in (
        ({"team": "KC", "gsis_id": "starter", "status": "RES"}, False),
        ({"team": "LV", "gsis_id": "starter", "status": "ACT"}, False),
        ({"team": "KC", "gsis_id": "starter", "status": "INA"}, True),
    ):
        other = {"team": "KC", "gsis_id": "someone", "status": "ACT"}
        status = availability.quarterback_status(starters, [], [roster_row, other, ROSTER[1]])
        assert status["KC"].available is expected, roster_row


def test_adjustment_sign_is_home_margin():
    starters = availability.usual_starters(ROWS, 2026, 3)
    injuries = [{"team": "KC", "gsis_id": "starter", "report_status": "Out"}]
    status = availability.quarterback_status(starters, injuries, ROSTER)
    assert availability.margin_adjustment(status, "KC", "BUF", points=4.0) == -4.0
    assert availability.margin_adjustment(status, "BUF", "KC", points=4.0) == 4.0
    assert availability.margin_adjustment(status, "BUF", "NE", points=4.0) == 0.0


def test_coefficient_is_positive_and_bounded():
    # A QB-out effect with the wrong sign, or one bigger than a touchdown, means
    # the fit broke rather than football changed.
    assert 0.0 < availability.QB_OUT_POINTS < 7.0


def test_adjustment_moves_model_margin_not_the_total():
    base = totals.project(None, None, rating_margin=3.0)
    moved = totals.project(None, None, rating_margin=3.0, adjustment=-4.0)
    assert moved.margin == base.margin - 4.0
    assert moved.total == base.total

    p = forecast.project_game(home="KC", away="BUF", team_ratings={"KC": 2.0, "BUF": 1.0},
                              market_margin=1.5, availability_margin=-4.0,
                              qb_out=("starter (injury report: Out)",))
    q = forecast.project_game(home="KC", away="BUF", team_ratings={"KC": 2.0, "BUF": 1.0},
                              market_margin=1.5)
    assert p.model_margin == q.model_margin - 4.0
    assert p.margin == q.margin == 1.5  # published margin stays market-anchored at lambda 0
    assert p.qb_out and p.availability_margin == -4.0
