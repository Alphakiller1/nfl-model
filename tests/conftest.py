"""Shared fixtures.

Every test here is offline. `season.assemble` downloads ten seasons from
nflverse, which is right for a build and wrong for a test suite: it would make
CI depend on GitHub being up and on the 2026 schedule not changing under it, and
a test that fails for either reason teaches nobody anything.

So the fixtures below synthesise a `Slate` directly. They are deliberately small
and deliberately asymmetric -- a league of identical teams would pass a sign
error in `defense_index` without noticing.
"""

from __future__ import annotations

import pytest

from nflmodel import authority as auth
from nflmodel import forecast, matrix, season, teams

# A four-team league with a clear ordering, so a ranking test can assert an
# expected order rather than merely that the code ran.
LEAGUE = {
    "KC": 6.0, "BUF": 3.0, "NYJ": -2.0, "CAR": -7.0,
}


def make_form(*, epa=0.0, first_down=0.30, explosive=0.06, sack=0.065, turnover=0.021,
              allowed_epa=None, allowed_first_down=None, allowed_explosive=None,
              allowed_sack=None, allowed_turnover=None, plays=62.0) -> matrix.TeamForm:
    """A `TeamForm` with league-average defaults, so a test varies one thing."""
    return matrix.TeamForm(
        off_epa=epa, off_first_down=first_down, off_explosive=explosive,
        off_sack=sack, off_turnover=turnover,
        def_epa=matrix.LEAGUE_MEAN_FORM["epa"] if allowed_epa is None else allowed_epa,
        def_first_down=(matrix.LEAGUE_MEAN_FORM["first_down"]
                        if allowed_first_down is None else allowed_first_down),
        def_explosive=(matrix.LEAGUE_MEAN_FORM["explosive"]
                       if allowed_explosive is None else allowed_explosive),
        def_sack=matrix.LEAGUE_MEAN_FORM["sack"] if allowed_sack is None else allowed_sack,
        def_turnover=(matrix.LEAGUE_MEAN_FORM["turnover"]
                      if allowed_turnover is None else allowed_turnover),
        plays=plays,
    )


def average_form() -> matrix.TeamForm:
    """Exactly league average on every rate: both indices must read 0.0."""
    mean = matrix.LEAGUE_MEAN_FORM
    return matrix.TeamForm(
        off_epa=mean["epa"], off_first_down=mean["first_down"],
        off_explosive=mean["explosive"], off_sack=mean["sack"],
        off_turnover=mean["turnover"],
        def_epa=mean["epa"], def_first_down=mean["first_down"],
        def_explosive=mean["explosive"], def_sack=mean["sack"],
        def_turnover=mean["turnover"], plays=62.0,
    )


def schedule_rows(season_year: int = 2026) -> list[dict]:
    """A double round-robin over the four teams, plus market prices."""
    rows = []
    order = list(LEAGUE)
    week = 1
    for home in order:
        for away in order:
            if home == away:
                continue
            rows.append({
                "game_id": f"{season_year}_{week:02d}_{away}_{home}",
                "season": season_year, "week": week, "game_type": "REG",
                "gameday": f"{season_year}-09-{10 + week:02d}", "gametime": "13:00",
                "weekday": "Sunday",
                "home_team": home, "away_team": away,
                "home_score": None, "away_score": None,
                "location": "Home",
                "spread_line": round(LEAGUE[home] - LEAGUE[away] + 1.2, 1),
                "total_line": 45.5,
                "home_moneyline": -150.0, "away_moneyline": 130.0,
                "div_game": False, "roof": "outdoors", "surface": "grass",
                "stadium": "Test Field",
                "home_qb_name": "", "away_qb_name": "",
                "home_coach": f"{home} Coach", "away_coach": f"{away} Coach",
                "home_rest": 7.0, "away_rest": 7.0,
            })
            week += 1
    return rows


@pytest.fixture
def slate() -> season.Slate:
    """A fully assembled `Slate` with no network access."""
    rows = schedule_rows()
    table = dict(LEAGUE)
    for abbr in teams.all_abbrs():
        table.setdefault(abbr, 0.0)
    forms = {
        "KC": make_form(epa=0.14, first_down=0.33, allowed_epa=-0.09),
        "BUF": make_form(epa=0.08, first_down=0.31, allowed_epa=-0.03),
        "NYJ": make_form(epa=-0.05, first_down=0.27, allowed_epa=0.02),
        "CAR": make_form(epa=-0.12, first_down=0.25, allowed_epa=0.08),
    }
    week_games = [row for row in rows if row["week"] <= 2]
    authority = auth.current()
    projections = [
        forecast.project_game(
            home=row["home_team"], away=row["away_team"], team_ratings=table,
            home_form=forms.get(row["home_team"]), away_form=forms.get(row["away_team"]),
            market_margin=row["spread_line"], market_total=row["total_line"],
            home_moneyline=row["home_moneyline"], away_moneyline=row["away_moneyline"],
            season=2026, week=row["week"], kickoff=season.kickoff_label(row),
            authority=authority,
        )
        for row in week_games
    ]
    return season.Slate(
        season=2026, week=1, table=table, forms=forms, games=week_games,
        schedule=rows, games_played={}, authority=authority, projections=projections,
    )
