import gzip

from nflmodel import scheme
from nflmodel.sources import nflverse


def _scheme_rows(plays=80):
    pbp = []
    participation = []
    charting = []
    for offense, defense in (("NE", "SEA"), ("SEA", "NE")):
        for index in range(plays):
            play_id = str(index + (0 if offense == "NE" else 1000))
            is_pass = index < 60
            pbp.append({
                "season": "2025", "week": str(index // 5 + 1),
                "season_type": "REG", "game_id": "2025_TEST",
                "play_id": play_id, "posteam": offense, "defteam": defense,
                "play_type": "pass" if is_pass else "run", "pass": str(int(is_pass)),
                "rush": str(int(not is_pass)), "down": "1", "qtr": "2", "wp": ".5",
                "score_differential": "0", "shotgun": "1", "no_huddle": "0",
                "epa": ".2" if is_pass else ".05", "success": "1",
                "run_location": "left", "run_gap": "guard",
                "receiver_player_id": "receiver" if is_pass else "",
            })
            participation.append({
                "nflverse_game_id": "2025_TEST", "play_id": play_id,
                "offense_formation": "SHOTGUN",
                "offense_personnel": "1 C, 2 G, 1 QB, 1 RB, 2 T, 1 TE, 3 WR",
                "defense_personnel": "3 CB, 2 DE, 2 DT, 1 FS, 1 MLB, 1 OLB, 1 SS",
                "defenders_in_box": "6", "was_pressure": "1" if index % 4 == 0 else "0",
                "defense_man_zone_type": "ZONE_COVERAGE" if is_pass else "",
                "defense_coverage_type": "COVER_3" if is_pass else "",
            })
            charting.append({
                "nflverse_game_id": "2025_TEST", "nflverse_play_id": play_id,
                "is_motion": "1", "is_play_action": "1" if is_pass else "0",
                "is_screen_pass": "0", "is_rpo": "0", "n_blitzers": "1",
            })
    return pbp, participation, charting


def _build(extra_pbp=None):
    pbp, participation, charting = _scheme_rows()
    pbp.extend(extra_pbp or [])
    games = [{"home_team": "NE", "away_team": "SEA"}]
    schedule = [
        {"season": 2025, "week": 18, "home_team": "NE", "away_team": "SEA",
         "home_coach": "NE Coach", "away_coach": "SEA Coach"},
        {"season": 2026, "week": 1, "home_team": "NE", "away_team": "SEA",
         "home_coach": "NE Coach", "away_coach": "SEA Coach"},
    ]
    return scheme.build(
        season=2026, week=1, games=games, schedule=schedule, pbp_rows=pbp,
        participation_rows=participation, charting_rows=charting,
        player_positions={"receiver": "WR"},
    )


def test_profiles_track_observed_tendencies_and_label_the_data_boundary():
    result = _build()
    profile = result.profiles["NE"]
    assert profile.offense_plays == 80
    assert profile.offense["personnel_11_rate"] > 0.80
    assert profile.offense["formation_shotgun_rate"] > 0.80
    assert profile.offense["run_gap_guard_rate"] > 0.80
    assert profile.defense["zone_rate"] > 0.80
    assert profile.defense["personnel_nickel_rate"] > 0.80
    assert profile.offense["pass_epa_motion"] > 0
    assert profile.defense["pass_epa_play_action"] > 0
    assert result.status["blocking_scheme"].startswith("unavailable")
    assert "charted run point (guard/tackle/end)" in result.status["proxies"]
    assert result.status["participation_source_seasons"] == [2025]
    assert result.status["charting_source_seasons"] == [2025]


def test_forecast_week_plays_are_excluded_by_construction():
    future = [{
        "season": "2026", "week": "1", "season_type": "REG",
        "game_id": "FUTURE", "play_id": "1", "posteam": "NE", "defteam": "SEA",
        "play_type": "pass", "pass": "1", "rush": "0", "down": "1", "qtr": "1",
        "wp": ".5", "score_differential": "0", "epa": "99", "success": "1",
    }]
    result = _build(extra_pbp=future)
    assert result.profiles["NE"].offense_plays == 80
    assert result.status["source_seasons"] == [2025]


def test_matchup_adjustments_are_bounded_and_preserve_position_level_response():
    matchup = _build().matchups[("NE", "SEA")]
    assert matchup.expected_zone_rate > matchup.expected_man_rate
    assert -2.0 <= matchup.pass_attempt_delta <= 2.0
    assert -0.35 <= matchup.pass_efficiency_delta <= 0.35
    assert set(matchup.target_multipliers) == {"RB", "WR", "TE"}
    assert all(0.88 <= value <= 1.12 for value in matchup.target_multipliers.values())


def test_feed_taxonomies_do_not_collapse_two_man_or_defensive_back_counts():
    assert scheme._coverage("2_MAN") == "cover_2_man"
    personnel = "3 CB, 2 DE, 2 DT, 1 FS, 1 MLB, 1 OLB, 1 SS"
    assert scheme._def_personnel(personnel) == "nickel"
    assert scheme._personnel("1 FB, 1 RB, 2 TE, 1 WR") == "22"


def test_large_source_parser_keeps_only_selected_columns():
    raw = gzip.compress(b"a,b,c\n1,2,3\n")
    assert nflverse._parse_selected(raw, ("a", "c", "missing"), compressed=True) == [
        {"a": "1", "c": "3", "missing": ""}
    ]


def test_player_position_index_uses_latest_position():
    rows = [
        {"player_id": "p", "position": "WR", "season": "2024", "week": "18"},
        {"player_id": "p", "position": "TE", "season": "2025", "week": "1"},
    ]
    assert scheme.position_index(rows) == {"p": "TE"}
