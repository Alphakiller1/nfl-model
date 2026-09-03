"""Point-in-time NFL scheme tendencies and matchup response profiles.

The module treats a tendency as a distribution, never as a permanent team
label. Rates are estimated from plays available before the forecast, shrunk to
the league mean, decayed for recency, and discounted after a head-coach change.

Data provenance is part of the model contract. Formation, personnel, coverage,
pressure, motion, play action and RPO are observed. Run direction and the named
gap are observed proxies. Offensive-line zone/gap/power/man blocking is not in
the nflverse/FTN public files and is therefore explicitly unavailable rather
than reverse-engineered and presented as charted fact.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field

from . import teams
from .sources.nflverse import number

MODEL_VERSION = "nfl-scheme-matrix/1.0.0"
HALF_LIFE_WEEKS = 16.0
RATE_PRIOR_PLAYS = 48.0
RESPONSE_PRIOR_PLAYS = 32.0
TARGET_PRIOR_PLAYS = 24.0

OBSERVED_FEATURES = (
    "offense formation",
    "offense personnel",
    "defense personnel",
    "defenders in box",
    "man/zone",
    "coverage family",
    "pass rushers and pressure",
    "motion",
    "play action",
    "RPO",
    "screen",
    "no huddle",
)
PROXY_FEATURES = (
    "run direction",
    "charted run point (guard/tackle/end)",
    "box count as run-front context",
)
UNAVAILABLE_FEATURES = (
    "offensive-line zone/gap/power/man blocking family",
    "individual blocking assignment",
)


@dataclass(frozen=True)
class TeamSchemeProfile:
    team: str
    source_seasons: tuple[int, ...]
    participation_source_seasons: tuple[int, ...]
    charting_source_seasons: tuple[int, ...]
    offense_plays: int
    defense_plays: int
    offense: dict[str, float]
    defense: dict[str, float]
    confidence: str
    staff_continuity: str
    carryover_weight: float
    coverage_samples: int
    charting_samples: int
    model_version: str = MODEL_VERSION


@dataclass(frozen=True)
class SchemeMatchup:
    team: str
    opponent: str
    expected_man_rate: float
    expected_zone_rate: float
    expected_coverages: dict[str, float]
    expected_motion_rate: float
    expected_play_action_rate: float
    expected_blitz_rate: float
    expected_pressure_rate: float
    neutral_pass_rate: float
    pass_attempt_delta: float
    carry_delta: float
    pass_efficiency_delta: float
    rush_efficiency_delta: float
    target_multipliers: dict[str, float]
    confidence: str
    factors: tuple[str, ...]
    source_seasons: tuple[int, ...]
    model_version: str = MODEL_VERSION


@dataclass(frozen=True)
class BuildResult:
    profiles: dict[str, TeamSchemeProfile]
    matchups: dict[tuple[str, str], SchemeMatchup]
    league: dict[str, float]
    status: dict


@dataclass
class _Accumulator:
    numerators: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    denominators: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    raw_plays: int = 0
    coverage_samples: int = 0
    charting_samples: int = 0

    def rate(self, key: str, event: bool, weight: float) -> None:
        self.denominators[key] += weight
        if event:
            self.numerators[key] += weight

    def mean(self, key: str, value: float | None, weight: float) -> None:
        if value is None:
            return
        self.denominators[key] += weight
        self.numerators[key] += value * weight


def _num(value) -> float | None:
    return number(value)


def _flag(value) -> bool | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y"}:
        return True
    if text in {"false", "f", "no", "n"}:
        return False
    parsed = number(value)
    return None if parsed is None else parsed > 0


def _key(game_id, play_id) -> tuple[str, str]:
    game = str(game_id or "").strip()
    play = str(play_id or "").strip()
    if play.endswith(".0"):
        play = play[:-2]
    return game, play


def _order(season: int, week: int) -> int:
    return season * 24 + week


def _source_season(row: dict, *, game_field: str) -> int | None:
    explicit = _num(row.get("season"))
    if explicit:
        return int(explicit)
    game_id = str(row.get(game_field) or "")
    match = re.match(r"(20\d{2})", game_id)
    return int(match.group(1)) if match else None


def _weight(row_season: int, row_week: int, season: int, week: int) -> float:
    age = max(0, _order(season, week) - _order(row_season, row_week))
    return 0.5 ** (age / HALF_LIFE_WEEKS)


def _personnel(value: str) -> str:
    """Translate '1 RB, 1 TE, 3 WR' into conventional 11/12/21 groupings."""
    text = str(value or "").upper()
    rb = re.search(r"(\d+)\s*RB", text)
    fb = re.search(r"(\d+)\s*FB", text)
    te = re.search(r"(\d+)\s*TE", text)
    if rb and te:
        backs = int(rb.group(1)) + (int(fb.group(1)) if fb else 0)
        code = f"{backs}{te.group(1)}"
        return code if code in {"10", "11", "12", "13", "20", "21", "22"} else "other"
    compact = re.sub(r"[^0-9]", "", text)
    return compact if compact in {"10", "11", "12", "13", "20", "21", "22"} else "other"


def _def_personnel(value: str) -> str:
    text = str(value or "").upper()
    defensive_backs = sum(
        int(match.group(1))
        for match in re.finditer(r"(\d+)\s*(?:CB|DB|FS|SS)\b", text)
    )
    if not defensive_backs:
        return "other"
    if defensive_backs >= 6:
        return "dime"
    if defensive_backs == 5:
        return "nickel"
    return "base"


def _formation(value: str) -> str:
    text = re.sub(r"[^A-Z0-9]+", "_", str(value or "").upper()).strip("_")
    if "PISTOL" in text:
        return "pistol"
    if "SHOTGUN" in text:
        return "shotgun"
    if "UNDER_CENTER" in text:
        return "under_center"
    return "other"


def _man_zone(value: str) -> str | None:
    text = str(value or "").upper()
    if "MAN" in text:
        return "man"
    if "ZONE" in text:
        return "zone"
    return None


def _coverage(value: str) -> str | None:
    text = str(value or "").upper().replace("ZERO", "0")
    if "2_MAN" in text or "2 MAN" in text or "TWO MAN" in text:
        return "cover_2_man"
    match = re.search(r"(?:COVER|COV)[ _-]*([0-6])", text)
    if match:
        return f"cover_{match.group(1)}"
    return None


def _run_gap(value: str) -> str | None:
    text = str(value or "").upper()
    for gap in ("GUARD", "TACKLE", "END"):
        if gap in text:
            return gap.lower()
    return None


def _canonical_position(value: str) -> str | None:
    position = str(value or "").upper().strip()
    if position in {"RB", "FB", "HB"}:
        return "RB"
    return position if position in {"WR", "TE"} else None


def position_index(player_rows: list[dict]) -> dict[str, str]:
    """Latest known fantasy position by GSIS player id."""
    ordered: dict[str, tuple[int, str]] = {}
    for row in player_rows:
        player_id = str(row.get("player_id") or "").strip()
        position = _canonical_position(row.get("position") or "")
        if not player_id or position is None:
            continue
        order = _order(int(_num(row.get("season")) or 0), int(_num(row.get("week")) or 0))
        if order >= ordered.get(player_id, (-1, ""))[0]:
            ordered[player_id] = (order, position)
    return {player_id: value[1] for player_id, value in ordered.items()}


def _league_accumulators(accumulators: dict[str, _Accumulator]) -> _Accumulator:
    league = _Accumulator()
    for acc in accumulators.values():
        league.raw_plays += acc.raw_plays
        league.coverage_samples += acc.coverage_samples
        league.charting_samples += acc.charting_samples
        for key, value in acc.numerators.items():
            league.numerators[key] += value
        for key, value in acc.denominators.items():
            league.denominators[key] += value
    return league


def _raw_mean(acc: _Accumulator, key: str, fallback: float = 0.0) -> float:
    denominator = acc.denominators.get(key, 0.0)
    return acc.numerators.get(key, 0.0) / denominator if denominator else fallback


def _shrunk(
    acc: _Accumulator,
    league: _Accumulator,
    key: str,
    *,
    pseudo: float = RATE_PRIOR_PLAYS,
) -> float:
    baseline = _raw_mean(league, key)
    denominator = acc.denominators.get(key, 0.0)
    return (acc.numerators.get(key, 0.0) + pseudo * baseline) / (denominator + pseudo)


def _profile_values(acc: _Accumulator, league: _Accumulator, *, defense: bool) -> dict[str, float]:
    prefix = "def_" if defense else "off_"
    values: dict[str, float] = {}
    names = (
        "neutral_pass", "shotgun", "no_huddle", "motion", "play_action", "rpo",
        "screen", "personnel_11", "personnel_12", "personnel_21", "personnel_13",
        "formation_pistol", "formation_shotgun", "formation_under_center",
        "run_left", "run_middle", "run_right", "run_gap_guard", "run_gap_tackle",
        "run_gap_end", "man", "zone",
        "cover_0", "cover_1", "cover_2", "cover_3", "cover_4", "cover_6",
        "cover_2_man", "blitz", "pressure", "stacked_box", "personnel_base",
        "personnel_nickel", "personnel_dime", "pass_success", "rush_success",
    )
    for name in names:
        key = prefix + name
        values[name + "_rate"] = _shrunk(acc, league, key)
    response_means = (
        "pass_epa", "rush_epa", "pass_epa_man", "pass_epa_zone",
        "pass_epa_motion", "pass_epa_no_motion", "pass_epa_play_action",
        "pass_epa_no_play_action", "pass_epa_blitz", "pass_epa_no_blitz",
        "pass_epa_pressure", "pass_epa_no_pressure", "rush_epa_motion",
        "rush_epa_no_motion", "rush_epa_rpo", "rush_epa_no_rpo",
        "rush_epa_stacked_box", "rush_epa_light_box", "avg_box",
        "pass_epa_cover_0", "pass_epa_cover_1", "pass_epa_cover_2",
        "pass_epa_cover_3", "pass_epa_cover_4", "pass_epa_cover_6",
        "pass_epa_cover_2_man",
    )
    for name in response_means:
        key = prefix + name
        pseudo = RESPONSE_PRIOR_PLAYS if "epa" in name else RATE_PRIOR_PLAYS
        values[name] = _shrunk(acc, league, key, pseudo=pseudo)
    for coverage in ("all", "man", "zone"):
        for position in ("RB", "WR", "TE"):
            key = prefix + f"target_{position.lower()}_{coverage}"
            values[f"target_share_{position.lower()}_{coverage}"] = _shrunk(
                acc, league, key, pseudo=TARGET_PRIOR_PLAYS
            )
    return {key: round(value, 4) for key, value in values.items()}


def _coach_map(schedule: list[dict], season: int) -> dict[str, str]:
    coaches: dict[str, tuple[int, str]] = {}
    for row in schedule:
        if int(row.get("season") or 0) != season:
            continue
        week = int(row.get("week") or 0)
        for side in ("home", "away"):
            team = teams.canonical(row.get(f"{side}_team") or "")
            coach = str(row.get(f"{side}_coach") or "").strip()
            if team and coach and week >= coaches.get(team, (-1, ""))[0]:
                coaches[team] = (week, coach)
    return {team: value[1] for team, value in coaches.items()}


def build(
    *,
    season: int,
    week: int,
    games: list[dict],
    schedule: list[dict],
    pbp_rows: list[dict],
    participation_rows: list[dict],
    charting_rows: list[dict],
    player_positions: dict[str, str] | None = None,
) -> BuildResult:
    """Build profiles and slate matchups using only plays before ``season/week``."""
    player_positions = player_positions or {}
    participation = {
        _key(row.get("nflverse_game_id"), row.get("play_id")): row
        for row in participation_rows
    }
    charting = {
        _key(row.get("nflverse_game_id"), row.get("nflverse_play_id")): row
        for row in charting_rows
    }
    participation_seasons = tuple(sorted({
        value for row in participation_rows
        if (value := _source_season(row, game_field="nflverse_game_id")) is not None
    }))
    charting_seasons = tuple(sorted({
        value for row in charting_rows
        if (value := _source_season(row, game_field="nflverse_game_id")) is not None
    }))
    offense: dict[str, _Accumulator] = defaultdict(_Accumulator)
    defense: dict[str, _Accumulator] = defaultdict(_Accumulator)
    source_seasons: set[int] = set()

    for row in pbp_rows:
        row_season = int(_num(row.get("season")) or 0)
        row_week = int(_num(row.get("week")) or 0)
        if not row_season or not row_week or _order(row_season, row_week) >= _order(season, week):
            continue
        if str(row.get("season_type") or "REG").upper() != "REG":
            continue
        team = teams.canonical(row.get("posteam") or "")
        opponent = teams.canonical(row.get("defteam") or "")
        is_pass = (_flag(row.get("pass")) is True) or row.get("play_type") == "pass"
        is_rush = (_flag(row.get("rush")) is True) or row.get("play_type") == "run"
        if not team or not opponent or not (is_pass or is_rush):
            continue
        source_seasons.add(row_season)
        weight = _weight(row_season, row_week, season, week)
        off = offense[team]
        deff = defense[opponent]
        off.raw_plays += 1
        deff.raw_plays += 1
        context = participation.get(_key(row.get("game_id"), row.get("play_id")), {})
        ftn = charting.get(_key(row.get("game_id"), row.get("play_id")), {})

        down = int(_num(row.get("down")) or 0)
        qtr = int(_num(row.get("qtr")) or 0)
        wp = _num(row.get("wp"))
        score = abs(_num(row.get("score_differential")) or 0.0)
        neutral = down in {1, 2} and qtr <= 3 and score <= 10 and (wp is None or 0.2 <= wp <= 0.8)
        if neutral:
            off.rate("off_neutral_pass", is_pass, weight)
            deff.rate("def_neutral_pass", is_pass, weight)

        for acc, prefix in ((off, "off_"), (deff, "def_")):
            shotgun = _flag(row.get("shotgun"))
            no_huddle = _flag(row.get("no_huddle"))
            if shotgun is not None:
                acc.rate(prefix + "shotgun", shotgun, weight)
            if no_huddle is not None:
                acc.rate(prefix + "no_huddle", no_huddle, weight)
            epa = _num(row.get("epa"))
            success = _flag(row.get("success"))
            if is_pass:
                acc.mean(prefix + "pass_epa", epa, weight)
                if success is not None:
                    acc.rate(prefix + "pass_success", success, weight)
            if is_rush:
                acc.mean(prefix + "rush_epa", epa, weight)
                if success is not None:
                    acc.rate(prefix + "rush_success", success, weight)

        formation_text = str(context.get("offense_formation") or "")
        personnel_text = str(context.get("offense_personnel") or "")
        defense_personnel = str(context.get("defense_personnel") or "")
        if formation_text:
            category = _formation(formation_text)
            off.rate("off_formation_" + category, True, weight)
            deff.rate("def_formation_" + category, True, weight)
            formations = {"pistol", "shotgun", "under_center", "other"}
            for other in formations - {category}:
                off.rate("off_formation_" + other, False, weight)
                deff.rate("def_formation_" + other, False, weight)
        if personnel_text:
            category = _personnel(personnel_text)
            for label in ("11", "12", "21", "13"):
                off.rate("off_personnel_" + label, category == label, weight)
                deff.rate("def_personnel_" + label, category == label, weight)
        if defense_personnel:
            category = _def_personnel(defense_personnel)
            for label in ("base", "nickel", "dime"):
                deff.rate("def_personnel_" + label, category == label, weight)

        box = _num(context.get("defenders_in_box"))
        if box is None:
            box = _num(ftn.get("n_defense_box"))
        for acc, prefix in ((off, "off_"), (deff, "def_")):
            acc.mean(prefix + "avg_box", box, weight)
            if box is not None:
                acc.rate(prefix + "stacked_box", box >= 8, weight)
                if is_rush:
                    box_key = (
                        "rush_epa_stacked_box" if box >= 8 else "rush_epa_light_box"
                    )
                    acc.mean(prefix + box_key, _num(row.get("epa")), weight)

        if is_rush:
            location = str(row.get("run_location") or "").lower()
            if location in {"left", "middle", "right"}:
                for label in ("left", "middle", "right"):
                    off.rate("off_run_" + label, location == label, weight)
                    deff.rate("def_run_" + label, location == label, weight)
            gap = _run_gap(row.get("run_gap") or "")
            if gap:
                for label in ("guard", "tackle", "end"):
                    off.rate("off_run_gap_" + label, gap == label, weight)
                    deff.rate("def_run_gap_" + label, gap == label, weight)

        if ftn:
            off.charting_samples += 1
            deff.charting_samples += 1
            for name, source in (("motion", "is_motion"), ("rpo", "is_rpo")):
                value = _flag(ftn.get(source))
                if value is not None:
                    off.rate("off_" + name, value, weight)
                    deff.rate("def_" + name, value, weight)
                    if name == "motion" or is_rush:
                        play_family = "pass" if is_pass else "rush"
                        suffix = name if value else "no_" + name
                        response_key = f"{play_family}_epa_{suffix}"
                        off.mean("off_" + response_key, _num(row.get("epa")), weight)
                        deff.mean("def_" + response_key, _num(row.get("epa")), weight)
            if is_pass:
                for name, source in (
                    ("play_action", "is_play_action"), ("screen", "is_screen_pass"),
                ):
                    value = _flag(ftn.get(source))
                    if value is not None:
                        off.rate("off_" + name, value, weight)
                        deff.rate("def_" + name, value, weight)
                        if name == "play_action":
                            response_key = (
                                "pass_epa_play_action" if value
                                else "pass_epa_no_play_action"
                            )
                            off.mean("off_" + response_key, _num(row.get("epa")), weight)
                            deff.mean("def_" + response_key, _num(row.get("epa")), weight)
                blitzers = _num(ftn.get("n_blitzers"))
                if blitzers is not None:
                    was_blitz = blitzers > 0
                    off.rate("off_blitz", was_blitz, weight)
                    deff.rate("def_blitz", was_blitz, weight)
                    response_key = "pass_epa_blitz" if was_blitz else "pass_epa_no_blitz"
                    off.mean("off_" + response_key, _num(row.get("epa")), weight)
                    deff.mean("def_" + response_key, _num(row.get("epa")), weight)

        if is_pass:
            pressure = _flag(context.get("was_pressure"))
            if pressure is not None:
                off.rate("off_pressure", pressure, weight)
                deff.rate("def_pressure", pressure, weight)
                pressure_key = (
                    "pass_epa_pressure" if pressure else "pass_epa_no_pressure"
                )
                off.mean("off_" + pressure_key, _num(row.get("epa")), weight)
                deff.mean("def_" + pressure_key, _num(row.get("epa")), weight)
            man_zone = _man_zone(context.get("defense_man_zone_type") or "")
            coverage = _coverage(context.get("defense_coverage_type") or "")
            if man_zone:
                off.coverage_samples += 1
                deff.coverage_samples += 1
                for acc, prefix in ((off, "off_"), (deff, "def_")):
                    acc.rate(prefix + "man", man_zone == "man", weight)
                    acc.rate(prefix + "zone", man_zone == "zone", weight)
                    acc.mean(prefix + "pass_epa_" + man_zone, _num(row.get("epa")), weight)
            if coverage:
                coverage_labels = (
                    "cover_0", "cover_1", "cover_2", "cover_3", "cover_4",
                    "cover_6", "cover_2_man",
                )
                for label in coverage_labels:
                    off.rate("off_" + label, coverage == label, weight)
                    deff.rate("def_" + label, coverage == label, weight)
                off.mean("off_pass_epa_" + coverage, _num(row.get("epa")), weight)
                deff.mean("def_pass_epa_" + coverage, _num(row.get("epa")), weight)

            position = player_positions.get(str(row.get("receiver_player_id") or "").strip())
            if position:
                for acc, prefix in ((off, "off_"), (deff, "def_")):
                    for label in ("RB", "WR", "TE"):
                        acc.rate(prefix + f"target_{label.lower()}_all", position == label, weight)
                        if man_zone:
                            acc.rate(
                                prefix + f"target_{label.lower()}_{man_zone}",
                                position == label,
                                weight,
                            )

    league_off = _league_accumulators(offense)
    league_def = _league_accumulators(defense)
    source_season = max(source_seasons, default=season - 1)
    prior_coaches = _coach_map(schedule, source_season)
    current_coaches = _coach_map(schedule, season)
    profiles: dict[str, TeamSchemeProfile] = {}
    all_teams = set(offense) | set(defense)
    for team in all_teams:
        off = offense[team]
        deff = defense[team]
        previous = prior_coaches.get(team)
        current = current_coaches.get(team)
        if previous and current:
            same_staff = previous == current
            continuity = "returning head coach" if same_staff else "head coach changed"
            carryover = 1.0 if same_staff else 0.55
        else:
            continuity = "staff continuity unverified"
            carryover = 0.80
        samples = min(off.raw_plays, deff.raw_plays)
        confidence = "high" if samples >= 700 else "medium" if samples >= 350 else "low"
        if carryover < 0.75 and confidence == "high":
            confidence = "medium"
        profiles[team] = TeamSchemeProfile(
            team=team,
            source_seasons=tuple(sorted(source_seasons)),
            participation_source_seasons=participation_seasons,
            charting_source_seasons=charting_seasons,
            offense_plays=off.raw_plays,
            defense_plays=deff.raw_plays,
            offense=_profile_values(off, league_off, defense=False),
            defense=_profile_values(deff, league_def, defense=True),
            confidence=confidence,
            staff_continuity=continuity,
            carryover_weight=carryover,
            coverage_samples=min(off.coverage_samples, deff.coverage_samples),
            charting_samples=min(off.charting_samples, deff.charting_samples),
        )

    league = _profile_values(league_off, league_off, defense=False)
    league.update({"def_" + key: value for key, value in _profile_values(
        league_def, league_def, defense=True
    ).items()})
    matchups = build_matchups(games=games, profiles=profiles, league=league)
    status = {
        "model_version": MODEL_VERSION,
        "profile_count": len(profiles),
        "matchup_count": len(matchups),
        "source_seasons": sorted(source_seasons),
        "pbp_source_seasons": sorted(source_seasons),
        "participation_source_seasons": list(participation_seasons),
        "charting_source_seasons": list(charting_seasons),
        "pbp_plays_ingested": sum(profile.offense_plays for profile in profiles.values()),
        "participation_rows": len(participation_rows),
        "charting_rows": len(charting_rows),
        "observed": list(OBSERVED_FEATURES),
        "proxies": list(PROXY_FEATURES),
        "unavailable": list(UNAVAILABLE_FEATURES),
        "blocking_scheme": "unavailable - no public charted field; no proxy is relabelled",
        "timing": (
            "all plays are strictly before the forecast week; participation coverage and "
            "personnel may remain prior-season until nflverse publishes the season file"
        ),
        "usage": (
            "research challenger: bounded player opportunity and efficiency adjustments; "
            "not an input to the published spread model"
        ),
        "attribution": "FTN Data via nflverse, CC-BY-SA 4.0 (2023 onward)",
    }
    return BuildResult(profiles=profiles, matchups=matchups, league=league, status=status)


def _blend(value: float, baseline: float, weight: float) -> float:
    return baseline + weight * (value - baseline)


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def build_matchups(
    *,
    games: list[dict],
    profiles: dict[str, TeamSchemeProfile],
    league: dict[str, float],
) -> dict[tuple[str, str], SchemeMatchup]:
    """Convert descriptive profiles into bounded, forward matchup adjustments."""
    output: dict[tuple[str, str], SchemeMatchup] = {}
    league_pass = league.get("neutral_pass_rate", 0.56)
    league_pass_epa = league.get("pass_epa", 0.0)
    league_rush_epa = league.get("rush_epa", 0.0)
    for game in games:
        home = teams.canonical(game.get("home_team") or "")
        away = teams.canonical(game.get("away_team") or "")
        for team, opponent in ((away, home), (home, away)):
            offense = profiles.get(team)
            defense = profiles.get(opponent)
            if offense is None or defense is None:
                continue
            carryover = min(offense.carryover_weight, defense.carryover_weight)
            man = _blend(defense.defense.get("man_rate", 0.35), 0.35, carryover)
            zone = _blend(defense.defense.get("zone_rate", 0.65), 0.65, carryover)
            total = max(man + zone, 0.01)
            man, zone = man / total, zone / total
            expected_pass = 0.58 * offense.offense.get("neutral_pass_rate", league_pass)
            expected_pass += 0.42 * defense.defense.get("neutral_pass_rate", league_pass)
            expected_pass = _blend(expected_pass, league_pass, carryover)
            motion = _blend(
                offense.offense.get("motion_rate", league.get("motion_rate", 0.50)),
                league.get("motion_rate", 0.50),
                carryover,
            )
            play_action = _blend(
                offense.offense.get(
                    "play_action_rate", league.get("play_action_rate", 0.22)
                ),
                league.get("play_action_rate", 0.22),
                carryover,
            )
            blitz = _blend(
                defense.defense.get("blitz_rate", league.get("def_blitz_rate", 0.27)),
                league.get("def_blitz_rate", 0.27),
                carryover,
            )
            pressure = _blend(
                defense.defense.get(
                    "pressure_rate", league.get("def_pressure_rate", 0.32)
                ),
                league.get("def_pressure_rate", 0.32),
                carryover,
            )

            offense_response = (
                man * offense.offense.get("pass_epa_man", league_pass_epa)
                + zone * offense.offense.get("pass_epa_zone", league_pass_epa)
            )
            defense_response = (
                man * defense.defense.get("pass_epa_man", league_pass_epa)
                + zone * defense.defense.get("pass_epa_zone", league_pass_epa)
            )
            coverage_epa = (offense_response + defense_response) / 2.0

            def concept_response(metric: str) -> float:
                return (
                    offense.offense.get(metric, league_pass_epa)
                    + defense.defense.get(metric, league_pass_epa)
                ) / 2.0

            motion_epa = (
                motion * concept_response("pass_epa_motion")
                + (1.0 - motion) * concept_response("pass_epa_no_motion")
            )
            play_action_epa = (
                play_action * concept_response("pass_epa_play_action")
                + (1.0 - play_action) * concept_response("pass_epa_no_play_action")
            )
            concept_epa = (motion_epa + play_action_epa) / 2.0
            # Coverage response carries most of the signal. Concept response is
            # descriptive until the prospective ledger has enough outcomes, so
            # it receives a deliberately smaller share of the bounded delta.
            pass_epa = 0.75 * coverage_epa + 0.25 * concept_epa
            rush_epa = (
                offense.offense.get("rush_epa", league_rush_epa)
                + defense.defense.get("rush_epa", league_rush_epa)
            ) / 2.0
            pass_efficiency = _clamp(
                carryover * (pass_epa - league_pass_epa) * 1.15, -0.35, 0.35
            )
            rush_efficiency = _clamp(
                carryover * (rush_epa - league_rush_epa) * 0.80, -0.25, 0.25
            )

            target_multipliers: dict[str, float] = {}
            for position in ("RB", "WR", "TE"):
                key = position.lower()
                fallback = {"rb": .18, "wr": .61, "te": .21}[key]
                baseline = league.get(f"target_share_{key}_all", fallback)
                offense_share = (
                    man * offense.offense.get(f"target_share_{key}_man", baseline)
                    + zone * offense.offense.get(f"target_share_{key}_zone", baseline)
                )
                defense_share = (
                    man * defense.defense.get(f"target_share_{key}_man", baseline)
                    + zone * defense.defense.get(f"target_share_{key}_zone", baseline)
                )
                expected_share = (offense_share + defense_share) / 2.0
                target_multipliers[position] = round(
                    _clamp(
                        1.0 + carryover * (expected_share / max(baseline, .02) - 1.0),
                        .88,
                        1.12,
                    ),
                    4,
                )

            coverages = {
                label: _blend(defense.defense.get(label + "_rate", 0.0), 0.0, carryover)
                for label in (
                    "cover_0", "cover_1", "cover_2", "cover_3", "cover_4",
                    "cover_6", "cover_2_man",
                )
            }
            coverage_total = sum(coverages.values())
            if coverage_total:
                coverages = {
                    key: round(value / coverage_total, 4)
                    for key, value in coverages.items()
                }
            top = max(coverages, key=coverages.get) if coverage_total else "coverage unavailable"
            factors = (
                f"{opponent} expected {man:.0%} man / {zone:.0%} zone",
                f"most frequent family: {top.replace('_', ' ')}",
                f"{team} {offense.staff_continuity}; {opponent} {defense.staff_continuity}",
                "target response is position-level and re-normalized to the team pool",
            )
            confidence = "low" if "low" in {offense.confidence, defense.confidence} else (
                "medium" if "medium" in {offense.confidence, defense.confidence} else "high"
            )
            output[(team, opponent)] = SchemeMatchup(
                team=team,
                opponent=opponent,
                expected_man_rate=round(man, 4),
                expected_zone_rate=round(zone, 4),
                expected_coverages=coverages,
                expected_motion_rate=round(motion, 4),
                expected_play_action_rate=round(play_action, 4),
                expected_blitz_rate=round(blitz, 4),
                expected_pressure_rate=round(pressure, 4),
                neutral_pass_rate=round(expected_pass, 4),
                pass_attempt_delta=round(
                    _clamp((expected_pass - league_pass) * 28.0, -2.0, 2.0), 3
                ),
                carry_delta=round(_clamp((league_pass - expected_pass) * 22.0, -1.6, 1.6), 3),
                pass_efficiency_delta=round(pass_efficiency, 4),
                rush_efficiency_delta=round(rush_efficiency, 4),
                target_multipliers=target_multipliers,
                confidence=confidence,
                factors=factors,
                source_seasons=tuple(
                    sorted(set(offense.source_seasons) | set(defense.source_seasons))
                ),
            )
    return output


def profile_payload(profile: TeamSchemeProfile) -> dict:
    return asdict(profile)


def matchup_payload(matchup: SchemeMatchup) -> dict:
    return asdict(matchup)
