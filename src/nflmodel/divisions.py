"""Projected division winners, seeds and records, by simulating the season.

A power rating answers "who is better". A division race asks something the rating
cannot answer on its own: who plays whom, and how often. The NFL formula gives
every team six games inside its own division and a schedule otherwise determined
by last year's finish, so two equally rated teams can have materially different
paths. The only honest way to turn ratings into division odds is to play the
actual fixture list out, many times.

Method
------
Each scheduled regular-season game is an independent Bernoulli draw on the home
team's win probability, which comes from the same blended margin the board
publishes, put through the normal CDF at the model's residual SD. The season is
replayed `SIMULATIONS` times and the standings are recomputed each pass.

Three things this deliberately does **not** model, stated because each one would
otherwise be an invisible overclaim:

* **Ties.** About 0.4% of NFL games end level -- roughly one a season. Including
  them would change a division probability in the third decimal and complicates
  every tiebreaker, so games are drawn as win/loss.
* **In-season change.** Every simulated week uses the same preseason ratings. A
  team does not get better in October because it won in September, and a
  quarterback does not get injured. Real division odds move; these do not.
* **Correlation.** Draws are independent, so the simulation understates the
  chance of a genuine collapse or a genuine tear -- the tails are thinner than
  reality's.

Tiebreakers
-----------
The NFL's tiebreaking procedure runs to a dozen steps. Implemented here, in
order: head-to-head record between the tied teams, then division win percentage,
then conference win percentage, then a coin flip. That covers the steps that
decide the overwhelming majority of real ties; common games, strength of
victory and the net-points steps are not implemented, and a race that would come
down to them is resolved by the coin flip instead. The alternative -- ordering by
power rating -- would quietly hand every tie to the favourite and inflate the
odds of exactly the teams the model already likes.

Determinism
-----------
The simulation is seeded. A dashboard whose division odds move by half a point
every time it rebuilds, with no new data, is indistinguishable from one that has
a bug, so the same inputs must produce the same page.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import ratings as ratings_mod
from . import teams as teams_mod

SIMULATIONS = 20000
SEED = 20260831

# Per conference: four division winners plus three wild cards.
PLAYOFF_SPOTS = 7
DIVISION_WINNERS = 4


@dataclass(frozen=True)
class SimGame:
    home: str
    away: str
    home_win_probability: float
    division_game: bool
    conference: str  # "" when the two teams are in different conferences


@dataclass
class TeamOutlook:
    team: str
    division: str
    conference: str
    rating: float
    projected_wins: float = 0.0
    win_division: float = 0.0
    make_playoffs: float = 0.0
    top_seed: float = 0.0
    # Distribution of final win totals, for a spark bar on the dashboard.
    win_histogram: dict[int, float] = field(default_factory=dict)

    @property
    def projected_losses(self) -> float:
        return 17.0 - self.projected_wins


def build_games(schedule: list[dict], table: dict[str, float], *,
                margin_of=None) -> list[SimGame]:
    """Turn scheduled games into win probabilities.

    `margin_of(home, away, neutral)` overrides the rating-only margin, so the
    caller can feed the same blended margin the board publishes rather than a
    second, quietly different estimate of the same thing.
    """
    out: list[SimGame] = []
    for row in schedule:
        if str(row.get("game_type") or "").upper() != "REG":
            continue
        home = teams_mod.canonical(row.get("home_team") or "")
        away = teams_mod.canonical(row.get("away_team") or "")
        if home not in table or away not in table:
            continue
        neutral = str(row.get("location") or "Home").lower() != "home"
        margin = None
        if margin_of is not None:
            margin = margin_of(home, away, neutral)
        if margin is None:
            margin = ratings_mod.projected_margin(table, home, away, neutral=neutral)
        if margin is None:
            continue
        home_team, away_team = teams_mod.get(home), teams_mod.get(away)
        out.append(SimGame(
            home=home, away=away,
            home_win_probability=ratings_mod.win_probability(margin),
            division_game=home_team.division == away_team.division,
            conference=(home_team.conference
                        if home_team.conference == away_team.conference else ""),
        ))
    return out


def _standings_order(teams: list[str], wins, division_wins, conference_wins,
                     head_to_head, rng) -> list[str]:
    """Sort tied teams by the implemented tiebreakers, coin-flipping the rest."""
    def key(team: str):
        return (-wins[team], -division_wins[team], -conference_wins[team], rng.random())

    ordered = sorted(teams, key=key)
    # Head-to-head outranks division record, but only where it is decisive: a
    # sweep between exactly two tied teams. Applying it to three-way ties needs
    # the full NFL procedure and is left to the steps below.
    for i in range(len(ordered) - 1):
        a, b = ordered[i], ordered[i + 1]
        if wins[a] != wins[b]:
            continue
        pair = head_to_head.get((a, b), 0) - head_to_head.get((b, a), 0)
        if pair < 0:
            ordered[i], ordered[i + 1] = b, a
    return ordered


def simulate(games: list[SimGame], table: dict[str, float], *,
             simulations: int = SIMULATIONS, seed: int = SEED) -> list[TeamOutlook]:
    """Replay the season and return every team's outlook."""
    names = sorted(table)
    if not games or not names:
        return []
    rng = random.Random(seed)
    outlooks = {
        team: TeamOutlook(team=team, division=teams_mod.get(team).division,
                          conference=teams_mod.get(team).conference,
                          rating=table[team])
        for team in names
    }
    divisions: dict[str, list[str]] = {}
    for team in names:
        divisions.setdefault(teams_mod.get(team).division, []).append(team)
    conferences: dict[str, list[str]] = {}
    for team in names:
        conferences.setdefault(teams_mod.get(team).conference, []).append(team)

    totals = {t: 0.0 for t in names}
    division_titles = {t: 0 for t in names}
    playoffs = {t: 0 for t in names}
    top_seeds = {t: 0 for t in names}
    histogram: dict[str, dict[int, int]] = {t: {} for t in names}

    for _ in range(simulations):
        wins = dict.fromkeys(names, 0)
        division_wins = dict.fromkeys(names, 0)
        conference_wins = dict.fromkeys(names, 0)
        head_to_head: dict[tuple[str, str], int] = {}
        for game in games:
            home_won = rng.random() < game.home_win_probability
            winner, loser = ((game.home, game.away) if home_won
                             else (game.away, game.home))
            wins[winner] += 1
            if game.division_game:
                division_wins[winner] += 1
            if game.conference:
                conference_wins[winner] += 1
            head_to_head[(winner, loser)] = head_to_head.get((winner, loser), 0) + 1

        for team in names:
            totals[team] += wins[team]
            histogram[team][wins[team]] = histogram[team].get(wins[team], 0) + 1

        seeded: dict[str, list[str]] = {}
        for division, members in divisions.items():
            order = _standings_order(members, wins, division_wins, conference_wins,
                                     head_to_head, rng)
            champion = order[0]
            division_titles[champion] += 1
            seeded.setdefault(teams_mod.get(champion).conference, []).append(champion)

        for conference, members in conferences.items():
            champions = seeded.get(conference, [])
            ranked_champions = _standings_order(champions, wins, division_wins,
                                                conference_wins, head_to_head, rng)
            if ranked_champions:
                top_seeds[ranked_champions[0]] += 1
            rest = [t for t in members if t not in champions]
            wildcards = _standings_order(rest, wins, division_wins, conference_wins,
                                         head_to_head, rng)[:PLAYOFF_SPOTS - DIVISION_WINNERS]
            for team in ranked_champions + wildcards:
                playoffs[team] += 1

    for team, outlook in outlooks.items():
        outlook.projected_wins = totals[team] / simulations
        outlook.win_division = division_titles[team] / simulations
        outlook.make_playoffs = playoffs[team] / simulations
        outlook.top_seed = top_seeds[team] / simulations
        outlook.win_histogram = {w: c / simulations
                                 for w, c in sorted(histogram[team].items())}
    return sorted(outlooks.values(), key=lambda o: (-o.win_division, -o.projected_wins))


def by_division(outlooks: list[TeamOutlook]) -> dict[str, list[TeamOutlook]]:
    """Grouped and ordered the way a standings page reads."""
    grouped: dict[str, list[TeamOutlook]] = {name: [] for name in teams_mod.DIVISIONS}
    for outlook in outlooks:
        grouped.setdefault(outlook.division, []).append(outlook)
    for members in grouped.values():
        members.sort(key=lambda o: (-o.projected_wins, -o.win_division))
    return grouped


def projected_champions(outlooks: list[TeamOutlook]) -> dict[str, TeamOutlook]:
    """The most likely winner of each division."""
    best: dict[str, TeamOutlook] = {}
    for outlook in outlooks:
        current = best.get(outlook.division)
        if current is None or outlook.win_division > current.win_division:
            best[outlook.division] = outlook
    return best
