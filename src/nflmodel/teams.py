"""The 32 franchises: identity, conference, division, and how they are drawn.

Three jobs, and the third is the one that is easy to get wrong.

**Relocation aliases.** nflverse keeps the abbreviation a team carried *at the
time*, so 2016-2019 rows say ``OAK`` and 2016-2017 rows say ``SD``. A rating
solve that treats those as separate franchises splits ten seasons of Raiders
results across two teams and rates both of them badly. `canonical()` folds them
forward, and every module that touches historical rows calls it.

**Accent colours are display colours, not brand colours.** The Chase palette puts
these on a #08090F ground, where Chicago navy (#0B162A) and Raiders black are
invisible. Each accent below is the readable member of the team's own palette,
picked so a hairline reads at a glance -- it is a UI decision, and stating that
here stops a future reader from "fixing" it back to the unreadable official hex.

Logos come from ESPN's public CDN, which a GitHub Pages host can reach. Its
slugs disagree with nflverse for exactly three teams (LA/LAR, WAS/WSH, and the
already-aliased relocations), so the mapping is explicit rather than lowercased.
"""

from __future__ import annotations

from dataclasses import dataclass

AFC = "AFC"
NFC = "NFC"

# Historical abbreviation -> the franchise it became.
ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "WSH": "WAS", "WFT": "WAS"}


def canonical(abbr: str) -> str:
    """Fold a historical abbreviation onto today's franchise."""
    key = str(abbr or "").strip().upper()
    return ALIASES.get(key, key)


@dataclass(frozen=True)
class Team:
    abbr: str
    name: str        # "Buffalo Bills"
    short: str       # "Bills"
    city: str
    conference: str
    division: str    # "AFC East"
    accent: str
    espn: str

    @property
    def logo(self) -> str:
        return f"https://a.espncdn.com/i/teamlogos/nfl/500/{self.espn}.png"


def _t(abbr, city, short, conference, side, accent, espn=None) -> Team:
    return Team(
        abbr=abbr,
        name=f"{city} {short}",
        short=short,
        city=city,
        conference=conference,
        division=f"{conference} {side}",
        accent=accent,
        espn=espn or abbr.lower(),
    )


TEAMS: dict[str, Team] = {t.abbr: t for t in (
    _t("BUF", "Buffalo", "Bills", AFC, "East", "#2A6AE0"),
    _t("MIA", "Miami", "Dolphins", AFC, "East", "#00B7BF"),
    _t("NE", "New England", "Patriots", AFC, "East", "#D9304A"),
    _t("NYJ", "New York", "Jets", AFC, "East", "#2E9E63"),
    _t("BAL", "Baltimore", "Ravens", AFC, "North", "#7B5BD6"),
    _t("CIN", "Cincinnati", "Bengals", AFC, "North", "#FB4F14"),
    _t("CLE", "Cleveland", "Browns", AFC, "North", "#FF5A1F"),
    _t("PIT", "Pittsburgh", "Steelers", AFC, "North", "#FFB612"),
    _t("HOU", "Houston", "Texans", AFC, "South", "#D3223C"),
    _t("IND", "Indianapolis", "Colts", AFC, "South", "#3E7FD0"),
    _t("JAX", "Jacksonville", "Jaguars", AFC, "South", "#D7A22A"),
    _t("TEN", "Tennessee", "Titans", AFC, "South", "#4B9CD3"),
    _t("DEN", "Denver", "Broncos", AFC, "West", "#FB4F14"),
    _t("KC", "Kansas City", "Chiefs", AFC, "West", "#E31837"),
    _t("LV", "Las Vegas", "Raiders", AFC, "West", "#C7CDD1"),
    _t("LAC", "Los Angeles", "Chargers", AFC, "West", "#0080C6"),
    _t("DAL", "Dallas", "Cowboys", NFC, "East", "#4C7FC9"),
    _t("NYG", "New York", "Giants", NFC, "East", "#3763CE"),
    _t("PHI", "Philadelphia", "Eagles", NFC, "East", "#17A398"),
    _t("WAS", "Washington", "Commanders", NFC, "East", "#C0392B", espn="wsh"),
    _t("CHI", "Chicago", "Bears", NFC, "North", "#E45C1E"),
    _t("DET", "Detroit", "Lions", NFC, "North", "#3EA0DE"),
    _t("GB", "Green Bay", "Packers", NFC, "North", "#FFB612"),
    _t("MIN", "Minnesota", "Vikings", NFC, "North", "#8B5CD6"),
    _t("ATL", "Atlanta", "Falcons", NFC, "South", "#E01931"),
    _t("CAR", "Carolina", "Panthers", NFC, "South", "#0085CA"),
    _t("NO", "New Orleans", "Saints", NFC, "South", "#D3BC8D"),
    _t("TB", "Tampa Bay", "Buccaneers", NFC, "South", "#E82A2A"),
    _t("ARI", "Arizona", "Cardinals", NFC, "West", "#C4123F"),
    _t("LA", "Los Angeles", "Rams", NFC, "West", "#4C7FE0", espn="lar"),
    _t("SEA", "Seattle", "Seahawks", NFC, "West", "#69BE28"),
    _t("SF", "San Francisco", "49ers", NFC, "West", "#D14040"),
)}

DIVISIONS: tuple[str, ...] = (
    "AFC East", "AFC North", "AFC South", "AFC West",
    "NFC East", "NFC North", "NFC South", "NFC West",
)

_UNKNOWN = Team("?", "Unknown", "Unknown", "", "", "", "#5B6172", "nfl")


def get(abbr: str) -> Team:
    """Never raises. An unmapped abbreviation renders as a neutral placeholder
    rather than taking down a whole board."""
    return TEAMS.get(canonical(abbr), _UNKNOWN)


def division_of(abbr: str) -> str:
    return get(abbr).division


def members(division: str) -> tuple[str, ...]:
    return tuple(sorted(t.abbr for t in TEAMS.values() if t.division == division))


def all_abbrs() -> tuple[str, ...]:
    return tuple(sorted(TEAMS))
