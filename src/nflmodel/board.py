"""
Chase Analytics **Board kernel** — the shared slate surface for every sport model.

This file is VENDORED VERBATIM into mlb-model, wnba-edge-model and nfl-model. It has no
imports outside the standard library and no knowledge of any sport, so the same card
anatomy renders MLB, WNBA and NFL. `tests/test_board_contract.py` in each repo checks the
copies have not drifted; keep them byte-identical and change all three together.

The structure is fixed; only the *slots* differ per sport:

    Board                      MLB                WNBA                NFL
    ─────────────────────────────────────────────────────────────────────────────
    card.principals            starting pitchers  usage leaders       quarterbacks
    card.groups                Full Game, First5  Full Game           Full Game
    card.away/home.score       expected runs      projected points    projected points

A sport adapter's only job is building `Board` objects. Rendering, filtering, sorting,
counting and empty states all live here, so a fix lands in every model at once.

Design contract: chase_tokens.css (deep navy #08090F, violet brand #9A6BFF, gold eyebrow
labels, DM Sans body / Roboto Condensed display, hairline borders + layered shadows —
never uniform violet outlines).
"""
from __future__ import annotations

import html
from dataclasses import dataclass, field
from pathlib import Path

BOARD_CONTRACT_VERSION = "1"

e = html.escape

# A "gem" is the model's strongest actionable edge. Defined once, here, so the count in
# the header, the chip on the card and the GEMS filter can never disagree.
GEM_EDGE_PTS = 3.0
GEM_STATES = frozenset({"BET", "MONITOR"})

# Tones map to the token colour classes in chase_tokens.css.
_TONES = frozenset({"pos", "neg", "warnc", "side", "mut"})


def _tone(value: str) -> str:
    return value if value in _TONES else "mut"


def _onclick(handler: str) -> str:
    """Optional inline handler. Kept out of the f-string so this file parses on 3.11."""
    return f' onclick="{handler}"' if handler else ""


@dataclass(frozen=True)
class Tile:
    """One market inside a group — the smallest unit on the board.

    ``priced`` is explicit rather than inferred from ``state``: a model probability with no
    matched book price is a projection, not a pick, and must never inflate the Picks counter.
    """

    label: str
    value: str = "—"
    state: str = "No market"
    tone: str = "mut"
    note: str = ""
    gem: bool = False
    priced: bool = False

    @property
    def is_priced(self) -> bool:
        return self.priced


@dataclass(frozen=True)
class Group:
    """A market family — "Full Game", "First 5 Innings", "First Half"."""

    label: str
    tiles: tuple[Tile, ...] = ()
    state: str = ""
    tag: str = ""  # filter tag, e.g. "fullgame" / "f5"

    @property
    def priced(self) -> int:
        return sum(1 for tile in self.tiles if tile.is_priced)


@dataclass(frozen=True)
class Side:
    """One team on a card."""

    abbr: str
    score: str = "—"
    detail: str = ""
    logo_html: str = ""
    favored: bool = False


@dataclass(frozen=True)
class Principal:
    """The player a card is read through: SP, QB, usage leader."""

    name: str
    team: str = ""
    stat: str = ""
    art_html: str = ""


@dataclass(frozen=True)
class Card:
    key: str
    away: Side
    home: Side
    league: str = ""
    start_text: str = ""
    status_label: str = ""
    status_tone: str = "mut"
    headline: str = ""
    headline_tone: str = "side"
    principals: tuple[Principal, ...] = ()
    principal_label: str = ""
    groups: tuple[Group, ...] = ()
    action_label: str = ""
    action_js: str = ""
    footer_label: str = ""
    footer_js: str = ""
    note: str = ""

    @property
    def picks(self) -> int:
        return sum(group.priced for group in self.groups)

    @property
    def gems(self) -> int:
        return sum(1 for group in self.groups for tile in group.tiles if tile.gem)

    @property
    def tags(self) -> tuple[str, ...]:
        tags = {group.tag for group in self.groups if group.tag and group.priced}
        if self.gems:
            tags.add("gems")
        return tuple(sorted(tags))


@dataclass(frozen=True)
class Counter:
    label: str
    value: str
    tone: str = ""


@dataclass
class Board:
    """A whole slate: the header, its filters and every game card."""

    sport: str
    cards: list[Card] = field(default_factory=list)
    date_label: str = ""
    meta: list[str] = field(default_factory=list)
    filters: list[tuple[str, str]] = field(default_factory=list)
    sorts: list[tuple[str, str]] = field(default_factory=list)
    empty_text: str = "No games on this slate."

    @property
    def picks(self) -> int:
        return sum(card.picks for card in self.cards)

    @property
    def gems(self) -> int:
        return sum(card.gems for card in self.cards)


# ── rendering ────────────────────────────────────────────────────────────────


def _tile_html(tile: Tile) -> str:
    classes = ["bd-tile", f"is-{_tone(tile.tone)}"]
    if tile.gem:
        classes.append("is-gem")
    if not tile.is_priced:
        classes.append("is-idle")
    title = f' title="{e(tile.note)}"' if tile.note else ""
    gem = '<span class="bd-tile__gem" aria-hidden="true">◆</span>' if tile.gem else ""
    return (
        f'<div class="{" ".join(classes)}"{title}>'
        f'<span class="bd-tile__label">{e(tile.label)}</span>'
        f'<span class="bd-tile__value">{gem}{e(tile.value)}</span>'
        f'<span class="bd-tile__state">{e(tile.state)}</span>'
        f"</div>"
    )


def _group_html(group: Group) -> str:
    if not group.tiles:
        return ""
    priced = group.priced
    count = f'{priced} market{"" if priced == 1 else "s"}' if priced else "no price"
    state = f'<span class="bd-group__state">{e(group.state)}</span>' if group.state else ""
    tiles = "".join(_tile_html(tile) for tile in group.tiles)
    return (
        f'<div class="bd-group">'
        f'<div class="bd-group__head">'
        f'<span class="bd-group__label">{e(group.label)}</span>{state}'
        f'<span class="bd-group__count">{e(count)}</span>'
        f"</div>"
        f'<div class="bd-group__tiles">{tiles}</div>'
        f"</div>"
    )


def _side_html(side: Side, *, align: str) -> str:
    logo = side.logo_html or f'<span class="bd-side__fallback">{e(side.abbr[:3])}</span>'
    detail = f'<span class="bd-side__detail">{e(side.detail)}</span>' if side.detail else ""
    fav = " is-fav" if side.favored else ""
    return (
        f'<div class="bd-side bd-side--{align}{fav}">'
        f'<span class="bd-side__logo">{logo}</span>'
        f'<span class="bd-side__body">'
        f'<span class="bd-side__abbr">{e(side.abbr)}</span>{detail}</span>'
        f'<span class="bd-side__score">{e(side.score)}</span>'
        f"</div>"
    )


def _principal_html(principal: Principal) -> str:
    # Sports without a headshot feed (WNBA, NFL) get a monogram rather than an empty disc,
    # so a card never looks like an image failed to load.
    initial = e(principal.name.strip()[:1].upper()) if principal.name.strip() else ""
    art = principal.art_html or f'<span class="bd-face bd-face--na">{initial}</span>'
    stat = f'<span class="bd-principal__stat">{e(principal.stat)}</span>' if principal.stat else ""
    team = f'<span class="bd-principal__team">{e(principal.team)}</span>' if principal.team else ""
    return (
        f'<div class="bd-principal">{art}'
        f'<span class="bd-principal__body">'
        f'<span class="bd-principal__name">{e(principal.name)}</span>'
        f"{team}{stat}</span></div>"
    )


def _card_html(card: Card) -> str:
    counts = ""
    if card.picks:
        counts += f'<span class="bd-count">{card.picks}</span>'
    if card.gems:
        counts += f'<span class="bd-count is-gem">◆ {card.gems}</span>'

    status = (
        f'<span class="bd-status is-{_tone(card.status_tone)}">{e(card.status_label)}</span>'
        if card.status_label
        else ""
    )
    league = f'<span class="bd-card__league">{e(card.league)}</span>' if card.league else ""
    start = f'<span class="bd-card__start">{e(card.start_text)}</span>' if card.start_text else ""

    headline = (
        f'<button type="button" class="bd-headline is-{_tone(card.headline_tone)}"'
        f"{_onclick(card.action_js)}>"
        f'<span class="bd-headline__text">{e(card.headline)}</span>'
        f'<span class="bd-headline__cta">{e(card.action_label or "Open")} →</span>'
        f"</button>"
        if card.headline
        else ""
    )

    principals = ""
    if card.principals:
        label = (
            f'<span class="bd-principals__label">{e(card.principal_label)}</span>'
            if card.principal_label
            else ""
        )
        inner = '<span class="bd-principals__vs">vs</span>'.join(
            _principal_html(principal) for principal in card.principals
        )
        principals = (
            f'<div class="bd-principals">{label}'
            f'<div class="bd-principals__row">{inner}</div></div>'
        )

    groups = "".join(_group_html(group) for group in card.groups)
    if not groups:
        groups = f'<div class="bd-empty">{e(card.note or "No priced markets for this game.")}</div>'

    footer = (
        f'<button type="button" class="bd-card__footer"{_onclick(card.footer_js)}>'
        f"{e(card.footer_label)} →</button>"
        if card.footer_label
        else ""
    )

    return (
        f'<article class="bd-card" data-key="{e(card.key)}" '
        f'data-tags="{e(" ".join(card.tags))}" data-picks="{card.picks}" data-gems="{card.gems}">'
        f'<header class="bd-card__head">{league}{status}{start}'
        f'<span class="bd-card__counts">{counts}</span></header>'
        f"{headline}"
        f'<div class="bd-score">{_side_html(card.away, align="away")}'
        f'<span class="bd-score__vs">vs</span>'
        f'{_side_html(card.home, align="home")}</div>'
        f"{principals}"
        f'<div class="bd-card__groups">{groups}</div>'
        f"{footer}"
        f"</article>"
    )


def board_html(board: Board) -> str:
    """Render a whole board: wordmark header, counters, filters, card grid."""
    cards = "".join(_card_html(card) for card in board.cards)
    if not cards:
        cards = f'<div class="bd-empty bd-empty--board">{e(board.empty_text)}</div>'

    counters = [
        Counter("Picks", str(board.picks)),
        Counter("Gems", str(board.gems), "gem"),
        Counter("Games", str(len(board.cards))),
    ]
    counter_html = "".join(
        f'<div class="bd-counter{" is-gem" if counter.tone == "gem" else ""}">'
        f'<span class="bd-counter__v">{e(counter.value)}</span>'
        f'<span class="bd-counter__k">{e(counter.label)}</span></div>'
        for counter in counters
    )

    filters = board.filters or [("all", "All"), ("gems", "◆ Gems")]
    filter_html = "".join(
        f'<button type="button" class="bd-filter{" on" if tag == "all" else ""}" '
        f'data-filter="{e(tag)}" onclick="boardFilter(this,\'{e(tag)}\')">{e(label)}</button>'
        for tag, label in filters
    )

    sorts = board.sorts or [("start", "Start time")]
    sort_html = "".join(
        f'<option value="{e(key)}">{e(label)}</option>' for key, label in sorts
    )

    meta = " · ".join(part for part in board.meta if part)
    meta_html = f'<div class="bd-head__meta">// {e(meta)}</div>' if meta else ""

    return (
        f'<section class="bd" data-board="{e(board.sport)}">'
        f'<header class="bd-head">'
        f'<div class="bd-head__title">'
        f'<h2 class="bd-wordmark">{e(board.sport)} <em>Board</em></h2>'
        f'<div class="bd-counters">{counter_html}</div>'
        f"</div>"
        f"{meta_html}"
        f'<div class="bd-controls">'
        f'<div class="bd-filters" role="group" aria-label="Filter board">{filter_html}</div>'
        f'<label class="bd-sort">Sort'
        f'<select onchange="boardSort(this.value)" aria-label="Sort board">{sort_html}</select>'
        f"</label></div>"
        f"</header>"
        f'<div class="bd-grid" id="bdGrid">{cards}</div>'
        f"</section>"
    )


BOARD_JS = (
    "function boardFilter(btn,tag){"
    "document.querySelectorAll('.bd-filter').forEach(function(b){b.classList.toggle('on',b===btn);});"
    "document.querySelectorAll('.bd-card').forEach(function(c){"
    "var tags=(c.dataset.tags||'').split(' ');"
    "c.hidden=!(tag==='all'||tags.indexOf(tag)>=0);});"
    "var grid=document.getElementById('bdGrid');"
    "if(grid){var vis=grid.querySelectorAll('.bd-card:not([hidden])').length;"
    "var msg=document.getElementById('bdFilterEmpty');"
    "if(!vis&&!msg){msg=document.createElement('div');msg.id='bdFilterEmpty';"
    "msg.className='bd-empty bd-empty--board';msg.textContent='No games match this filter.';"
    "grid.appendChild(msg);}else if(msg){msg.remove();}}}"
    "function boardSort(key){var grid=document.getElementById('bdGrid');if(!grid)return;"
    "var cards=Array.prototype.slice.call(grid.querySelectorAll('.bd-card'));"
    "cards.sort(function(a,b){"
    "if(key==='picks')return (+b.dataset.picks)-(+a.dataset.picks);"
    "if(key==='gems')return (+b.dataset.gems)-(+a.dataset.gems);"
    "return (a.dataset.key||'').localeCompare(b.dataset.key||'');});"
    "cards.forEach(function(c){grid.appendChild(c);});}"
)


def board_css() -> str:
    return (Path(__file__).resolve().parent / "static" / "board.css").read_text(
        encoding="utf-8"
    )
