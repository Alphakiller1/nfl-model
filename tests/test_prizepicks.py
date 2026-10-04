import json
from dataclasses import replace
from datetime import timedelta
from html import escape

import pytest

from nflmodel.sources import prizepicks
from test_weekly_props import KICKOFF, NOW, player, quote, slate


def page(p, stat="Rush Yards", line=59.5, *, kickoff=None, team=None, date="Oct 5, 4:00 AM ET"):
    label = f"{p.player_name}, {line} {stat} projection"
    stream = (
        "game:"
        + json.dumps({"startTime": kickoff or KICKOFF.isoformat()})
        + "\n"
        + "card:"
        + json.dumps(["$", "article", "12345", {"aria-label": label}], separators=(",", ":"))
        + "\n"
    )
    return (
        f"<script>self.__next_f.push({json.dumps([1, stream])})</script>"
        f'<article aria-label="{escape(label, quote=True)}"><p>NFL</p>'
        f"<h3>{escape(p.player_name)}</h3><p>{team or p.team} · {p.position}</p>"
        f"<span>{p.team} vs {p.opponent}</span><span>{date}</span><p>Projection</p>"
        f"<span>{stat}</span><span>{line}</span><span>Less</span><span>More</span></article>"
    )


def parse(html, p=None):
    return prizepicks.parse(
        html, p or player(), observed_at=NOW.isoformat(), source_url=prizepicks.BASE + "player"
    )


def test_observed_cards_keep_actual_ids_lines_sides_and_source_uncertainty():
    p = player()
    rows = parse(page(p), p)
    assert len(rows) == 1
    q = rows[0]
    assert q.projection_id == "12345" and q.line == 59.5 and q.metric == "rushing_yards"
    assert q.player_id == p.player_id and q.kickoff_utc == KICKOFF.isoformat()
    assert q.variant == "unverified" and not q.entry_availability_verified
    assert q.published_sides == ("MORE", "LESS")


@pytest.mark.parametrize(
    "change",
    [
        {"kickoff": (KICKOFF + timedelta(days=7)).isoformat()},
        {"team": "BUF"},
        {"date": "Oct 6, 4:00 AM ET"},
        {"stat": "Pass Yards 1H"},
        {"stat": "Fantasy Score"},
    ],
)
def test_wrong_fixture_partial_game_and_unsupported_stats_are_not_imported(change):
    p = player()
    assert parse(page(p, **change), p) == []


def test_missing_id_unidentified_player_and_missing_card_sides_fail_closed():
    p = player()
    html = page(p)
    assert parse(html.replace("self.__next_f.push", "unrecognized"), p) == []
    assert parse(html, replace(p, player_name="Someone Else")) == []
    assert parse(html.replace("<span>More</span>", ""), p) == []


def test_snapshot_is_prizepicks_only_and_excludes_stale_observations(tmp_path, monkeypatch):
    p = player()
    path = tmp_path / "lines.json"
    prizepicks.write_snapshot(
        [
            quote(p),
            replace(
                quote(p),
                projection_id="stale",
                observed_at_utc=(NOW - timedelta(hours=2)).isoformat(),
            ),
            replace(quote(p), projection_id="other", provider="draftkings"),
        ],
        path,
    )
    monkeypatch.setenv("NFL_PRIZEPICKS_SNAPSHOT_PATH", str(path))
    rows, status = prizepicks.fetch([p], now=NOW)
    assert len(rows) == 1 and rows[0].provider == "prizepicks"
    assert status["state"] == "fresh"


def test_missing_prizepicks_feed_has_no_cross_provider_fallback(monkeypatch):
    p = player()
    monkeypatch.setattr(
        prizepicks,
        "_page",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("feed unavailable")),
    )
    rows, status = prizepicks.fetch([p], now=NOW)
    assert rows == [] and status["state"] == "unavailable"
    assert len(status["errors"]) == 1


def test_observed_one_sided_projection_does_not_select_unoffered_less():
    from nflmodel import weekly_props

    p = player()
    # At a very high line LESS would be the winning choice if both sides existed.
    q = replace(quote(p, line=150.5), published_sides=("MORE",))
    report = weekly_props.build(slate([p], [q]), now=NOW)
    assert report["groups"]["RB"]["rows"][0]["selection"] == "MORE"


@pytest.mark.parametrize(
    "position,stat,metric",
    [
        ("QB", "Pass ATT", "pass_attempts"),
        ("RB", "Rush ATT", "rush_attempts"),
        ("K", "Kick Pts", "kicking_points"),
    ],
)
def test_verified_abbreviated_attempt_and_kicking_labels_parse(position, stat, metric):
    p = player(position)
    rows = parse(page(p, stat=stat, line=7.5), p)
    assert len(rows) == 1 and rows[0].metric == metric


@pytest.mark.parametrize(
    "bad", [{"projection_id": 123}, {"source_url": None}, {"published_sides": None}]
)
def test_invalid_snapshot_attribution_and_side_metadata_fail_closed(bad):
    from nflmodel import weekly_props

    p = player()
    report = weekly_props.build(slate([p], [replace(quote(p), **bad)]), now=NOW)
    assert report["groups"]["RB"]["published"] == 0
    assert report["excluded"]["unsupported_or_unattributed_projection"] == 1
