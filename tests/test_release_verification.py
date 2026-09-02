from __future__ import annotations

import json

from scripts.verify_build import verify


def _manifest() -> dict:
    return {
        "generated_at": "2026-09-02T16:00:00Z",
        "season": 2026,
        "week": 1,
        "nflverse": [],
        "issues": [],
        "odds": {
            "state": "fresh",
            "requested_book": "draftkings",
            "events": 272,
            "matched": 272,
            "slate_games": 16,
            "slate_matched": 16,
            "slate_spreads": 16,
            "slate_totals": 16,
            "slate_moneylines": 16,
            "slate_complete": 16,
            "age_seconds": 10,
        },
    }


def test_complete_fresh_exact_book_release_passes(tmp_path) -> None:
    path = tmp_path / "build.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")
    assert verify(path) == []


def test_partial_or_stale_release_fails_closed(tmp_path) -> None:
    payload = _manifest()
    payload["odds"]["slate_complete"] = 15
    payload["odds"]["slate_totals"] = 15
    payload["odds"]["age_seconds"] = 1201
    payload["issues"] = ["one incomplete sportsbook row"]
    path = tmp_path / "build.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    errors = verify(path)
    assert any("complete DraftKings coverage" in error for error in errors)
    assert any("totals coverage" in error for error in errors)
    assert any("snapshot is 1201s old" in error for error in errors)
    assert any("publication issue" in error for error in errors)
