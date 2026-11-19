"""Point-in-time, settlement, persistence and CLV contracts shared by four models."""

import importlib.util
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "track_projections.py"
if not SCRIPT.exists():
    SCRIPT = Path(__file__).with_name("track_projections.py")
SPEC = importlib.util.spec_from_file_location("projection_tracker", SCRIPT)
tracker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tracker)
NOW = datetime(2026, 11, 18, 15, tzinfo=timezone.utc)


def board(count=1, sport="nba", players=False):
    games = []
    for i in range(count):
        games.append(
            {
                "id": str(100 + i),
                "home": "H" + str(i),
                "away": "A" + str(i),
                "kickoff_utc": tracker.stamp(NOW + timedelta(hours=8)),
                "model_margin": 5.0 + i / 10,
                "projected_total": 150.0,
                "model_win_probability": 0.6,
            }
        )
    return {
        "season": 2026,
        "week": 11,
        "sport": sport,
        "generated_at_utc": tracker.stamp(NOW),
        "model_version": "test-v1",
        "games": games,
        "player_projections": [
            {
                "game_id": "100",
                "player_id": "77",
                "player_name": "A Player",
                "stats": {"points": {"mean": 25.0}},
            }
        ]
        if players
        else [],
    }


class Provider:
    issues = []
    receipts = {}

    def __init__(self, final=False, price=-110, line=-3.5, closing=-5.0, props=False):
        self.final, self.price, self.line, self.closing, self.props = (
            final,
            price,
            line,
            closing,
            props,
        )

    def collect(self, game):
        return {
            "event_id": game["game_id"],
            "pre": not self.final,
            "final": self.final,
            "canceled": False,
            "tip": game["tip"],
            "source": "ESPN / DraftKings",
            "retrieved_at": tracker.stamp(NOW),
            "quotes": {
                "spread:home": {"line": self.line, "price": self.price},
                "spread:away": {"line": -self.line, "price": self.price},
                "total:over": {"line": 145.0, "price": self.price},
                "total:under": {"line": 145.0, "price": self.price},
            },
            "closing": {
                "spread:home": {"line": self.closing, "price": -110},
                "total:over": {"line": 147.0, "price": -110},
            },
            "scores": {"home": 80, "away": 70},
            "players": {"77": {"name": "A Player", "played": True, "metrics": {"points": 27.0}}},
            "prop_quotes": {("77", "points"): {"line": 23.5, "price": None}} if self.props else {},
        }


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def week(self):
        manifest = tracker.read(self.root / "manifest.json")
        return tracker.read(self.root / next(iter(manifest["weeks"].values()))["file"])

    def test_all_projections_and_wagers_survive_featured_limit(self):
        tracker.run(board(31), "nba", self.root, now=NOW, provider=Provider())
        week = self.week()
        self.assertEqual(len(week["observations"]), 31)
        self.assertEqual(len(week["wagers"]), 62)
        self.assertEqual(sum(w["top20"] for w in week["wagers"].values()), 20)
        self.assertTrue((self.root / "index.html").exists())
        self.assertTrue(next(self.root.glob("*.csv")).exists())

    def test_repeated_build_is_idempotent_and_entry_never_changes(self):
        source = board()
        tracker.run(source, "nba", self.root, now=NOW, provider=Provider())
        original = list(self.week()["wagers"].values())[0]
        source["games"][0]["model_margin"] = -15
        source["generated_at_utc"] = tracker.stamp(NOW + timedelta(hours=1))
        tracker.run(
            source, "nba", self.root, now=NOW + timedelta(hours=1), provider=Provider(line=-4.5)
        )
        tracker.run(
            source, "nba", self.root, now=NOW + timedelta(hours=2), provider=Provider(line=-4.5)
        )
        week = self.week()
        wager = week["wagers"][original["id"]]
        self.assertEqual(
            (wager["side"], wager["line"], wager["recorded_at"]),
            (original["side"], original["line"], original["recorded_at"]),
        )
        self.assertEqual(len(week["observations"]), 2)
        self.assertEqual(len(wager["quote_history"]), 2)
        self.assertEqual(wager["last_prestart_quote"]["line"], -4.5)

    def test_grade_at_original_number_with_real_price_and_provider_close(self):
        source = board()
        tracker.run(source, "nba", self.root, now=NOW, provider=Provider(price=120))
        tracker.run(
            source, "nba", self.root, now=NOW + timedelta(days=1), provider=Provider(final=True)
        )
        wagers = list(self.week()["wagers"].values())
        spread = next(w for w in wagers if w["market"] == "spread")
        self.assertEqual(spread["result"], "win")
        self.assertEqual(spread["units"], 1.2)
        self.assertEqual(spread["clv"], 1.5)
        self.assertEqual(spread["clv_status"], "provider_close")

    def test_missing_price_is_not_replaced_with_minus_110(self):
        tracker.run(board(), "nba", self.root, now=NOW, provider=Provider(price=None))
        tracker.run(
            board(), "nba", self.root, now=NOW + timedelta(days=1), provider=Provider(final=True)
        )
        rows = list(self.week()["wagers"].values())
        self.assertTrue(all(w["units"] is None for w in rows))
        self.assertEqual(tracker.stats(rows)["settled_without_price"], 2)

    def test_props_and_unpriced_projection_accuracy_are_preserved(self):
        tracker.run(board(players=True), "nba", self.root, now=NOW, provider=Provider(props=True))
        tracker.run(
            board(players=True),
            "nba",
            self.root,
            now=NOW + timedelta(days=1),
            provider=Provider(final=True),
        )
        week = self.week()
        prop = next(w for w in week["wagers"].values() if w["market"] == "prop")
        self.assertEqual(prop["result"], "win")
        self.assertIsNone(prop["units"])
        self.assertIsNone(prop["clv"])
        self.assertEqual(prop["clv_status"], "missing_close")
        obs = next(o for o in week["observations"].values() if o["evaluation_eligible"])
        self.assertEqual(obs["player_results"][0]["absolute_errors"]["points"], 2.0)

    def test_post_start_projection_is_archived_but_cannot_be_a_selection(self):
        tracker.run(
            board(players=True),
            "nba",
            self.root,
            now=NOW + timedelta(days=1),
            provider=Provider(final=True),
        )
        week = self.week()
        self.assertEqual(len(week["observations"]), 1)
        self.assertEqual(len(week["wagers"]), 0)
        self.assertFalse(next(iter(week["observations"].values()))["evaluation_eligible"])

    def test_official_football_week_and_eastern_monday_boundary(self):
        monday = datetime(2026, 11, 17, 1, tzinfo=timezone.utc)
        self.assertEqual(tracker.week_id("nfl", 2026, 11, monday), "nfl-2026-week-11")
        self.assertEqual(tracker.week_id("cfb", 2026, 12, monday), "cfb-2026-week-12")
        sunday_eastern = datetime(2026, 11, 16, 3, tzinfo=timezone.utc)
        self.assertEqual(tracker.week_id("cbb", 2027, None, sunday_eastern), "cbb-2026-11-09")

    def test_clv_signs_for_each_selected_side(self):
        cases = [
            ("spread", "home", -3, -5, 2),
            ("spread", "away", 7, 5, 2),
            ("total", "over", 145, 148, 3),
            ("total", "under", 145, 142, 3),
            ("prop", "over", 20, 22, 2),
            ("prop", "under", 20, 18, 2),
        ]
        for market, side, entry, end, expected in cases:
            with self.subTest(market=market, side=side):
                self.assertEqual(
                    tracker.clv(
                        {"market": market, "side": side, "line": entry},
                        {market + ":" + side: {"line": end}},
                    ),
                    expected,
                )
        val = tracker.clv(
            {"market": "moneyline", "side": "home", "price": 120},
            {"moneyline:home": {"price": -110}},
        )
        self.assertGreater(val, 0)

    def test_game_push_and_dnp_void(self):
        wager = {"market": "spread", "side": "home", "line": -10, "price": -110}
        tracker.settle(wager, margin=10)
        self.assertEqual((wager["result"], wager["units"]), ("push", 0))
        prop = {"market": "prop", "side": "over", "line": 50, "price": -110}
        tracker.settle(prop, dnp=True)
        self.assertEqual((prop["result"], prop["units"]), ("void", 0))

    def test_corrupt_published_archive_fails_instead_of_resetting(self):
        manifest = {
            "schema": tracker.SCHEMA,
            "weeks": {"nba-week": {"file": "nba-week.json.gz", "sha256": "bad"}},
        }
        import json

        with patch.object(
            tracker, "request", side_effect=[json.dumps(manifest).encode(), b"invalid"]
        ):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                tracker.recover(self.root, "https://example.com/tracking/manifest.json")

    def test_closing_parser_never_substitutes_live_current_line(self):
        item = {
            "provider": {"name": "DraftKings"},
            "spread": -20,
            "homeTeamOdds": {"current": {"pointSpread": {"american": "-20"}}},
        }
        self.assertEqual(tracker.game_quotes(item, phase="close", core=True), {})

    def test_provider_close_accepts_scalar_over_odds_and_explicit_phase_prices(self):
        item = {
            "provider": {"name": "DraftKings"},
            "overOdds": -110.0,
            "close": {
                "total": {"american": "156.5"},
                "over": {"american": "-115"},
                "under": {"american": "-105"},
            },
        }
        quote = tracker.game_quotes(item, phase="close", core=True)
        self.assertEqual(quote["total:over"], {"line": 156.5, "price": -115.0})
        self.assertEqual(quote["total:under"], {"line": 156.5, "price": -105.0})

    def test_missing_final_player_box_remains_pending_then_recovers(self):
        source = board(players=True)
        tracker.run(source, "nba", self.root, now=NOW, provider=Provider(props=True))

        class MissingBox(Provider):
            def collect(self, game):
                result = super().collect(game)
                result["players"] = {}
                return result

        tracker.run(
            source, "nba", self.root, now=NOW + timedelta(days=1), provider=MissingBox(final=True)
        )
        week = self.week()
        prop = next(w for w in week["wagers"].values() if w["market"] == "prop")
        self.assertIsNone(prop["result"])
        self.assertEqual(
            next(iter(week["observations"].values()))["status"], "awaiting_player_results"
        )
        tracker.run(
            source, "nba", self.root, now=NOW + timedelta(days=2), provider=Provider(final=True)
        )
        prop = next(w for w in self.week()["wagers"].values() if w["market"] == "prop")
        self.assertEqual(prop["result"], "win")

    def test_changed_results_do_not_rewrite_settled_wager(self):
        w = {"market": "total", "side": "over", "line": 140, "price": -110}
        tracker.settle(w, total=150)
        tracker.settle(w, total=130)
        self.assertEqual(w["result"], "win")


if __name__ == "__main__":
    unittest.main()
