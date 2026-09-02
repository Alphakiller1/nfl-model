from nflmodel.sources import nflverse


def test_failed_refresh_surfaces_bounded_snapshot_status(tmp_path, monkeypatch):
    monkeypatch.setattr(nflverse, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(nflverse, "_download", lambda url: b"a,b\n1,2\n")
    nflverse.clear_run_state()
    assert nflverse.fetch_csv("https://example.test/data", "data.csv", ttl=0)

    def fail(url):
        raise nflverse.NflverseError("provider unavailable")

    monkeypatch.setattr(nflverse, "_download", fail)
    assert nflverse.fetch_csv("https://example.test/data", "data.csv", ttl=0)
    status = nflverse.status_report()[0]
    assert status["state"] == "bounded_snapshot"
    assert status["stale"] is True
    assert "provider unavailable" in status["error"]
