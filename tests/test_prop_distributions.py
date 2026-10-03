from nflmodel import prop_distributions as pdist


def test_yardage_has_an_ordered_band_and_prices_a_line():
    dist = pdist.distribution("receiving_yards", 60.0)
    assert dist["p10"] < dist["p50"] < dist["p90"]
    assert len(dist["q"]) == len(dist["q_levels"]) == 5
    assert pdist.over_probability(dist, dist["p50"]) == 0.5
    # Tails: certain to clear zero, and thinning (not flat) past the 90th.
    assert pdist.over_probability(dist, 0.0) == 1.0
    beyond = [pdist.over_probability(dist, dist["p90"] + gap) for gap in (0.0, 20.0, 40.0)]
    assert 0.1 >= beyond[0] > beyond[1] > beyond[2] > 0.0


def test_counts_carry_a_pmf_that_sums_to_one():
    dist = pdist.distribution("passing_tds", 1.7)
    assert abs(sum(dist["pmf"].values()) - 1.0) < 0.01
    assert 0.0 < pdist.over_probability(dist, 1.5) < 1.0


def test_unfitted_metrics_are_left_out():
    out = pdist.stats({"receptions": 4.2, "fg_made": 1.8, "anytime_td_probability": 0.3})
    assert set(out) == {"receptions"}
    assert pdist.distribution("receptions", None) is None
