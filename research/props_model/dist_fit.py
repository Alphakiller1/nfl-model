"""Fit outcome distributions on the model's own errors (played rows), check calibration."""
import json, math, statistics as st, sys
from collections import defaultdict
import numpy as np
LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90)
LADDER = ["passing_yards", "pass_attempts", "completions", "rushing_yards", "carries", "rush_attempts",
          "targets", "receiving_yards"]
COUNTS = ["receptions", "passing_tds", "interceptions"]

def rows_of(path, weeks=None):
    out = []
    for r in json.load(open(path)):
        if not r["played"] or (weeks and r["week"] not in weeks):
            continue
        for m, mu in r["m"].items():
            if m in LADDER + COUNTS and mu is not None and mu > 0.05:
                out.append((m, float(mu), float(r["a"][m]), r))
    return out

def fit_ladders(rows, nb=4):
    lad = {}
    for m in LADDER:
        xs = sorted((mu, a / mu) for mm, mu, a, _ in rows if mm == m)
        if len(xs) < 200: continue
        edges = [xs[int(len(xs) * q / nb)][0] for q in range(1, nb)]
        buckets = []
        for i in range(nb):
            lo = edges[i - 1] if i else -1; hi = edges[i] if i < nb - 1 else 1e9
            rat = np.array([r for mu, r in xs if lo < mu <= hi])
            buckets.append((round(hi, 2) if hi < 1e9 else 1e9, tuple(round(float(np.quantile(rat, q)), 3) for q in LEVELS)))
        lad[m] = buckets
    return lad

def fit_nb(rows):
    out = {}
    for m in COUNTS:
        xs = [(mu, a) for mm, mu, a, _ in rows if mm == m]
        best = None
        for size in (None, 1, 2, 3, 4, 6, 8, 12, 20, 40):
            ll = 0
            for mu, a in xs:
                k = int(round(a))
                if size is None: ll += -mu + k * math.log(mu) - math.lgamma(k + 1)
                else:
                    p = size / (size + mu); ll += math.lgamma(k + size) - math.lgamma(size) - math.lgamma(k + 1) + size * math.log(p) + k * math.log(1 - p)
            if best is None or ll > best[0]: best = (ll, size)
        out[m] = best[1]
    return out

def p_over(m, mu, line, lad, nb):
    if m in nb:
        size = nb[m]; tot = 0.0
        for k in range(0, 80):
            if size is None: mass = math.exp(-mu + k * math.log(mu) - math.lgamma(k + 1))
            else:
                p = size / (size + mu); mass = math.exp(math.lgamma(k + size) - math.lgamma(size) - math.lgamma(k + 1) + size * math.log(p) + k * math.log(1 - p))
            if k > line: tot += mass
        return tot
    b = next(q for hi, q in lad[m] if mu <= hi)
    lad_x = [mu * r for r in b]
    if line < lad_x[0]: return 1 - LEVELS[0] * line / max(lad_x[0], 1e-9) if lad_x[0] > 0 else 0.9
    if line >= lad_x[-1]: return 0.10 * math.exp(-(line - lad_x[-1]) / max(lad_x[-1] - lad_x[2], 1))
    for (x0, p0), (x1, p1) in zip(zip(lad_x, LEVELS), zip(lad_x[1:], LEVELS[1:])):
        if x0 <= line < x1:
            return 1 - (p0 + (p1 - p0) * (line - x0) / max(x1 - x0, 1e-9))
    return 0.5

if __name__ == "__main__":
    tr = rows_of("rm2_2025.json", set(range(2, 11))); te = rows_of("rm2_2025.json", set(range(11, 19)))
    lad, nb = fit_ladders(tr), fit_nb(tr)
    print("NB sizes", nb)
    # PIT coverage on held-out half
    for m in LADDER:
        xs = [(mu, a) for mm, mu, a, _ in te if mm == m]
        if not xs or m not in lad: continue
        cov = np.mean([lo <= a <= hi for mu, a in xs for b in [next(q for h, q in lad[m] if mu <= h)] for lo, hi in [(mu * b[0], mu * b[-1])]])
        med = np.mean([a > mu * next(q for h, q in lad[m] if mu <= h)[2] for mu, a in xs])
        print(f"  {m:16s} n={len(xs)} 10-90 coverage {cov:.3f} (target .80)  share above p50 {med:.3f}")
    full = rows_of("rm2_2025.json")
    json.dump({"ladders": fit_ladders(full), "nb": fit_nb(full)}, open("dist_2025.json", "w"))
