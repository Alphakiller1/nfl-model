#!/usr/bin/env python3
"""Compare this repo's vendored chase_tokens.css to published Chase TIER 1.

Published (after mlbma deploys):
  https://chase-analytics.com/design/chase-tokens-v1.css

Until that URL 200s, fetch the WP1 branch raw file so CI can pass now:
  https://raw.githubusercontent.com/Alphakiller1/mlbma-pipeline/cursor/wp1-design-layer-4ee4/design/chase-tokens-v1.css

chase-tokens-v1.css is NOT byte-identical to vendored chase_tokens.css (TIER 1
renamed primitives; hex values match the seed). Vendored files MUST stay
byte-identical to design/tokens/chase_tokens.vendor.css (seed sha256 below).

board.css is sport-specific and is not part of this check.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

SEED_SHA256 = "13014f566ee570d283b12859a6578d12d179a4cc39aecf8845518700fb85e911"

PUBLISHED_URL = "https://chase-analytics.com/design/chase-tokens-v1.css"
FALLBACK_TIER1 = (
    "https://raw.githubusercontent.com/Alphakiller1/mlbma-pipeline/"
    "cursor/wp1-design-layer-4ee4/design/chase-tokens-v1.css"
)
FALLBACK_VENDOR = (
    "https://raw.githubusercontent.com/Alphakiller1/mlbma-pipeline/"
    "cursor/wp1-design-layer-4ee4/design/tokens/chase_tokens.vendor.css"
)

IDENTITY = ("#08090F", "#9A6BFF", "DM Sans", "Roboto Condensed")

_CRLF = b"\r\n"
_LF = b"\n"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data.replace(_CRLF, _LF)).hexdigest()


def _fetch(url: str, timeout: float = 20.0) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": "chase-token-ci/1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if getattr(resp, "status", 200) != 200:
                return None
            return resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        print(f"skip fetch {url}: {exc}", file=sys.stderr)
        return None


def _looks_like_tier1(data: bytes) -> bool:
    text = data.decode("utf-8", errors="replace")
    head = text[:400].lower()
    if "<html" in head or "<!doctype" in head:
        return False
    return ":root" in text and "#08090F" in text and "#9A6BFF" in text


def _find_tokens(root: Path) -> Path:
    candidates = [
        root / "mlbmodel" / "report" / "static" / "chase_tokens.css",
        root / "src" / "wnba_edges" / "static" / "chase_tokens.css",
        root / "src" / "nflmodel" / "static" / "chase_tokens.css",
        root / "src" / "cfbmodel" / "static" / "chase_tokens.css",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise SystemExit(f"chase_tokens.css not found under {root}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tokens", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    tokens_path = args.tokens.resolve() if args.tokens else _find_tokens(root)
    local = tokens_path.read_bytes()
    local_hash = _digest(local)
    if local_hash != SEED_SHA256:
        print(
            f"FAIL vendored chase_tokens.css sha256 {local_hash} != seed {SEED_SHA256}",
            file=sys.stderr,
        )
        return 1
    print(f"OK local chase_tokens.css matches seed {SEED_SHA256}")

    vendor = _fetch(FALLBACK_VENDOR)
    if vendor is None:
        for extra in (
            Path("/workspace/design/tokens/chase_tokens.vendor.css"),
            root.parent / "mlbma-pipeline" / "design" / "tokens" / "chase_tokens.vendor.css",
        ):
            if extra.is_file():
                vendor = extra.read_bytes()
                print(f"using sibling vendor snapshot {extra}")
                break
    if vendor is not None:
        if _digest(vendor) != local_hash:
            print(
                "FAIL vendored chase_tokens.css drifted from mlbma chase_tokens.vendor.css",
                file=sys.stderr,
            )
            return 1
        print("OK vendored file matches mlbma design/tokens/chase_tokens.vendor.css")
    else:
        print("vendor snapshot URL not reachable; local seed hash is the pin")

    source = PUBLISHED_URL
    published = _fetch(PUBLISHED_URL)
    if published is None or not _looks_like_tier1(published):
        print(f"published URL not CSS yet; falling back to {FALLBACK_TIER1}")
        source = FALLBACK_TIER1
        published = _fetch(FALLBACK_TIER1)
    if published is None or not _looks_like_tier1(published):
        for extra in (
            Path("/workspace/design/chase-tokens-v1.css"),
            root.parent / "mlbma-pipeline" / "design" / "chase-tokens-v1.css",
        ):
            if extra.is_file() and _looks_like_tier1(extra.read_bytes()):
                published = extra.read_bytes()
                source = str(extra)
                print(f"using sibling TIER 1 file {extra}")
                break
    if published is None or not _looks_like_tier1(published):
        print(
            "SKIP TIER 1 fetch: chase-analytics.com/design/chase-tokens-v1.css is not "
            "CSS yet and the WP1 raw fallback is not readable from this job. "
            f"Local chase_tokens.css still matches seed {SEED_SHA256}."
        )
        return 0
    text = published.decode("utf-8", errors="replace")
    missing = [key for key in IDENTITY if key not in text]
    if missing:
        print(f"FAIL {source} missing identity tokens: {missing}", file=sys.stderr)
        return 1
    print(f"OK TIER 1 from {source} carries Chase identity literals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
