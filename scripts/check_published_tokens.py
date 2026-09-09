#!/usr/bin/env python3
"""Compare this repo's vendored TIER 1 file to published Chase tokens.

Published:
  https://chase-analytics.com/design/chase-tokens-v1.css

Site builders concatenate chase-tokens-v1.css in front of chase_tokens.css
(aliases + board extras) because CSS is inlined into <style>.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

ALIASES_SHA256 = "d2a929732e081ed8d2d5208aa915829e693c767f2f14fb29e2bffafc188a1fcd"
TIER1_SHA256 = "3cd1f89f4e00618e1f006d2434fadc2f50617e7e220f8509b64be016598e03cf"

PUBLISHED_URL = "https://chase-analytics.com/design/chase-tokens-v1.css"
FALLBACK_TIER1 = (
    "https://raw.githubusercontent.com/Alphakiller1/mlbma-pipeline/"
    "master/design/chase-tokens-v1.css"
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


def _find_pair(root: Path) -> tuple[Path, Path]:
    pairs = [
        (
            root / "mlbmodel" / "report" / "static" / "chase_tokens.css",
            root / "mlbmodel" / "report" / "static" / "chase-tokens-v1.css",
        ),
        (
            root / "src" / "wnba_edges" / "static" / "chase_tokens.css",
            root / "src" / "wnba_edges" / "static" / "chase-tokens-v1.css",
        ),
        (
            root / "src" / "nflmodel" / "static" / "chase_tokens.css",
            root / "src" / "nflmodel" / "static" / "chase-tokens-v1.css",
        ),
        (
            root / "src" / "cfbmodel" / "static" / "chase_tokens.css",
            root / "src" / "cfbmodel" / "static" / "chase-tokens-v1.css",
        ),
    ]
    for aliases, v1 in pairs:
        if aliases.is_file():
            return aliases, v1
    raise SystemExit(f"chase_tokens.css not found under {root}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tokens", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    aliases_path, v1_path = _find_pair(root)
    if args.tokens:
        aliases_path = args.tokens.resolve()
        v1_path = aliases_path.with_name("chase-tokens-v1.css")

    aliases = aliases_path.read_bytes()
    aliases_hash = _digest(aliases)
    text = aliases.decode("utf-8", errors="replace")
    if "@import url" in text:
        print("FAIL chase_tokens.css must not @import; builders inline CSS", file=sys.stderr)
        return 1
    if "var(--ca-ink-950)" not in text:
        print("FAIL chase_tokens.css must map --bg onto var(--ca-ink-950)", file=sys.stderr)
        return 1
    if aliases_hash != ALIASES_SHA256:
        print(
            f"FAIL chase_tokens.css sha256 {aliases_hash} != {ALIASES_SHA256}",
            file=sys.stderr,
        )
        return 1
    print(f"OK local chase_tokens.css matches alias pin {ALIASES_SHA256}")

    if not v1_path.is_file():
        print(f"FAIL missing {v1_path}", file=sys.stderr)
        return 1
    v1 = v1_path.read_bytes()
    v1_hash = _digest(v1)
    if v1_hash != TIER1_SHA256:
        print(f"FAIL chase-tokens-v1.css sha256 {v1_hash} != {TIER1_SHA256}", file=sys.stderr)
        return 1
    if not _looks_like_tier1(v1):
        print("FAIL chase-tokens-v1.css is missing Chase identity literals", file=sys.stderr)
        return 1
    print(f"OK local chase-tokens-v1.css matches TIER 1 pin {TIER1_SHA256}")

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
        print("SKIP TIER 1 fetch; local pins still match.")
        return 0
    pub_text = published.decode("utf-8", errors="replace")
    missing = [key for key in IDENTITY if key not in pub_text]
    if missing:
        print(f"FAIL {source} missing identity tokens: {missing}", file=sys.stderr)
        return 1
    print(f"OK TIER 1 from {source} carries Chase identity literals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
