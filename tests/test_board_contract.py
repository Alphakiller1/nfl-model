"""Local integrity pins for board.py / board.css, plus the shared token seed.

`chase_tokens.css` is the only file that is byte-identical across sport-model
repos (seed sha256 13014f56…). `board.css` is sport-specific. `board.py` copies
may diverge. `BOARD_CONTRACT.sha256` pins this checkout; it is not a claim that
board.css is identical across MLB / WNBA / NFL / CFB.

Hashes are over LF-normalised bytes so Windows CRLF checkouts match CI.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]

_VENDORED = {
    "board.py": _REPO / "src" / "nflmodel" / "board.py",
    "board.css": _REPO / "src" / "nflmodel" / "static" / "board.css",
    "chase_tokens.css": _REPO / "src" / "nflmodel" / "static" / "chase_tokens.css",
    "chase-tokens-v1.css": _REPO / "src" / "nflmodel" / "static" / "chase-tokens-v1.css",
}

SHARED_TOKENS_SHA256 = "d2a929732e081ed8d2d5208aa915829e693c767f2f14fb29e2bffafc188a1fcd"

_CRLF = b"\r\n"
_LF = b"\n"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(_CRLF, _LF)).hexdigest()


def _expected() -> dict[str, str]:
    manifest = (_REPO / "BOARD_CONTRACT.sha256").read_text(encoding="utf-8")
    out = {}
    for line in manifest.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, name = line.split()
        out[name] = digest
    return out


@pytest.mark.parametrize("name", sorted(_VENDORED))
def test_vendored_file_matches_the_local_pin(name):
    path = _VENDORED[name]
    assert path.is_file(), f"{name} is missing from this repo"
    actual = _digest(path)
    assert actual == _expected()[name], (
        f"{name} has drifted from BOARD_CONTRACT.sha256.\n"
        f"If this is an intentional local board.css / board.py edit, regenerate the "
        f"manifest. chase_tokens.css must remain the shared seed {SHARED_TOKENS_SHA256}."
    )


def test_manifest_covers_every_vendored_file():
    assert set(_expected()) == set(_VENDORED)


def test_chase_tokens_is_the_cross_sport_seed():
    assert _digest(_VENDORED["chase_tokens.css"]) == SHARED_TOKENS_SHA256
