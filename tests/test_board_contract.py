"""
The shared board contract.

`board.py`, `board.css` and `chase_tokens.css` are vendored BYTE-IDENTICAL into mlb-model,
wnba-edge-model and nfl-model. That is the whole mechanism keeping the three products on
one brand: same card anatomy, same palette, same typefaces.

If this test fails you changed a shared file in one repo only. The fix is to copy it into
all three and regenerate `BOARD_CONTRACT.sha256` in each — not to edit the hash here.

This file is itself vendored; keep the copies identical apart from `_VENDORED`.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]

# The only per-repo line: where this product keeps its vendored copies.
_VENDORED = {
    "board.py": _REPO / "src" / "nflmodel" / "board.py",
    "board.css": _REPO / "src" / "nflmodel" / "static" / "board.css",
    "chase_tokens.css": _REPO / "src" / "nflmodel" / "static" / "chase_tokens.css",
}


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
def test_vendored_file_matches_the_shared_contract(name):
    path = _VENDORED[name]
    assert path.is_file(), f"{name} is missing from this repo"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == _expected()[name], (
        f"{name} has drifted from the shared board contract.\n"
        f"Copy it to mlb-model, wnba-edge-model and nfl-model, then regenerate "
        f"BOARD_CONTRACT.sha256 in all three."
    )


def test_manifest_covers_every_vendored_file():
    assert set(_expected()) == set(_VENDORED)
