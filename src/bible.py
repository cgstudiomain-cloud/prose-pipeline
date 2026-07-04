"""Bible loader: assembles the split bible into the dict shape all
agents already consume. Single place to add per-beat scoping later.

Layout:
  bible/doctrine.json        -> assembled["style"]
  bible/world.json           -> assembled["world"]
  bible/characters/*.json    -> assembled["characters"][<stem>]
  bible/debts.json           -> assembled["debts"]

Falls back to legacy bible/bible.json if the new layout is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

BIBLE_DIR = Path("bible")


def load_bible(root: str | Path = BIBLE_DIR) -> dict:
    root = Path(root)
    if not (root / "doctrine.json").exists():
        legacy = root / "bible.json"
        return json.loads(legacy.read_text(encoding="utf-8")) if legacy.exists() else {}
    out = {
        "style": json.loads((root / "doctrine.json").read_text(encoding="utf-8")),
        "world": json.loads((root / "world.json").read_text(encoding="utf-8"))
        if (root / "world.json").exists() else {},
        "characters": {},
    }
    for p in sorted((root / "characters").glob("*.json")):
        out["characters"][p.stem] = json.loads(p.read_text(encoding="utf-8"))
    if (root / "debts.json").exists():
        out["debts"] = json.loads((root / "debts.json").read_text(encoding="utf-8"))
    return out


def load_naming(root: str | Path = BIBLE_DIR) -> dict[str, str]:
    return load_bible(root).get("style", {}).get("naming", {}).get("canonical", {})
