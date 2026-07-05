"""Spark: fill the middle of a chain.

A map file has a seed, a START, an END, and beads - numbered lines in
the author's own register. Open slots are marked ???. The spark pass
proposes beads for the middle: one line each, woven in as {? ...}
lines. Take what hooks, delete the rest, rerun. Every proposal ever
made is logged next to the map and never re-proposed.

    uv run python -m src.agents.spark maps\\ch02_the_cell.txt

Map format (everything except START/END optional):

    === CHAPTER: ch02_the_cell ===
    {actors: Mila, Cat, Fox(arrives); tone: gag; speed: slow after
     ch01 sprint; job: trio reunited at the bottom}
    START: thrown in prison - Mila and cat locked in a cell
    1. Mila and cat talk, it'll be fun, she said.
    2. ???
    3. how are we getting out of here? and where's fox?
    END: fox tossed into the cell across - trio at the bottom together
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.agents.director import _flex_pattern, new_run_dir
from src.bible import load_bible
from src.client import ModelClient, load_config


class SparkBead(BaseModel):
    model_config = ConfigDict(extra="forbid")

    after_anchor: str = Field(
        ..., description="Exact quote (<=12 words) of the map line this bead follows. "
                         "Use 'START' to place right after the START line."
    )
    bead: str = Field(..., description="One line, event register, no braces")


class SparkOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    beads: list[SparkBead]


SPARK_SYSTEM = """\
You fill the middle of a chain of events. START and END are fixed
walls - never propose changing them. Plain numbered lines are the
author's beads: locked, untouchable; build around and between them.
Lines marked ??? are open slots asking to be filled. If there are no
??? and the middle is empty, propose about five beads bridging START
to END.

Each proposal is ONE line in the author's own register - terse,
event-first, plain (study the locked beads and match them). The chain
law: every bead answers a question the previous bead posed and poses
the next one. Know where you start, know where you finish; the bead's
job is to pull.

Fitness - this is a middle-grade adventure: FUN AND FORWARD. Prefer:
- character free-money: beads that spend traits the bible already owns
  (the trickster picks locks; the overbearing sibling overprotects)
- reversals and paid setups over new machinery
- tonal contrast with the neighbors, honoring the seed's tone/speed
- beads that touch the debts ledger when natural (flag with a short
  parenthetical when you do)

Self-limit: propose the SETUP, never the punchline. Situations,
reversals, arrivals, discoveries - yes. Jokes' wordings, character-
defining lines, dialogue - the author's pen; leave room, don't fill it.

Respect {seed} braces (actors, tone, speed, job) and any {} intent as
author truth. NEVER propose anything on the DO NOT REPROPOSE list, or
a trivial rewording of it.
"""


def load_graveyard(map_path: Path) -> list[str]:
    log = map_path.with_suffix(map_path.suffix + ".sparks.log")
    if not log.exists():
        return []
    return [ln.strip() for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]


def append_graveyard(map_path: Path, beads: list[str]) -> None:
    log = map_path.with_suffix(map_path.suffix + ".sparks.log")
    with open(log, "a", encoding="utf-8") as f:
        for b in beads:
            f.write(b.strip() + "\n")


def weave_sparks(text: str, beads: list[SparkBead]) -> tuple[str, list[SparkBead]]:
    """Insert each bead as a standalone '{? ...}' line after its anchor
    line. Never alters existing lines."""
    placed, unplaced = [], []
    for b in beads:
        needle = b.after_anchor.strip()
        if needle.upper() == "START":
            m = _flex_pattern("START").search(text)
        else:
            m = _flex_pattern(needle).search(text)
        if m is None:
            unplaced.append(b)
            continue
        line_end = text.find("\n", m.end())
        line_end = len(text) if line_end == -1 else line_end
        placed.append((line_end, b))
    placed.sort(key=lambda t: t[0])

    pieces, cursor = [], 0
    for pos, b in placed:
        pieces.append(text[cursor:pos])
        pieces.append("\n{? " + b.bead.replace("{", "").replace("}", "").strip() + "}")
        cursor = pos
    pieces.append(text[cursor:])
    return "".join(pieces), unplaced


def run_spark(map_path: str | Path) -> Path:
    map_path = Path(map_path)
    text = map_path.read_text(encoding="utf-8")
    graveyard = load_graveyard(map_path)
    bible = load_bible()

    run_dir = new_run_dir(label=f"spark_{map_path.stem}")
    client = ModelClient(load_config(), run_dir)
    print(f"spark: filling the middle of {map_path.name} (1 call)...")

    grave_block = ("\n\nDO NOT REPROPOSE (already offered, author tossed or kept):\n"
                   + "\n".join(f"- {g}" for g in graveyard)) if graveyard else ""
    out = client.call_structured(
        agent="spark",
        system=SPARK_SYSTEM,
        user=(f"STORY BIBLE (characters, world, debts):\n"
              f"{json.dumps(bible, indent=2, ensure_ascii=False)}\n\n"
              f"MAP:\n{text}{grave_block}"),
        schema=SparkOutput,
    )
    annotated, unplaced = weave_sparks(text, out.beads)
    append_graveyard(map_path, [b.bead for b in out.beads])

    out_path = run_dir / f"00_{map_path.stem}_sparked.md"
    body = annotated
    if unplaced:
        body += "\n\n⟦UNPLACED⟧\n" + "\n".join(
            f"after {b.after_anchor!r}: {b.bead}" for b in unplaced) + "\n⟦/UNPLACED⟧"
    out_path.write_text(body, encoding="utf-8")
    print(f"spark -> {out_path}  ({len(out.beads) - len(unplaced)} beads proposed, "
          f"{len(unplaced)} unplaced, graveyard now {len(graveyard) + len(out.beads)})")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: uv run python -m src.agents.spark <map_file>")
    run_spark(sys.argv[1])
