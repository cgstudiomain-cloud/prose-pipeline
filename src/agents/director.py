"""Director agent.

Input:  a skeleton file with beat markers, the story bible, the
        guideline pack.
Output: one BeatSpec + one ContextPacket per beat, written to
        <run_dir>/beats/<beat_id>/.

Run from the project root:
    uv run python -m src.agents.director skeletons\\stealing_the_sword_marked.txt
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.client import ModelClient, load_config, new_run_dir
from src.schemas import BeatSpec, ContextPacket

BEAT_MARKER = re.compile(r"^===\s*BEAT:\s*(?P<beat_id>[\w-]+)\s*===\s*$", re.MULTILINE)


class DirectorOutput(BaseModel):
    """Wrapper so a single structured call returns everything."""

    model_config = ConfigDict(extra="forbid")

    beats: list[BeatSpec]
    packets: list[ContextPacket] = Field(
        ..., description="Exactly one packet per beat, same beat_ids"
    )


# ------------------------------------------------------------- inputs


def split_beats(skeleton_text: str) -> list[tuple[str, str]]:
    """Return [(beat_id, excerpt), ...] from a marked skeleton."""
    matches = list(BEAT_MARKER.finditer(skeleton_text))
    if not matches:
        raise ValueError(
            "No beat markers found. Add lines like '=== BEAT: 01_name ===' "
            "to the skeleton."
        )
    beats = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(skeleton_text)
        excerpt = skeleton_text[start:end].strip()
        if not excerpt:
            raise ValueError(f"Beat '{m['beat_id']}' has no skeleton text under it.")
        beats.append((m["beat_id"], excerpt))
    return beats


def load_guidelines(folder: str | Path = "guidelines") -> str:
    parts = []
    for p in sorted(Path(folder).glob("*")):
        if p.is_file():
            parts.append(f"--- {p.name} ---\n{p.read_text(encoding='utf-8')}")
    return "\n\n".join(parts) if parts else "(no guidelines provided)"


def load_bible(path: str | Path = "bible/bible.json") -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ------------------------------------------------------------- prompts

SYSTEM = """\
You are the Director in a prose-production pipeline. You do not write
prose. You produce precise, complete beat specifications that a Writer
who has never seen the rest of the story can execute in isolation.

Principles:
- entering_state and exiting_state must chain: beat N's exiting_state
  is beat N+1's entering_state. No gaps, no contradictions.
- dialogue_anchors: pull the lines from the skeleton that carry the
  scene. Mark keep_verbatim=true for lines whose exact wording matters.
- directing_notes follow the flashlight/wave system from the guidelines:
  each note names what the reader's attention is pointed at (focus) and
  how long it holds there and why (hold).
- ContextPackets exist so beats can be written in parallel by writers
  who cannot see each other's work. reader_knows, must_remain_true and
  forbidden must be complete enough to prevent contradictions.
- scene_goal is about story function, not plot summary: what must this
  beat change in the reader's understanding or the characters' standing.
- Stay strictly inside the skeleton's events. Invent texture, never plot.
"""


def build_user_prompt(skeleton_text: str, beats: list[tuple[str, str]], bible: dict, guidelines: str) -> str:
    beat_list = "\n".join(f"- {bid}" for bid, _ in beats)
    return f"""\
STORY BIBLE:
{json.dumps(bible, indent=2, ensure_ascii=False)}

GUIDELINE PACK:
{guidelines}

FULL SKELETON (beat markers included):
{skeleton_text}

Produce one BeatSpec and one ContextPacket for each of these beat_ids,
in this order:
{beat_list}

Use each beat's own skeleton text as its skeleton_excerpt, verbatim.
"""


# ------------------------------------------------------------ pipeline


def verify(output: DirectorOutput, expected_ids: list[str]) -> None:
    """Cheap structural checks before we trust the output."""
    spec_ids = [b.beat_id for b in output.beats]
    packet_ids = [p.beat_id for p in output.packets]
    if spec_ids != expected_ids:
        raise ValueError(f"BeatSpec ids {spec_ids} != expected {expected_ids}")
    if sorted(packet_ids) != sorted(expected_ids):
        raise ValueError(f"ContextPacket ids {packet_ids} != expected {expected_ids}")


def run_director(skeleton_path: str | Path, run_dir: Path | None = None) -> Path:
    skeleton_text = Path(skeleton_path).read_text(encoding="utf-8")
    beats = split_beats(skeleton_text)
    expected_ids = [bid for bid, _ in beats]

    config = load_config()
    run_dir = run_dir or new_run_dir(label=Path(skeleton_path).stem)
    client = ModelClient(config, run_dir)

    output = client.call_structured(
        agent="director",
        system=SYSTEM,
        user=build_user_prompt(skeleton_text, beats, load_bible(), load_guidelines()),
        schema=DirectorOutput,
    )
    verify(output, expected_ids)

    packets = {p.beat_id: p for p in output.packets}
    for spec in output.beats:
        beat_dir = run_dir / "beats" / spec.beat_id
        beat_dir.mkdir(parents=True, exist_ok=True)
        (beat_dir / "01_beat_spec.json").write_text(
            spec.model_dump_json(indent=2), encoding="utf-8"
        )
        (beat_dir / "02_context_packet.json").write_text(
            packets[spec.beat_id].model_dump_json(indent=2), encoding="utf-8"
        )
    print(f"director: {len(output.beats)} beats specced -> {run_dir / 'beats'}")
    return run_dir


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: uv run python -m src.agents.director <skeleton_file>")
    run_director(sys.argv[1])
