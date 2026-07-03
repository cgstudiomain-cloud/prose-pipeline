"""Writer agent. Two modes:

Draft mode: BeatSpec + ContextPacket -> prose draft (v1).
    uv run python -m src.agents.writer <run_dir>              # all beats
    uv run python -m src.agents.writer <run_dir> <beat_id>    # one beat

Revision mode: draft vN + PatchDocument -> draft vN+1.
    Called by the orchestrator once editors exist. The model returns
    replacement text per patch; code splices it in. Text outside the
    patched spans cannot change, by construction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from src.agents.director import load_bible, load_guidelines
from src.client import ModelClient, load_config
from src.schemas import BeatSpec, ContextPacket, Draft, PatchDocument

NOTES_MARKER = "=== WRITER NOTES ==="


# ---------------------------------------------------------- draft mode

DRAFT_SYSTEM = """\
You are the Writer in a prose-production pipeline. You turn one beat
specification into finished prose. You will not see the rest of the
chapter; the spec and context packet are your entire world.

Rules:
- Execute the spec's movements in order, as ONE continuous scene. The
  movements are scaffolding and must be invisible in the output: no
  separators, no headers, no section breaks between them. Bridge
  through motion, sound, and physical continuity.
- A movement's content is a SUMMARY to dramatize at full resolution,
  never sentences to transcribe. word_budget is camera time to SPEND:
  landing more than ~15% under budget means the moment was rushed.
  Expand through observable texture - micro-action, sound, light, the
  staging elements - never through interpretation.
- Phase discipline: anticipation builds the question, action is
  delivered lean and short, absorption HOLDS - reactions, stillness,
  silence - and is never cut early.
- Sentence rhythm carries the phase: anticipation runs in longer,
  connected sentences; action snaps short; absorption settles into
  medium, even sentences. Never default to staccato throughout.
- Serve the riddle: the question must be posed by its evidence before
  the reveal lands. Never pre-label the reveal; never explain it after.
- Use the staging elements given in each movement. Do not add staging
  of your own unless it has a clear function; texture without function
  is cut.
- Stay strictly inside the events of skeleton_excerpt. Movements and
  staging tell you HOW to shoot the scene, never license new plot or
  new dialogue that changes what happens.
- dialogue_anchors with keep_verbatim=true must appear word for word.
- Respect character_voices from the context packet in every line of
  dialogue.
- Never contradict must_remain_true. Never touch anything in forbidden.
- Land inside the target word range.
- Match the audience and register given in the style guide.
- Before output, run the camera test from the guidelines on every
  narration sentence: any mind-reading, judgment, personification,
  metaphor, or speculation gets replaced with the observable fact.

Output format: the prose only, as clean markdown paragraphs. Then, on
its own line, the marker === WRITER NOTES === followed by 2-5 short
bullet points on choices you made that an editor should know about.
"""


def build_draft_prompt(spec: BeatSpec, packet: ContextPacket, bible: dict, guidelines: str) -> str:
    return f"""\
GUIDELINE PACK:
{guidelines}

STYLE (from story bible):
{json.dumps(bible.get("style", {}), indent=2, ensure_ascii=False)}

BEAT SPEC:
{spec.model_dump_json(indent=2)}

CONTEXT PACKET:
{packet.model_dump_json(indent=2)}

Write the beat.
"""


def split_prose_and_notes(text: str) -> tuple[str, str]:
    if NOTES_MARKER in text:
        prose, notes = text.split(NOTES_MARKER, 1)
        return prose.strip(), notes.strip()
    return text.strip(), ""


def draft_beat(client: ModelClient, run_dir: Path, beat_id: str) -> Draft:
    beat_dir = run_dir / "beats" / beat_id
    spec = BeatSpec.model_validate_json(
        (beat_dir / "01_beat_spec.json").read_text(encoding="utf-8")
    )
    packet = ContextPacket.model_validate_json(
        (beat_dir / "02_context_packet.json").read_text(encoding="utf-8")
    )
    raw = client.call_text(
        agent="writer",
        system=DRAFT_SYSTEM,
        user=build_draft_prompt(spec, packet, load_bible(), load_guidelines()),
    )
    prose, notes = split_prose_and_notes(raw)
    draft = Draft(beat_id=beat_id, version=1, prose=prose, writer_notes=notes)
    save_draft(beat_dir, draft)
    words = len(prose.split())
    print(f"writer: {beat_id} v1 drafted ({words} words)")
    return draft


def save_draft(beat_dir: Path, draft: Draft) -> None:
    (beat_dir / f"03_draft_v{draft.version}.md").write_text(
        draft.prose, encoding="utf-8"
    )
    (beat_dir / f"03_draft_v{draft.version}.json").write_text(
        draft.model_dump_json(indent=2), encoding="utf-8"
    )


# ------------------------------------------------------- revision mode

REVISE_SYSTEM = """\
You are the Writer revising your own draft according to the Chief
Editor's patch document. For each patch, produce replacement text for
the anchored span - and nothing else. The replacement should read
seamlessly against the untouched text around it.

Honor the patch instruction exactly. Preserve the established voice
and register. If a patch instruction conflicts with a keep_verbatim
dialogue anchor, the anchor wins: adjust around it.
"""


class SpanReplacement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patch_id: str
    anchor_text: str
    replacement_text: str


class RevisionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    replacements: list[SpanReplacement]


def build_revision_prompt(draft: Draft, patch_doc: PatchDocument, spec: BeatSpec) -> str:
    return f"""\
BEAT SPEC (for reference):
{spec.model_dump_json(indent=2)}

CURRENT DRAFT (v{draft.version}):
{draft.prose}

PATCH DOCUMENT:
{patch_doc.model_dump_json(indent=2)}

Return one replacement per patch, keyed by patch_id, echoing each
patch's anchor_text exactly as it appears in the draft.
"""


def apply_replacements(prose: str, replacements: list[SpanReplacement]) -> tuple[str, list[str]]:
    """Splice replacements into prose by anchor. Returns (new_prose,
    list of patch_ids whose anchors were not found)."""
    missed = []
    for rep in replacements:
        if rep.anchor_text in prose:
            prose = prose.replace(rep.anchor_text, rep.replacement_text, 1)
        else:
            missed.append(rep.patch_id)
    return prose, missed


def revise_beat(client: ModelClient, run_dir: Path, beat_id: str, patch_doc: PatchDocument) -> Draft:
    beat_dir = run_dir / "beats" / beat_id
    spec = BeatSpec.model_validate_json(
        (beat_dir / "01_beat_spec.json").read_text(encoding="utf-8")
    )
    draft = Draft.model_validate_json(
        (beat_dir / f"03_draft_v{patch_doc.draft_version}.json").read_text(encoding="utf-8")
    )
    out = client.call_structured(
        agent="writer",
        system=REVISE_SYSTEM,
        user=build_revision_prompt(draft, patch_doc, spec),
        schema=RevisionOutput,
    )
    new_prose, missed = apply_replacements(draft.prose, out.replacements)
    if missed:
        print(f"writer: WARNING - anchors not found for patches {missed} "
              f"(logged, skipped)")
    new_draft = Draft(
        beat_id=beat_id,
        version=draft.version + 1,
        prose=new_prose,
        writer_notes=f"revision of v{draft.version}; "
                     f"applied {len(out.replacements) - len(missed)} patches"
                     + (f"; missed anchors: {missed}" if missed else ""),
    )
    save_draft(beat_dir, new_draft)
    print(f"writer: {beat_id} v{new_draft.version} spliced "
          f"({len(out.replacements) - len(missed)}/{len(out.replacements)} patches applied)")
    return new_draft


# --------------------------------------------------------------- CLI


def main() -> None:
    if len(sys.argv) not in (2, 3):
        sys.exit("usage: uv run python -m src.agents.writer <run_dir> [beat_id]")
    run_dir = Path(sys.argv[1])
    client = ModelClient(load_config(), run_dir)
    if len(sys.argv) == 3:
        beat_ids = [sys.argv[2]]
    else:
        beat_ids = sorted(p.name for p in (run_dir / "beats").iterdir() if p.is_dir())
    for beat_id in beat_ids:
        draft_beat(client, run_dir, beat_id)


if __name__ == "__main__":
    main()
