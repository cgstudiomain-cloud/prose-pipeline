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


def latest_version(beat_dir: Path) -> int:
    versions = [int(p.stem.split("_v")[1]) for p in beat_dir.glob("03_draft_v*.json")]
    return max(versions, default=0)


def load_latest_draft(beat_dir: Path) -> Draft:
    v = latest_version(beat_dir)
    if v == 0:
        raise FileNotFoundError(f"no drafts in {beat_dir}")
    return Draft.model_validate_json(
        (beat_dir / f"03_draft_v{v}.json").read_text(encoding="utf-8")
    )


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
- Execute each movement's tempo. SLOW: the camera lingers - long,
  connected sentences, micro-detail, full paragraphs; let the moment
  breathe. FAST: boom - one event per sentence, one or two sentences
  per paragraph, bare dialogue without tags where speakers are clear,
  white space as speed, staging at the minimum. MEDIUM: standard.
  The contour between movements must be audible: a fast movement after
  a slow one should read like a gear change.
- Within a movement, sentence rhythm still follows the phase where the
  tempo allows: anticipation connects, action snaps, absorption
  settles. Never default to one rhythm throughout.
- Never release the reader. Each movement's handoff names the
  unresolved thing that must still be pulling when the next movement
  begins - end the passage with that hook live, not closed. Resolve a
  question only while another is open or opening. Full release happens
  only at the chapter's final beat.
- Narration is a chain, not a list: every sentence is pulled onto the
  page by the previous one - cause then effect, stimulus then
  response, attention then object. A character stops BECAUSE of a
  sound: the sound comes first. Objects enter when attention lands on
  them, never as advance decor. If two adjacent sentences could be
  swapped without loss, rewrite until order is load-bearing.
- Serve the riddle: the question must be posed by its evidence before
  the reveal lands. Never pre-label the reveal; never explain it after.
- Use the staging elements given in each movement. Do not add staging
  of your own unless it has a clear function; texture without function
  is cut.
- Text inside {braces} anywhere in the spec is blueprint intent:
  obey it, never transcribe its wording into prose.
- Staging FUNCTIONS are the Director's reasoning, never yours to
  print: do not surface a function, a plant, or the connection it
  documents in narration. If the evidence is staged, the reader makes
  the connection - the narrator never does.
- Physical truth: no dialogue during maximal effort. Set, pull, fail,
  release, exhale - speech comes after the release. Pose unknowns
  affirmatively ("Fox was gripping something"), never as negated
  perception. Vary the wording of repeated facts.
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
    draft = Draft(beat_id=beat_id, version=latest_version(beat_dir) + 1,
                  prose=prose, writer_notes=notes)
    save_draft(beat_dir, draft)
    words = len(prose.split())
    print(f"writer: {beat_id} v{draft.version} drafted ({words} words)")
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


# ------------------------------------------------- human round-trip

FINISH_SYSTEM = """\
You are the Writer continuing a beat that a human author has partially
written. The human text is final and untouchable: you will not repeat,
restate, or alter one word of it. Read the spec, work out which
movements the human text already covers, and continue seamlessly from
its last sentence through the remaining movements - matching the
established rhythm, voice, and all guideline rules (camera test, chain,
tempo, handoffs).

Output ONLY the continuation, starting mid-scene where the human text
stops. Then, on its own line, === WRITER NOTES === and 2-4 short
bullets on the choices made.
"""

POLISH_SYSTEM = """\
You are the Writer revising your draft according to a human author's
free-form instruction. Produce span replacements: for each place the
instruction touches, echo the exact anchor_text from the draft and
supply replacement text. Touch the minimum number of spans that
honestly implements the instruction; everything else stays untouched.
All guideline rules apply to the replacement text. keep_verbatim
dialogue anchors win over the instruction - adjust around them.
Use patch_id values h1, h2, ...
"""


def ingest_beat(run_dir: Path, beat_id: str) -> Draft:
    """Promote a human-edited .md of the latest version to a new draft version."""
    beat_dir = run_dir / "beats" / beat_id
    current = load_latest_draft(beat_dir)
    md_text = (beat_dir / f"03_draft_v{current.version}.md").read_text(encoding="utf-8").strip()
    if md_text == current.prose.strip():
        print(f"writer: {beat_id} v{current.version} .md matches the record - nothing to ingest")
        return current
    draft = Draft(beat_id=beat_id, version=current.version + 1,
                  prose=md_text, writer_notes="human edit ingested")
    save_draft(beat_dir, draft)
    print(f"writer: {beat_id} human edit ingested as v{draft.version}")
    return draft


def finish_beat(client: ModelClient, run_dir: Path, beat_id: str) -> Draft:
    """Complete a partially human-written beat. Human text is preserved
    verbatim as the prefix, by construction."""
    beat_dir = run_dir / "beats" / beat_id
    current = load_latest_draft(beat_dir)
    md_text = (beat_dir / f"03_draft_v{current.version}.md").read_text(encoding="utf-8").strip()
    spec = BeatSpec.model_validate_json((beat_dir / "01_beat_spec.json").read_text(encoding="utf-8"))
    packet = ContextPacket.model_validate_json((beat_dir / "02_context_packet.json").read_text(encoding="utf-8"))
    user = (build_draft_prompt(spec, packet, load_bible(), load_guidelines())
            + f"\nTHE BEAT SO FAR (human text, final, do not repeat or alter):\n{md_text}\n\nContinue.")
    raw = client.call_text(agent="writer", system=FINISH_SYSTEM, user=user)
    continuation, notes = split_prose_and_notes(raw)
    draft = Draft(beat_id=beat_id, version=current.version + 1,
                  prose=md_text + "\n\n" + continuation,
                  writer_notes=f"human prefix preserved ({len(md_text.split())} words); {notes}")
    save_draft(beat_dir, draft)
    print(f"writer: {beat_id} v{draft.version} finished "
          f"(+{len(continuation.split())} words after human text)")
    return draft


def polish_beat(client: ModelClient, run_dir: Path, beat_id: str, instruction: str) -> Draft:
    """Apply a human free-form instruction as span-targeted splices."""
    beat_dir = run_dir / "beats" / beat_id
    current = load_latest_draft(beat_dir)
    spec = BeatSpec.model_validate_json((beat_dir / "01_beat_spec.json").read_text(encoding="utf-8"))
    out = client.call_structured(
        agent="writer", system=POLISH_SYSTEM,
        user=(f"BEAT SPEC (reference):\n{spec.model_dump_json(indent=2)}\n\n"
              f"CURRENT DRAFT (v{current.version}):\n{current.prose}\n\n"
              f"HUMAN INSTRUCTION:\n{instruction}"),
        schema=RevisionOutput,
    )
    new_prose, missed = apply_replacements(current.prose, out.replacements)
    if missed:
        print(f"writer: WARNING - anchors not found for {missed} (skipped)")
    draft = Draft(beat_id=beat_id, version=current.version + 1, prose=new_prose,
                  writer_notes=f"polish per human instruction: {instruction!r}; "
                               f"applied {len(out.replacements) - len(missed)} spans")
    save_draft(beat_dir, draft)
    print(f"writer: {beat_id} v{draft.version} polished "
          f"({len(out.replacements) - len(missed)}/{len(out.replacements)} spans)")
    return draft


# --------------------------------------------------------------- CLI


USAGE = """usage:
  uv run python -m src.agents.writer <run_dir> [beat_id]              draft
  uv run python -m src.agents.writer ingest <run_dir> <beat_id>       promote your .md edit
  uv run python -m src.agents.writer finish <run_dir> <beat_id>       continue your partial text
  uv run python -m src.agents.writer polish <run_dir> <beat_id> "..." apply your instruction"""


def main() -> None:
    args = sys.argv[1:]
    if not args:
        sys.exit(USAGE)
    verb = args[0] if args[0] in ("ingest", "finish", "polish") else "draft"
    if verb == "draft":
        if len(args) not in (1, 2) or not Path(args[0]).is_dir():
            sys.exit(USAGE)
        run_dir = Path(args[0])
        client = ModelClient(load_config(), run_dir)
        beat_ids = [args[1]] if len(args) == 2 else sorted(
            p.name for p in (run_dir / "beats").iterdir() if p.is_dir())
        for beat_id in beat_ids:
            draft_beat(client, run_dir, beat_id)
        return
    if verb == "ingest" and len(args) == 3:
        ingest_beat(Path(args[1]), args[2]); return
    if verb == "finish" and len(args) == 3:
        run_dir = Path(args[1])
        finish_beat(ModelClient(load_config(), run_dir), run_dir, args[2]); return
    if verb == "polish" and len(args) == 4:
        run_dir = Path(args[1])
        polish_beat(ModelClient(load_config(), run_dir), run_dir, args[2], args[3]); return
    sys.exit(USAGE)


if __name__ == "__main__":
    main()
