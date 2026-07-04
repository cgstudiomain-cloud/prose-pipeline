"""Director agent, two-phase.

Pass 1: one small call -> ChapterPlan (state chain, reader knowledge,
        plant/payoff threads across all beats).
Pass 2: one call per beat -> BeatSpec + ContextPacket, honoring the plan.

Every beat is a checkpoint: if its spec files already exist in the run
directory, it is skipped. A crashed or unsatisfying run resumes with:

    uv run python -m src.agents.director <skeleton_file> <existing_run_dir>

Fresh run:
    uv run python -m src.agents.director skeletons\\stealing_the_sword_marked.txt

To re-roll a single beat: delete its two json files from the run
directory and resume.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.client import ModelClient, load_config, new_run_dir
from src.schemas import BeatSpec, ContextPacket, Phase

BEAT_MARKER = re.compile(r"^===\s*BEAT:\s*(?P<beat_id>[\w-]+)\s*===\s*$", re.MULTILINE)


# ------------------------------------------------------------- models


class BeatPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    beat_id: str
    purpose: str = Field(..., description="Story function of this beat, one or two sentences")
    entering_state: str
    exiting_state: str
    reader_knows_entering: list[str] = Field(
        default_factory=list, description="Facts the reader holds as this beat opens"
    )
    threads: list[str] = Field(
        default_factory=list,
        description="Plants and payoffs touching this beat, e.g. 'plant: sword refuses fox (pays off beat 03)'",
    )
    chapter_phase: Phase = Field(
        ..., description="This beat's role in the chapter-scale wave"
    )
    peak: str = Field(
        ..., description="The beat's single peak moment, in camera terms"
    )


class ChapterPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    beats: list[BeatPlan]


class DecodeFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = Field(..., description="one of: annotate, beat")
    anchor_text: str = Field(
        ..., description="Exact verbatim quote from the text (<=12 words) marking where this attaches"
    )
    proposed_text: str = Field(
        ..., description="annotate: the intent annotation, one clause to one sentence, no braces. "
                         "beat: the beat id in snake_case, e.g. 01_cold_open"
    )


class DecodeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: list[DecodeFinding]


class FlowFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = Field(..., description="one of: insert_before, insert_after, replace")
    anchor_text: str = Field(
        ..., description="Exact, verbatim quote from the skeleton (<=15 words) locating the spot"
    )
    proposed_text: str = Field(..., description="The proposed insertion or replacement, in skeleton register")
    gap: str = Field(..., description="One or two sentences: what is missing and why it matters")


class FlowOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: list[FlowFinding]
    works_well: list[str] = Field(default_factory=list)


class BeatOutput(BaseModel):
    """One beat's full specification."""

    model_config = ConfigDict(extra="forbid")

    spec: BeatSpec
    packet: ContextPacket


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


from src.bible import load_bible  # noqa: E402  (re-exported for writer)


# ------------------------------------------------------------- prompts

FLOW_SYSTEM = """\
You are the Director running a FLOW CHECK on a chapter skeleton,
before any beats are specced. You propose; the human author decides.
You never rewrite the skeleton - you produce a report.

Walk the skeleton as a chain of reader attention - beads: question,
answer that poses the next question, on and on. At every transition,
test:
- MOTIVATION: does each character have an externalized purpose? Is
  every pursuit driven (recognition, need, interruption of a purpose),
  never idle curiosity?
- BEADS: is each question posed by evidence before its answer? Does
  each answer open the next question? Where does the chain drop?
- MICRO-BEATS: are reaction moments missing that the flow demands -
  recognitions, freezes, double-takes, the pause before a reply?
- BLOCKING: do entrances, exits, and positions work logistically?
  Where does each character enter from, where do they stand, can they
  speak at conversational distance? Does the end of each beat place
  everyone where the next beat needs them?
- STAKES AND SETUP: does each beat's ending set the stage - physically
  and dramatically - for what follows?

Output contract: structured findings. Each finding has:
- kind: insert_before, insert_after, or replace
- anchor_text: an EXACT verbatim quote from the skeleton, 15 words or
  fewer, locating the spot (the line the proposal attaches to, or the
  line being replaced)
- proposed_text: the proposal, written in the skeleton's own terse
  register, ready to keep as-is
- gap: one or two sentences naming what is missing and why it matters
Plus works_well: a list of things that work and must not be touched.

Text inside {braces} is RATIFIED AUTHOR INTENT - highest authority.
Judge the events against it: if the events do not deliver what an
annotation promises, that is a finding. Never propose changes to the
annotations themselves. And the stronger clause: when an annotation
DELIBERATELY breaks a doctrine rule - skips an absorption, removes a
grace period, denies a character dignity - that is a design choice.
Proposing the missing element back is a violation, not a finding.

POV DISCIPLINE: identify the POV character from the bible and the
skeleton. The chain runs through the POV character's attention:
never propose showing an off-POV character's preparation, motivation,
or movement that the POV character does not witness. Off-POV actions
arrive as stimuli, unprepared.

READER-PROTAGONIST UNITY: the reader and the POV character live in
the same space - they see, feel, and ask the same things at the same
moments. Never propose information that reaches the reader ahead of
the protagonist, in any form: staging, sound cues, differentiating
footsteps, shadows. The wave is SUBORDINATE to unity - anticipation
exists only when the POV character experiences it; a designed ambush
stays unannounced. Perception is gated by the POV character's state:
a stunned character's camera records shock, not room inventory -
overload first, observation later.

RUN THE SIM before every finding: inhabit the POV body in that
moment. Where is attention, actually? What questions is the reader
asking right now? Propose only what survives the sim.

MOTION CARRIES DIALOGUE: setup, exposition, and relationship beats
ride on movement toward the destination; arrival carries the event.
Flag static talk at a destination; never propose parking characters
so a conversation can happen.

MOMENTUM PROTECTION: where {} designs abruptness - a jolt, a hard
cut, no grace period - add NOTHING around it: no preparatory motion,
no turns, no transitional blocking, no softening in any form.

NO RESTATEMENT: do not propose a beat the skeleton's text already
contains or directly implies.

CAST HIERARCHY: the bible names the protagonist. Supporting
characters do not gain scenes, exchanges, or motivation beats merely
because their sheets are detailed. Screen time follows story role.

CANON-FIRST MOTIVATION: ground every character motivation you propose
in their bible sheet. Where the bible is silent, do not invent - say
so in the gap ("canon gap: the bible does not state...") and keep the
proposal minimal or ask. Be conservative when proposing replacement
dialogue for voiced characters: conform to the voice sheets, and
prefer flagging a delivery problem over rewriting the line.

SUMMARY REGISTER: lines that summarize instead of sequencing ("a walk
up to X: A doing this, B doing that") are compressed placeholders.
Flag each one and propose the unfolded event sequence - locations
staged through movement, purposes externalized, the approach given
its own anticipation.

You may propose events, micro-beats, and blocking. The human ratifies.
Stay inside the story's world and characters as the bible defines them.
"""

DECODE_SYSTEM = """\
You are the Director running a DECODE pass on plain development text -
before beat boundaries exist, before any flow check. The text mixes
events and implied intent; your job is to surface the intent into an
explicit annotation channel and propose beat boundaries. You propose;
the human ratifies.

The annotation channel:
- {...} is RATIFIED author intent. Never modify it, never contradict
  it, never annotate territory it already covers. Read it as truth.
- Your proposals are emitted as findings of kind "annotate"; they will
  be rendered as {? ...} for the author to ratify or delete.

ADMISSION RULE - strict: an annotation earns its place only if it
would change what a Director builds from the events. Qualifying:
timing ("lands one beat too late"), register ("deadpan-ominous"),
riddle mechanics ("the reader assembles it from one word"), staging
philosophy ("established by overwhelm, no tour"), dramatic irony that
spans scenes, tempo intent. Not qualifying: restatement of the events,
"this shows their relationship", anything a competent Director would
build identically without. When in doubt, stay silent.

NO DUBBING: if a ratified {...} annotation already touches the same
line or moment, you may propose only the DELTA - a mechanism the
existing annotation lacks (timing, ordering, a tempo constraint) -
and your proposal must not restate any part of the existing
annotation's content, not even one of its words of art. If what you
would add is mostly overlap, stay silent.

Annotations are blueprint language - explicit is correct here. Keep
each to one clause or one sentence. Attach each with anchor_text: an
exact verbatim quote (12 words or fewer) from the text; the annotation
will be inserted immediately after it.

Beat boundaries: propose findings of kind "beat" at the natural
dramatic joints - shifts of location, time, cast, or dramatic
function. proposed_text is the beat id in snake_case, numbered in
document order (01_..., 02_...). anchor_text is the first words of
the line the beat begins at.
"""


PLAN_SYSTEM = """\
You are the Director in a prose-production pipeline, pass one: chapter
planning. You do not write prose and you do not spec beats yet. You
produce the chapter's connective tissue so that each beat can later be
specced in isolation without the story falling apart.

For each beat, in order:
- purpose: what this beat must change in the reader's understanding or
  the characters' standing. Story function, not plot summary.
- entering_state / exiting_state: the observable state of the world and
  characters. These MUST chain: beat N's exiting_state is beat N+1's
  entering_state, word-compatible, no gaps, no contradictions. An
  entering_state contains NO new events - anything that happens
  belongs inside the beat's movements, not at its threshold.
- reader_knows_entering: the facts the reader holds as the beat opens.
  Grows monotonically as beats reveal things.
- threads: every plant and payoff that touches this beat, marked as
  'plant: ... (pays off in beat X)' or 'payoff: ... (planted in beat X)'.
- chapter_phase and peak: the chapter is itself a wave. Decide which
  beat delivers the chapter's central action, which beats build
  anticipation toward it, and which absorb it - assign each beat its
  chapter_phase accordingly. Then name each beat's single peak moment
  in camera terms: the one moment the beat exists to deliver.

All states in camera terms: observable facts only, no interior states.
Stay strictly inside the skeleton's events.
"""

BEAT_SYSTEM = """\
You are the Director in a prose-production pipeline, pass two: beat
specification. You unfold ONE beat's skeleton into a filmable
specification that a Writer who has never seen the rest of the story
can execute in isolation. You are given the full skeleton and the
chapter plan for orientation; you spec only the beat you are asked for.

Work in three passes:

PASS 1 - PULL APART. Identify the locations, the flow of events, and
the beat's riddle: what question is the reader made to hold, what
observable evidence poses it, and where does the answer land. Effect
before cause; sound before source; the reveal is the payoff.

PASS 2 - UNFOLD INTO MOVEMENTS. Break the beat into an ordered shot
list. Each movement has a phase from the AAA system:
- anticipation: build the question before the answer
- action: deliver cleanly and briefly - actions get SMALL budgets
- absorption: hold after the action while the reader processes -
  reactions, stillness, silence. Absorption is never skipped and
  never rushed.
Give each movement a word_budget (camera time). Budgets must sum to
roughly the beat's target word range. A typical beat is 4-8 movements.

The movements form ONE wave rising to the beat's peak (given in the
chapter plan): anticipation movements build toward it, the action
movement delivers it lean, absorption follows and holds. Micro-waves
inside are allowed but stay subordinate to the main wave - never
alternate phases mechanically.

Every movement gets a tempo - slow, medium, or fast: the narration
density, independent of phase (banter is fast action; a blade drawn in
silence is slow action). Design the beat's tempo CONTOUR: speeds must
vary across the movements, trading slow against fast so the beat rides
like a wave, not a flat line. A beat where every movement shares one
tempo is a defect.

Every movement declares its handoff: the unresolved thing - open
question, unfinished motion, unanswered line - that carries the
reader's attention into the next movement. The final movement's
handoff is the beat's exit hook into the next beat. A question may
only resolve while another is open or opening; attention is released
only at the chapter's final beat. If a movement has no honest handoff,
the movement boundary is wrong - merge or re-cut.

Every movement's content must be written as a causal chain in strict
stimulus -> response order: the cause on the page before the effect,
the character reacting only after the stimulus, objects appearing when
a character's attention lands on them - never as advance decor. If two
sentences of content could be swapped without breaking anything, the
chain is broken. When several things become visible at once, register
them in salience order: the loud, bright, moving, out-of-place first.
Reader-protagonist unity governs everything in close POV: nothing
reaches the reader ahead of the POV character, anticipation exists
only as the POV character experiences it, and perception is gated by
the POV character's state - a stunned camera records shock, not a
room inventory.
Respect physical truth in content: no dialogue during maximal effort -
effort runs set, pull, fail, release, and speech follows the release.
Pose unknowns affirmatively ("gripping something"), never as negated
perception ("could not see what").

PASS 3 - STAGE. Invent concrete texture for each movement: places,
props, physical business. You MAY invent texture; you may NEVER invent
plot, events, or dialogue that changes what happens. Every staging
element must declare its function - what it stages, plants, or
characterizes. Texture with no function is not admitted.

All staging content must be written in camera terms: observable
action and speech only, no interior states. Canonical example of the
difference - the skeleton says the cat walks a corridor. WRONG staging
(processing): "cat heads toward the kitchens." RIGHT staging
(evidence): "tight, dim service corridor; cat stops, sniffs the air,
says 'This way. Definitely.'" The destination is carried by behavior
and speech, never by narration.

Text inside {braces} in the skeleton is RATIFIED AUTHOR INTENT -
binding, highest authority. Translate it into movements, tempo,
riddle, staging, and handoffs. NEVER quote its wording into movement
content or staging - blueprint language dies at this boundary; only
its effect survives.

Other duties:
- entering_state and exiting_state must match the chapter plan's
  entries for this beat.
- The ContextPacket must let a writer who cannot see the other beats
  avoid contradictions: build reader_knows from the plan's
  reader_knows_entering; make must_remain_true and forbidden complete.
- dialogue_anchors: pull the lines from the skeleton that carry the
  scene. Mark keep_verbatim=true for lines whose exact wording matters.
  Place anchored dialogue inside the movements where it belongs (refer
  to it in movement content).
- Use the beat's own skeleton text, verbatim, as skeleton_excerpt.
- scene_goal comes from the plan's purpose for this beat.
"""


def build_plan_prompt(skeleton_text: str, beats: list[tuple[str, str]], bible: dict, guidelines: str) -> str:
    beat_list = "\n".join(f"- {bid}" for bid, _ in beats)
    return f"""\
STORY BIBLE:
{json.dumps(bible, indent=2, ensure_ascii=False)}

GUIDELINE PACK:
{guidelines}

FULL SKELETON (beat markers included):
{skeleton_text}

Produce the chapter plan for these beats, in this order:
{beat_list}
"""


def build_beat_prompt(
    beat_id: str,
    excerpt: str,
    skeleton_text: str,
    plan: ChapterPlan,
    bible: dict,
    guidelines: str,
) -> str:
    return f"""\
STORY BIBLE:
{json.dumps(bible, indent=2, ensure_ascii=False)}

GUIDELINE PACK:
{guidelines}

CHAPTER PLAN:
{plan.model_dump_json(indent=2)}

FULL SKELETON (for orientation only):
{skeleton_text}

SPEC THIS BEAT: {beat_id}

THIS BEAT'S SKELETON TEXT (use verbatim as skeleton_excerpt):
{excerpt}
"""


# ------------------------------------------------------------ checks


def verify_plan(plan: ChapterPlan, expected_ids: list[str]) -> None:
    plan_ids = [b.beat_id for b in plan.beats]
    if plan_ids != expected_ids:
        raise ValueError(f"ChapterPlan ids {plan_ids} != expected {expected_ids}")


def verify_beat(out: BeatOutput, beat_id: str) -> None:
    if out.spec.beat_id != beat_id or out.packet.beat_id != beat_id:
        raise ValueError(
            f"beat id mismatch: asked for {beat_id}, "
            f"got spec={out.spec.beat_id} packet={out.packet.beat_id}"
        )
    spec = out.spec
    if not spec.movements:
        print(f"director: WARNING {beat_id} has no movements")
        return
    orders = [m.order for m in spec.movements]
    if orders != sorted(orders):
        print(f"director: WARNING {beat_id} movements out of order")
    total = sum(m.word_budget for m in spec.movements)
    lo, hi = spec.target_words_min * 0.8, spec.target_words_max * 1.2
    if not (lo <= total <= hi):
        print(f"director: WARNING {beat_id} budgets sum to {total}, "
              f"target {spec.target_words_min}-{spec.target_words_max}")
    for m in spec.movements:
        if not m.handoff.strip():
            print(f"director: WARNING {beat_id} movement {m.order} has no handoff")
    if len(spec.movements) >= 3 and len({m.tempo for m in spec.movements}) == 1:
        print(f"director: WARNING {beat_id} tempo contour is flat "
              f"(all movements {spec.movements[0].tempo.value})")


# ------------------------------------------------------------ pipeline


def _line_bounds(text: str, idx: int, length: int) -> tuple[int, int]:
    start = text.rfind("\n", 0, idx) + 1
    end = text.find("\n", idx + length)
    return start, (len(text) if end == -1 else end + 1)


def annotate_skeleton(skeleton: str, findings: list[FlowFinding]) -> tuple[str, list[FlowFinding]]:
    """Weave marker blocks into a copy of the skeleton. The skeleton's
    own lines are never altered; proposals are standalone marked lines.
    Returns (annotated_text, unplaced_findings)."""
    placed, unplaced = [], []
    for f in findings:
        m = _flex_pattern(f.anchor_text.strip()).search(skeleton)
        if m is None:
            unplaced.append(f)
            continue
        placed.append((m.start(), f))
    placed.sort(key=lambda t: t[0])

    pieces, cursor = [], 0
    for n, (idx, f) in enumerate(placed, start=1):
        fid = f"F{n:02d}"
        f.gap = f"{fid} — {f.gap}"          # tag rationale with its id
        start, end = _line_bounds(skeleton, idx, len(f.anchor_text.strip()))
        if f.kind == "insert_before":
            pieces.append(skeleton[cursor:start])
            pieces.append(f"⟦{fid}⟧\n{f.proposed_text.strip()}\n⟦/{fid}⟧\n")
            cursor = start
        elif f.kind == "replace":
            pieces.append(skeleton[cursor:end])
            pieces.append(f"⟦{fid} replaces the line above⟧\n{f.proposed_text.strip()}\n⟦/{fid}⟧\n")
            cursor = end
        else:  # insert_after (default)
            pieces.append(skeleton[cursor:end])
            pieces.append(f"⟦{fid}⟧\n{f.proposed_text.strip()}\n⟦/{fid}⟧\n")
            cursor = end
    pieces.append(skeleton[cursor:])
    annotated = "".join(pieces)

    # fidelity guarantee: stripping marker blocks restores the original
    kept, inside = [], False
    for line in annotated.splitlines(keepends=True):
        if line.startswith("⟦/"):
            inside = False
        elif line.startswith("⟦"):
            inside = True
        elif not inside:
            kept.append(line)
    assert "".join(kept) == skeleton, "annotation altered the skeleton - aborting"
    return annotated, unplaced


PAREN_RE = re.compile(r"\(([^()]*)\)", re.S)
BRACE_STRIP_RE = re.compile(r"\s?\{\??[^{}]*\}")


def lift_parentheticals(text: str) -> str:
    """Author parentheticals become ratified {} intent, in place."""
    return PAREN_RE.sub(lambda m: "{" + m.group(1) + "}", text)


BRACE_SPAN_RE = re.compile(r"\{\??[^{}]*\}", re.S)


def _flex_pattern(needle: str) -> re.Pattern:
    """Match the needle with any whitespace run (incl. line wraps)
    wherever the needle has whitespace."""
    return re.compile(r"\s+".join(re.escape(w) for w in needle.split()))


def _brace_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in BRACE_SPAN_RE.finditer(text)]


def _find_outside_braces(text: str, needle: str, spans: list[tuple[int, int]]) -> tuple[int, int, bool]:
    """First whitespace-flexible occurrence of needle not overlapping
    any {} span. Returns (start, end, found_anywhere)."""
    found_any = False
    for m in _flex_pattern(needle).finditer(text):
        found_any = True
        if not any(a < m.end() and m.start() < b for a, b in spans):
            return m.start(), m.end(), True
    return -1, -1, found_any


def weave_decode(text: str, findings: list[DecodeFinding]) -> tuple[str, list[tuple[DecodeFinding, str]]]:
    """Insert {? ...} inline after anchors; wrap proposed beat markers
    in ⟦Bxx⟧ blocks before the anchor's line. Never alters the text.
    Ratified {} regions are positionally off-limits: anchors inside
    them are routed to unplaced. Returns (annotated, [(finding, why)])."""
    spans = _brace_spans(text)
    placed, unplaced = [], []
    for f in findings:
        needle = f.anchor_text.strip()
        idx, a_end, found_any = _find_outside_braces(text, needle, spans)
        if idx == -1:
            why = ("anchor only occurs inside ratified {} intent"
                   if found_any else "anchor not found")
            unplaced.append((f, why))
            continue
        if f.kind == "beat":
            start = text.rfind("\n", 0, idx) + 1
            if any(a < start < b for a, b in spans):
                unplaced.append((f, "beat boundary would split a {} region"))
                continue
        placed.append((idx, a_end, f))
    placed.sort(key=lambda t: t[0])

    pieces, cursor, beat_n = [], 0, 0
    for idx, a_end, f in placed:
        clean = f.proposed_text.replace("{", "").replace("}", "").strip()
        if f.kind == "beat":
            beat_n += 1
            start = text.rfind("\n", 0, idx) + 1
            if start < cursor:
                unplaced.append((f, "overlaps an earlier insertion"))
                continue
            pieces.append(text[cursor:start])
            pieces.append(f"⟦B{beat_n:02d}⟧\n=== BEAT: {clean} ===\n⟦/B{beat_n:02d}⟧\n")
            cursor = start
        else:
            if a_end < cursor:
                unplaced.append((f, "overlaps an earlier insertion"))
                continue
            pieces.append(text[cursor:a_end])
            pieces.append(" {? " + clean + "}")
            cursor = a_end
    pieces.append(text[cursor:])
    annotated = "".join(pieces)

    # fidelity: strip ⟦⟧ blocks and all {} / {?} spans -> must recover
    # the brace-stripped original exactly
    kept, inside = [], False
    for line in annotated.splitlines(keepends=True):
        if line.startswith("⟦/"):
            inside = False
        elif line.startswith("⟦"):
            inside = True
        elif not inside:
            kept.append(line)
    check = BRACE_STRIP_RE.sub("", "".join(kept))
    base = BRACE_STRIP_RE.sub("", text)
    if check != base:
        hint = next((repr(check[max(0, i - 40):i + 40])
                     for i, (a, b) in enumerate(zip(check, base)) if a != b),
                    "length mismatch")
        raise RuntimeError(
            f"decode weaving altered the text - aborting. First divergence near: {hint}"
        )
    return annotated, unplaced


def run_decode(text_path: str | Path) -> Path:
    """Pass 0a: decode plain development text. Lifts author (...) to
    ratified {}, proposes {? ...} intent and ⟦Bxx⟧ beat boundaries."""
    raw = Path(text_path).read_text(encoding="utf-8")
    lifted = lift_parentheticals(raw)
    bible, guidelines = load_bible(), load_guidelines()
    run_dir = new_run_dir(label=f"decode_{Path(text_path).stem}")
    client = ModelClient(load_config(), run_dir)
    print(f"director: decoding {Path(text_path).name} (1 call)...")
    out = client.call_structured(
        agent="director",
        system=DECODE_SYSTEM,
        user=(f"STORY BIBLE:\n{json.dumps(bible, indent=2, ensure_ascii=False)}\n\n"
              f"GUIDELINE PACK:\n{guidelines}\n\n"
              f"DEVELOPMENT TEXT ({{...}} is ratified intent - do not touch or duplicate):\n{lifted}"),
        schema=DecodeOutput,
    )
    annotated, unplaced = weave_decode(lifted, out.findings)
    doc = ["# Decoded skeleton — ratify the {? } proposals",
           "",
           "{...} = your ratified intent (lifted from your parentheticals or "
           "written by you). {? ...} = machine proposal: delete '? ' to "
           "ratify, delete the block to reject. ⟦Bxx⟧ blocks propose beat "
           "boundaries: delete the two marker lines to accept the "
           "=== BEAT === line, delete the whole block to reject.",
           "", "---", "", annotated]
    if unplaced:
        doc += ["", "⟦UNPLACED — review manually⟧"]
        doc += [f"[{f.kind}] ({why}) anchor: {f.anchor_text!r} -> {f.proposed_text}"
                for f, why in unplaced]
        doc += ["⟦/UNPLACED⟧"]
    out_path = run_dir / "00_decoded.md"
    out_path.write_text("\n".join(doc), encoding="utf-8")
    n_beats = sum(1 for f in out.findings if f.kind == "beat")
    print(f"decode -> {out_path}  ({len(out.findings) - len(unplaced) - n_beats} annotations proposed, "
          f"{n_beats} beat boundaries, {len(unplaced)} unplaced)")
    return out_path


def run_flow_check(skeleton_path: str | Path) -> Path:
    """Pass 0: flow-check the skeleton. Output: the skeleton itself,
    annotated inline with marked proposals + rationale footnotes.
    Accept a proposal: delete its two marker lines (and, for a
    replacement, the original line above). Reject: delete the block."""
    skeleton_text = Path(skeleton_path).read_text(encoding="utf-8")
    bible, guidelines = load_bible(), load_guidelines()
    run_dir = new_run_dir(label=f"flow_{Path(skeleton_path).stem}")
    client = ModelClient(load_config(), run_dir)
    print(f"director: flow-checking {Path(skeleton_path).name} (1 call)...")
    out = client.call_structured(
        agent="director",
        system=FLOW_SYSTEM,
        user=(f"STORY BIBLE:\n{json.dumps(bible, indent=2, ensure_ascii=False)}\n\n"
              f"GUIDELINE PACK (the flow ruleset lives here):\n{guidelines}\n\n"
              f"SKELETON:\n{skeleton_text}"),
        schema=FlowOutput,
    )
    annotated, unplaced = annotate_skeleton(skeleton_text, out.findings)

    doc = ["# Annotated skeleton — flow check",
           "",
           "Accept a proposal: delete its ⟦Fxx⟧ / ⟦/Fxx⟧ marker lines "
           "(for a replacement, also delete the original line above). "
           "Reject: delete the whole block. Rationales: 00_flow_rationale.md.",
           "", "---", "", annotated]
    out_path = run_dir / "00_flow_annotated.md"
    out_path.write_text("\n".join(doc), encoding="utf-8")

    rat = ["# Flow check — rationale", ""]
    rat += [f"- {f.gap}" for f in out.findings if f.gap.startswith("F")]
    if unplaced:
        rat += ["", "## Unplaced findings (anchor not found — review manually)"]
        rat += [f"- [{f.kind}] anchor: {f.anchor_text!r}\n  proposal: {f.proposed_text}\n  gap: {f.gap}"
                for f in unplaced]
    if out.works_well:
        rat += ["", "## Works well — do not touch"]
        rat += [f"- {w}" for w in out.works_well]
    (run_dir / "00_flow_rationale.md").write_text("\n".join(rat), encoding="utf-8")

    print(f"flow check -> {out_path}  (+ 00_flow_rationale.md; "
          f"{len(out.findings) - len(unplaced)} placed, {len(unplaced)} unplaced)")
    return out_path


def run_director(
    skeleton_path: str | Path,
    run_dir: Path | None = None,
    only_beat: str | None = None,
) -> Path:
    skeleton_text = Path(skeleton_path).read_text(encoding="utf-8")
    beats = split_beats(skeleton_text)
    expected_ids = [bid for bid, _ in beats]
    bible, guidelines = load_bible(), load_guidelines()

    run_dir = run_dir or new_run_dir(label=Path(skeleton_path).stem)
    client = ModelClient(load_config(), run_dir)

    # ---- pass 1: chapter plan (checkpointed)
    plan_path = run_dir / "00_chapter_plan.json"
    if plan_path.exists():
        plan = ChapterPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
        print("director: chapter plan found, resuming")
    else:
        print("director: planning chapter (1 call)...")
        plan = client.call_structured(
            agent="director",
            system=PLAN_SYSTEM,
            user=build_plan_prompt(skeleton_text, beats, bible, guidelines),
            schema=ChapterPlan,
        )
        verify_plan(plan, expected_ids)
        plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        print(f"director: chapter plan written ({len(plan.beats)} beats)")

    # ---- pass 2: one call per beat (each beat a checkpoint)
    if only_beat is not None and only_beat not in expected_ids:
        raise ValueError(f"unknown beat_id {only_beat!r}; have {expected_ids}")
    for beat_id, excerpt in beats:
        if only_beat is not None and beat_id != only_beat:
            continue
        beat_dir = run_dir / "beats" / beat_id
        spec_path = beat_dir / "01_beat_spec.json"
        packet_path = beat_dir / "02_context_packet.json"
        if only_beat is None and spec_path.exists() and packet_path.exists():
            print(f"director: {beat_id} already specced, skipping")
            continue
        print(f"director: speccing {beat_id} "
              f"({expected_ids.index(beat_id) + 1}/{len(expected_ids)})...")
        out = client.call_structured(
            agent="director",
            system=BEAT_SYSTEM,
            user=build_beat_prompt(beat_id, excerpt, skeleton_text, plan, bible, guidelines),
            schema=BeatOutput,
        )
        verify_beat(out, beat_id)
        beat_dir.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(out.spec.model_dump_json(indent=2), encoding="utf-8")
        packet_path.write_text(out.packet.model_dump_json(indent=2), encoding="utf-8")
        print(f"director: {beat_id} specced "
              f"({len(out.spec.movements)} movements, "
              f"{sum(m.word_budget for m in out.spec.movements)} words budgeted)")

    print(f"director: done -> {run_dir}")
    return run_dir


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "decode":
        if len(sys.argv) != 3:
            sys.exit("usage: uv run python -m src.agents.director decode <plain_text_file>")
        run_decode(sys.argv[2])
        sys.exit(0)
    if len(sys.argv) >= 2 and sys.argv[1] == "flow":
        if len(sys.argv) != 3:
            sys.exit("usage: uv run python -m src.agents.director flow <skeleton_file>")
        run_flow_check(sys.argv[2])
        sys.exit(0)
    if len(sys.argv) not in (2, 3, 4):
        sys.exit("usage: uv run python -m src.agents.director "
                 "<skeleton_file> [existing_run_dir] [beat_id]\n"
                 "       uv run python -m src.agents.director flow <skeleton_file>")
    run_director(
        sys.argv[1],
        Path(sys.argv[2]) if len(sys.argv) >= 3 else None,
        sys.argv[3] if len(sys.argv) == 4 else None,
    )
