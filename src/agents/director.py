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


def load_bible(path: str | Path = "bible/bible.json") -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


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

Report format, in markdown:
For each finding: a numbered heading with the location (beat marker or
quoted skeleton line), the GAP (one or two sentences), and PROPOSED
INSERTION written in the skeleton's own terse register, ready to paste.
Order findings by position in the skeleton. End with a short section
listing anything that works well and should not be touched.

You may propose events, micro-beats, and blocking. The human ratifies.
Stay inside the story's world and characters as the bible defines them.
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


def run_flow_check(skeleton_path: str | Path) -> Path:
    """Pass 0: flow-check the skeleton, write a human-facing report.
    Proposes skeleton insertions; changes nothing."""
    skeleton_text = Path(skeleton_path).read_text(encoding="utf-8")
    bible, guidelines = load_bible(), load_guidelines()
    run_dir = new_run_dir(label=f"flow_{Path(skeleton_path).stem}")
    client = ModelClient(load_config(), run_dir)
    report = client.call_text(
        agent="director",
        system=FLOW_SYSTEM,
        user=(f"STORY BIBLE:\n{json.dumps(bible, indent=2, ensure_ascii=False)}\n\n"
              f"GUIDELINE PACK (the flow ruleset lives here):\n{guidelines}\n\n"
              f"SKELETON:\n{skeleton_text}"),
    )
    out = run_dir / "00_flow_report.md"
    out.write_text(report, encoding="utf-8")
    print(f"flow check -> {out}")
    return out


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
