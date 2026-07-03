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
from src.schemas import BeatSpec, ContextPacket

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
  entering_state, word-compatible, no gaps, no contradictions.
- reader_knows_entering: the facts the reader holds as the beat opens.
  Grows monotonically as beats reveal things.
- threads: every plant and payoff that touches this beat, marked as
  'plant: ... (pays off in beat X)' or 'payoff: ... (planted in beat X)'.

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


# ------------------------------------------------------------ pipeline


def run_director(skeleton_path: str | Path, run_dir: Path | None = None) -> Path:
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
    for beat_id, excerpt in beats:
        beat_dir = run_dir / "beats" / beat_id
        spec_path = beat_dir / "01_beat_spec.json"
        packet_path = beat_dir / "02_context_packet.json"
        if spec_path.exists() and packet_path.exists():
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
    if len(sys.argv) not in (2, 3):
        sys.exit("usage: uv run python -m src.agents.director <skeleton_file> [existing_run_dir]")
    run_director(sys.argv[1], Path(sys.argv[2]) if len(sys.argv) == 3 else None)
