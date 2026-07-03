"""Typed contracts for the prose pipeline.

Every agent reads and writes these models, serialized as JSON on disk.
Prose exists only inside `Draft.prose` and `FinalBeat.prose` — everything
else is structured data.

Design decisions baked in here:
- Spans are referenced by exact quotes (`anchor_text`), not character
  offsets. LLMs quote reliably; they count characters unreliably.
- `extra="forbid"` everywhere: if a model call returns JSON with fields
  we didn't ask for, validation fails loudly instead of silently
  accepting drift. Loud failures are debuggable failures.
- Enums for anything an agent must choose from a closed set.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- enums


class Severity(str, Enum):
    BLOCKER = "blocker"   # beat cannot pass with this unresolved
    MAJOR = "major"       # hurts the beat, fix if possible
    MINOR = "minor"       # polish; Chief Editor may drop it


class Verdict(str, Enum):
    PASS = "pass"
    REVISE = "revise"


class AgentName(str, Enum):
    DIRECTOR = "director"
    WRITER = "writer"
    CONTINUITY = "continuity_editor"
    STYLE = "style_editor"
    READER = "reader_panel"
    CHIEF = "chief_editor"
    COPY = "copy_editor"
    ASSEMBLER = "assembler"
    CHAPTER_QA = "chapter_qa"


# ------------------------------------------------------- director output


class Callback(StrictModel):
    """A setup/payoff the Writer must plant or honor in this beat."""

    description: str = Field(..., description="What must be planted or paid off")
    kind: str = Field(..., description="'plant' or 'payoff'")
    source: str = Field(
        "", description="Where the counterpart lives, e.g. 'beat 02' or 'chapter 3'"
    )


class DialogueAnchor(StrictModel):
    """A line from the skeleton that must survive into prose,
    verbatim or near-verbatim."""

    line: str
    speaker: str
    keep_verbatim: bool = True


class DirectingNote(StrictModel):
    """Flashlight/wave instruction: where the reader's attention points,
    and how long it holds."""

    focus: str = Field(..., description="What the flashlight is pointed at")
    hold: str = Field(
        ..., description="How long / why it holds, e.g. 'linger through the silence'"
    )


class BeatSpec(StrictModel):
    beat_id: str = Field(..., description="e.g. '03_drawing_the_blade'")
    title: str
    pov: str = Field(..., description="POV character and distance, e.g. 'cat, close third'")
    scene_goal: str = Field(..., description="What this beat must accomplish in the story")
    entering_state: str = Field(..., description="World + character state as the beat opens")
    exiting_state: str = Field(..., description="What must be true when the beat ends")
    callbacks: list[Callback] = Field(default_factory=list)
    dialogue_anchors: list[DialogueAnchor] = Field(default_factory=list)
    directing_notes: list[DirectingNote] = Field(default_factory=list)
    target_words_min: int = 300
    target_words_max: int = 700
    skeleton_excerpt: str = Field(..., description="The raw skeleton text this beat covers")


class ContextPacket(StrictModel):
    """Everything a Writer needs to draft this beat in isolation —
    the key to running beats in parallel."""

    beat_id: str
    reader_knows: list[str] = Field(
        default_factory=list, description="Facts the reader has at this point"
    )
    must_remain_true: list[str] = Field(
        default_factory=list, description="Canon facts this beat may not contradict"
    )
    prev_beat_exit: str = Field("", description="How the previous beat ends (summary)")
    next_beat_entry: str = Field("", description="How the next beat opens (summary)")
    character_voices: dict[str, str] = Field(
        default_factory=dict, description="Voice notes per character, e.g. {'fox': 'fast, deflecting'}"
    )
    forbidden: list[str] = Field(
        default_factory=list, description="Things the beat must NOT reveal or do"
    )


# --------------------------------------------------------- writer output


class Draft(StrictModel):
    beat_id: str
    version: int = 1
    prose: str
    writer_notes: str = Field(
        "", description="Choices the Writer made and why — feeds the editors"
    )


# --------------------------------------------------------- editor output


class Note(StrictModel):
    """One issue found by one editor."""

    note_id: str = Field(..., description="e.g. 'cont-01', 'style-03'")
    agent: AgentName
    anchor_text: str = Field(
        ..., description="Exact quote from the draft locating the issue"
    )
    issue: str
    severity: Severity
    suggested_fix: str = ""


class EditorReport(StrictModel):
    beat_id: str
    draft_version: int
    agent: AgentName
    notes: list[Note] = Field(default_factory=list)


class ReaderScores(StrictModel):
    beat_id: str
    draft_version: int
    pacing: int = Field(..., ge=1, le=10)
    clarity: int = Field(..., ge=1, le=10)
    vocab_fit: int = Field(..., ge=1, le=10, description="Fit to target audience level")
    engagement: int = Field(..., ge=1, le=10)
    flagged: list[Note] = Field(default_factory=list)


# --------------------------------------------------- chief editor output


class PatchInstruction(StrictModel):
    """One span-targeted edit. The Writer changes this span and
    leaves the rest of the draft alone."""

    patch_id: str
    anchor_text: str = Field(..., description="Exact quote from the draft to modify")
    instruction: str = Field(..., description="What to change and toward what effect")
    source_note_ids: list[str] = Field(
        default_factory=list, description="Which editor notes this patch resolves"
    )
    priority: int = Field(1, ge=1, le=3, description="1 = do first")


class PatchDocument(StrictModel):
    beat_id: str
    draft_version: int
    verdict: Verdict
    patches: list[PatchInstruction] = Field(default_factory=list)
    dropped_note_ids: list[str] = Field(
        default_factory=list, description="Notes the Chief Editor chose to ignore"
    )
    rationale: str = Field("", description="Why this verdict, what conflicts were resolved how")


# ----------------------------------------------------------- final stages


class FinalBeat(StrictModel):
    beat_id: str
    prose: str
    copyedit_changelog: list[str] = Field(default_factory=list)
    revision_loops_used: int = 0


class SeamEdit(StrictModel):
    """One change the Assembler made while stitching beats."""

    location: str = Field(..., description="e.g. 'seam between 02 and 03'")
    reason: str = Field(..., description="e.g. 'repeated amber-eyes image', 'abrupt transition'")
    anchor_text: str = ""


class AssembledChapter(StrictModel):
    chapter_id: str
    beat_ids: list[str]
    prose: str
    seam_edits: list[SeamEdit] = Field(default_factory=list)


class ChapterQAReport(StrictModel):
    chapter_id: str
    verdict: Verdict
    notes: list[Note] = Field(default_factory=list)
    summary: str = ""


if __name__ == "__main__":
    # Smoke test: build a tiny spec from the sword-theft skeleton and
    # round-trip it through JSON — proves serialization works.
    spec = BeatSpec(
        beat_id="03_drawing_the_blade",
        title="Drawing the blade",
        pov="cat, close third",
        scene_goal="The bluff becomes real: the sword answers the cat.",
        entering_state="Firebird demands proof; guards tense; fox cornered.",
        exiting_state="Blade drawn a third and resheathed; the room silent; balance of power flipped.",
        dialogue_anchors=[
            DialogueAnchor(line="Behind the pillar first, then into the window", speaker="fox"),
        ],
        directing_notes=[
            DirectingNote(
                focus="the blade sliding out, sun on dark metal",
                hold="hold through the full silence before anyone moves",
            ),
        ],
        skeleton_excerpt="Cat takes the sword from the floor and puts his right paw on the handle...",
    )
    as_json = spec.model_dump_json(indent=2)
    back = BeatSpec.model_validate_json(as_json)
    assert back == spec
    print(as_json)
    print("\nschemas: round-trip OK")
