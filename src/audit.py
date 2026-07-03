"""Audit: zero-cost mechanical checks on drafts and specs. Pure code,
no model calls. Run after every draft; treat 'error' findings as
must-fix, 'review' findings as judgment calls.

    uv run python -m src.audit <run_dir> [beat_id]

Checks (narration only, except anchors - dialogue rules are deferred
and get their own system later):
- banned constructions (camera-test violations: mind-reading, judgment,
  hedges) as errors; similes and hedged phrasing as review
- naming convention against bible style.naming
- word count vs movement budgets and target range
- dialogue anchors, SOFT: found / split-by-tag / missing (report only)
- anatomy wordlist for the animal cast (review)
- spec hygiene: empty handoffs, flat tempo contour, budget sum

Writes 04_audit_v<N>.json next to the audited draft.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.schemas import BeatSpec, Draft

# --------------------------------------------------------------- model


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check: str
    severity: str  # "error" | "review" | "info"
    snippet: str = ""
    note: str = ""


class AuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    beat_id: str
    draft_version: int
    word_count: int
    errors: int = 0
    reviews: int = 0
    findings: list[Finding] = Field(default_factory=list)


# ------------------------------------------------------------- helpers

QUOTE_RE = re.compile(r'["\u201c][^"\u201c\u201d]*["\u201d]')


def narration_only(text: str) -> str:
    """Blank out quoted dialogue, preserving offsets for snippets."""
    return QUOTE_RE.sub(lambda m: " " * len(m.group(0)), text)


def snippet_at(text: str, start: int, end: int, radius: int = 40) -> str:
    lo, hi = max(0, start - radius), min(len(text), end + radius)
    return ("…" if lo else "") + text[lo:hi].replace("\n", " ").strip() + ("…" if hi < len(text) else "")


WORDS_RE = re.compile(r"[A-Za-z']+")


def word_list(text: str) -> list[str]:
    return [w.lower() for w in WORDS_RE.findall(text)]


# -------------------------------------------------------------- checks

BANNED = [
    (r"\bas though\b", "mind-reading / interpretation"),
    (r"\bas if\b", "mind-reading / interpretation"),
    (r"\bseem(?:s|ed|ing|ingly)?\b", "hedge / interpretation"),
    (r"\bsuggest(?:s|ed|ing)\b", "interpretation"),
    (r"\bapparently\b", "narrator judgment"),
    (r"\bclearly\b", "narrator judgment"),
    (r"\bevidently\b", "narrator judgment"),
    (r"\bobviously\b", "narrator judgment"),
    (r"\bno doubt\b", "narrator judgment"),
    (r"\b(?:felt|wondered|realized|decided|knew|thought|hoped|wanted)\b",
     "interior state narrated"),
]
REVIEW = [
    (r"\blike\s+(?:a|an|the|something|someone)\b", "possible simile"),
    (r"\b(?:kind|sort)\s+of\b", "hedged phrasing"),
    (r"\bprobably\b|\bperhaps\b|\bmaybe\b", "speculation"),
]
ANATOMY_ERROR = re.compile(
    r"\b(hand|hands|finger|fingers|finger's|fingers'|knuckle|knuckles|fist|fists)\b",
    re.IGNORECASE,
)
ANATOMY_REVIEW = re.compile(r"\b(arm|arms|arm's)\b", re.IGNORECASE)


def check_constructions(narration: str) -> list[Finding]:
    out = []
    for pattern, note in BANNED:
        for m in re.finditer(pattern, narration, re.IGNORECASE):
            out.append(Finding(check="camera", severity="error",
                               snippet=snippet_at(narration, m.start(), m.end()), note=note))
    for pattern, note in REVIEW:
        for m in re.finditer(pattern, narration, re.IGNORECASE):
            out.append(Finding(check="camera", severity="review",
                               snippet=snippet_at(narration, m.start(), m.end()), note=note))
    return out


def check_naming(narration: str, canonical: dict[str, str]) -> list[Finding]:
    out = []
    for key, canon in canonical.items():
        for m in re.finditer(rf"(?i)\b(the\s+)?{re.escape(key)}\b", narration):
            form = re.sub(r"\s+", " ", m.group(0))
            # sentence-start 'The Firebird' is fine for canonical 'the Firebird'
            normalized = ("the " + form[4:]) if form.lower().startswith("the ") else form
            if normalized == canon:
                continue
            if normalized.lower() == canon.lower():
                sev, note = "error", f"capitalization: use {canon!r}"
            elif canon.lower().startswith("the ") and normalized.lower() == canon.lower()[4:]:
                sev, note = "review", f"bare titular (address?): canonical is {canon!r}"
            else:
                sev, note = "error", f"use {canon!r}"
            out.append(Finding(check="naming", severity=sev,
                               snippet=snippet_at(narration, m.start(), m.end()), note=note))
    return out


def check_anatomy(narration: str) -> list[Finding]:
    out = [Finding(check="anatomy", severity="error",
                   snippet=snippet_at(narration, m.start(), m.end()),
                   note="paw/claws/grip - hands and their parts do not exist")
           for m in ANATOMY_ERROR.finditer(narration)]
    out += [Finding(check="anatomy", severity="review",
                    snippet=snippet_at(narration, m.start(), m.end()),
                    note="arms allowed only in bipedal action - check posture")
            for m in ANATOMY_REVIEW.finditer(narration)]
    return out


def check_length(draft: Draft, spec: BeatSpec) -> list[Finding]:
    out = []
    n = len(draft.prose.split())
    if not (spec.target_words_min <= n <= spec.target_words_max):
        out.append(Finding(check="length", severity="error",
                           note=f"{n} words, target {spec.target_words_min}-{spec.target_words_max}"))
    if spec.movements:
        budget = sum(m.word_budget for m in spec.movements)
        if budget and abs(n - budget) > 0.2 * budget:
            out.append(Finding(check="length", severity="review",
                               note=f"{n} words vs {budget} budgeted (>20% off)"))
    return out


def check_anchors(draft: Draft, spec: BeatSpec) -> list[Finding]:
    """SOFT: dialogue rules deferred. Reports found / split / missing."""
    out = []
    draft_words = word_list(draft.prose)
    for a in spec.dialogue_anchors:
        if not a.keep_verbatim:
            continue
        if a.line in draft.prose:
            continue  # verbatim, nothing to report
        anchor_words = word_list(a.line)
        # in-order subsequence scan
        i = 0
        positions = []
        for j, w in enumerate(draft_words):
            if i < len(anchor_words) and w == anchor_words[i]:
                positions.append(j); i += 1
        if i < len(anchor_words):
            out.append(Finding(check="anchor", severity="review",
                               snippet=a.line, note="words missing from draft"))
        elif positions and positions[-1] - positions[0] + 1 > len(anchor_words):
            out.append(Finding(check="anchor", severity="info",
                               snippet=a.line, note="present but split (tag/beat inserted)"))
        else:
            out.append(Finding(check="anchor", severity="info",
                               snippet=a.line, note="words intact; punctuation differs"))
    return out


def check_spec(spec: BeatSpec) -> list[Finding]:
    out = []
    for m in spec.movements:
        if not m.handoff.strip():
            out.append(Finding(check="spec", severity="review",
                               note=f"movement {m.order} has no handoff"))
    if len(spec.movements) >= 3 and len({m.tempo for m in spec.movements}) == 1:
        out.append(Finding(check="spec", severity="review",
                           note=f"flat tempo contour (all {spec.movements[0].tempo.value})"))
    return out


# ------------------------------------------------------------ pipeline


def load_naming(path: str | Path = "bible/bible.json") -> dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    bible = json.loads(p.read_text(encoding="utf-8"))
    return bible.get("style", {}).get("naming", {}).get("canonical", {})


def audit_beat(run_dir: Path, beat_id: str) -> AuditReport:
    beat_dir = run_dir / "beats" / beat_id
    versions = sorted(int(p.stem.split("_v")[1]) for p in beat_dir.glob("03_draft_v*.json"))
    if not versions:
        raise FileNotFoundError(f"no drafts for {beat_id}")
    v = versions[-1]
    draft = Draft.model_validate_json((beat_dir / f"03_draft_v{v}.json").read_text(encoding="utf-8"))
    spec = BeatSpec.model_validate_json((beat_dir / "01_beat_spec.json").read_text(encoding="utf-8"))

    narration = narration_only(draft.prose)
    findings = (check_constructions(narration)
                + check_naming(narration, load_naming())
                + check_anatomy(narration)
                + check_length(draft, spec)
                + check_anchors(draft, spec)
                + check_spec(spec))

    report = AuditReport(
        beat_id=beat_id, draft_version=v, word_count=len(draft.prose.split()),
        errors=sum(f.severity == "error" for f in findings),
        reviews=sum(f.severity == "review" for f in findings),
        findings=findings,
    )
    (beat_dir / f"04_audit_v{v}.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


def print_report(r: AuditReport) -> None:
    print(f"\n== {r.beat_id} v{r.draft_version}  "
          f"({r.word_count} words, {r.errors} errors, {r.reviews} review)")
    for f in r.findings:
        tag = {"error": "ERROR ", "review": "review", "info": "info  "}[f.severity]
        line = f"  [{tag}] {f.check}: {f.note}"
        if f.snippet:
            line += f"  |  {f.snippet}"
        print(line)
    if not r.findings:
        print("  clean")


def main() -> None:
    if len(sys.argv) not in (2, 3):
        sys.exit("usage: uv run python -m src.audit <run_dir> [beat_id]")
    run_dir = Path(sys.argv[1])
    beat_ids = [sys.argv[2]] if len(sys.argv) == 3 else sorted(
        p.name for p in (run_dir / "beats").iterdir() if p.is_dir())
    total_err = 0
    for beat_id in beat_ids:
        report = audit_beat(run_dir, beat_id)
        print_report(report)
        total_err += report.errors
    sys.exit(1 if total_err else 0)


if __name__ == "__main__":
    main()
