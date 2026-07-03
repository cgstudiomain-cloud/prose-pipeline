"""Model client: the single chokepoint for every LLM call in the pipeline.

Responsibilities:
- Per-agent model/temperature/token config from config.yaml
- Structured calls: ask for JSON, validate against a Pydantic schema,
  retry with the validation error fed back to the model
- Text calls: for prose outputs (drafts), where JSON wrapping would hurt
- Log every attempt to <run_dir>/calls.jsonl — timestamps, tokens,
  latency, success/failure. This file is your debug trail and,
  later, your fine-tuning dataset.

Nothing else in the codebase talks to the API directly.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

import yaml
from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

load_dotenv()

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class ModelClient:
    def __init__(self, config: dict, run_dir: str | Path):
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "calls.jsonl"
        self.api = Anthropic()

    # ------------------------------------------------------------ config

    def _agent_cfg(self, agent: str) -> dict:
        cfg = dict(self.config.get("defaults", {}))
        cfg.update(self.config.get("agents", {}).get(agent, {}))
        return cfg

    # ------------------------------------------------------------ logging

    def _log(self, entry: dict) -> None:
        entry["ts"] = datetime.now(timezone.utc).isoformat()
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ---------------------------------------------------------- raw call

    def _raw_call(self, agent: str, system: str, user: str) -> tuple[str, dict]:
        cfg = self._agent_cfg(agent)
        t0 = time.perf_counter()
        resp = self.api.messages.create(
            model=cfg["model"],
            max_tokens=cfg.get("max_tokens", 4096),
            temperature=cfg.get("temperature", 0.7),
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        meta = {
            "agent": agent,
            "model": cfg["model"],
            "latency_ms": round((time.perf_counter() - t0) * 1000),
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }
        return resp.content[0].text, meta

    # -------------------------------------------------------- public API

    def call_text(self, agent: str, system: str, user: str) -> str:
        """For prose outputs (Writer drafts). Returns raw text."""
        text, meta = self._raw_call(agent, system, user)
        self._log({**meta, "kind": "text", "ok": True,
                   "system": system, "user": user, "response": text})
        return text

    def call_structured(
        self,
        agent: str,
        system: str,
        user: str,
        schema: type[T],
        max_retries: int = 2,
    ) -> T:
        """Ask for JSON matching `schema`. Validate. On failure, retry
        with the validation error shown to the model. Raises after
        max_retries exhausted."""
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        system_full = (
            f"{system}\n\n"
            f"Respond with a single JSON object matching this schema. "
            f"No preamble, no markdown fences, no commentary.\n\n{schema_json}"
        )
        prompt = user
        last_err: Exception | None = None

        for attempt in range(1, max_retries + 2):
            text, meta = self._raw_call(agent, system_full, prompt)
            cleaned = _FENCE_RE.sub("", text).strip()
            try:
                result = schema.model_validate_json(cleaned)
                self._log({**meta, "kind": "structured", "attempt": attempt,
                           "ok": True, "schema": schema.__name__,
                           "system": system, "user": user, "response": cleaned})
                return result
            except (ValidationError, json.JSONDecodeError) as err:
                last_err = err
                self._log({**meta, "kind": "structured", "attempt": attempt,
                           "ok": False, "schema": schema.__name__,
                           "error": str(err)[:2000],
                           "system": system, "user": user, "response": text})
                prompt = (
                    f"{user}\n\nYour previous response failed validation "
                    f"with this error:\n{err}\n\nReturn corrected JSON only."
                )

        raise RuntimeError(
            f"{agent}: output failed {schema.__name__} validation "
            f"after {max_retries + 1} attempts"
        ) from last_err


def new_run_dir(base: str | Path = "runs", label: str = "run") -> Path:
    """Create runs/<date>_<label>_<time>/ for this pipeline execution."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = Path(base) / f"{stamp}_{label}"
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    # Live smoke test: one structured call, validated against a tiny schema.
    from pydantic import Field

    class Probe(BaseModel):
        status: str = Field(..., description="the single word: online")
        lucky_number: int

    config = load_config()
    client = ModelClient(config, new_run_dir(label="smoke"))
    result = client.call_structured(
        agent="copy_editor",
        system="You are a test probe.",
        user="Report status 'online' and any lucky number.",
        schema=Probe,
    )
    print(f"structured call OK: {result!r}")
    print(f"log written to: {client.log_path}")
