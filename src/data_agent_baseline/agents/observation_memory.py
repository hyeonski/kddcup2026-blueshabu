from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


def _serialize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(content)


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "...<truncated>"


def cap_preserved_observation(
    observation: dict[str, Any], step_index: int, max_chars: int
) -> dict[str, Any]:
    """preserved-K 윈도우용 observation 캡 처리.

    content를 직렬화한 문자열이 max_chars를 초과하면 잘라내고, 원본은
    recall_observation(step_index=N)으로 조회 가능함을 명시.
    """
    if max_chars <= 0 or not isinstance(observation, dict):
        return observation
    content = observation.get("content")
    serialized = _serialize_content(content)
    if len(serialized) <= max_chars:
        return observation
    head = serialized[:max_chars]
    capped_content = (
        f"{head}\n...<truncated; full content available via "
        f"recall_observation(step_index={step_index})>"
    )
    capped = dict(observation)
    capped["content"] = capped_content
    capped["_truncated"] = True
    capped["_original_length"] = len(serialized)
    return capped


@dataclass(frozen=True, slots=True)
class ObservationEntry:
    step_index: int
    tool: str
    action_input: dict[str, Any]
    observation: dict[str, Any]
    content_length: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RecallEvent:
    at_step: int
    recalled_step: int
    ok: bool


@dataclass(slots=True)
class ObservationMemory:
    head_chars: int = 200
    entries: dict[int, ObservationEntry] = field(default_factory=dict)
    recall_events: list[RecallEvent] = field(default_factory=list)

    def store(
        self,
        step_index: int,
        tool: str,
        action_input: dict[str, Any],
        observation: dict[str, Any],
    ) -> None:
        content = observation.get("content", "") if isinstance(observation, dict) else ""
        content_text = _serialize_content(content)
        self.entries[step_index] = ObservationEntry(
            step_index=step_index,
            tool=tool,
            action_input=dict(action_input) if isinstance(action_input, dict) else {},
            observation=observation,
            content_length=len(content_text),
        )

    def recall(self, step_index: int) -> ObservationEntry | None:
        return self.entries.get(step_index)

    def record_recall(self, at_step: int, recalled_step: int, ok: bool) -> None:
        self.recall_events.append(
            RecallEvent(at_step=at_step, recalled_step=recalled_step, ok=ok)
        )

    def render_index(self, exclude_step_indices: set[int]) -> str:
        rows = []
        for step_index in sorted(self.entries):
            if step_index in exclude_step_indices:
                continue
            entry = self.entries[step_index]
            input_repr = json.dumps(entry.action_input, ensure_ascii=False)
            if len(input_repr) > 160:
                input_repr = input_repr[:160] + "..."
            content_text = _serialize_content(entry.observation.get("content", ""))
            head = _truncate(content_text, self.head_chars).replace("\n", "\\n")
            ok_flag = entry.observation.get("ok", True)
            rows.append(
                f"[step={entry.step_index}] {entry.tool}({input_repr}) "
                f"ok={ok_flag} len={entry.content_length} head=\"{head}\""
            )
        if not rows:
            return ""
        body = "\n".join(rows)
        return (
            "<MEMORY_INDEX>\n"
            "# Older observations live in memory. Call `recall_observation` with `step_index` to retrieve any full content below.\n"
            f"{body}\n"
            "</MEMORY_INDEX>"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stored_steps": sorted(self.entries),
            "recall_calls": [
                {"at_step": e.at_step, "recalled_step": e.recalled_step, "ok": e.ok}
                for e in self.recall_events
            ],
        }
