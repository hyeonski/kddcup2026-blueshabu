from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 16
    parse_retries: int = 2
    wall_budget_seconds: float | None = None
    safety_margin_seconds: float = 30.0


CheckpointWriter = Callable[[dict[str, Any]], None]


_PARSE_RECOVERY_HINT = (
    "Your previous response could not be parsed as a single JSON object: {error}. "
    "Reply with ONLY one JSON object containing keys `thought`, `action`, `action_input`. "
    "Do not include any prose, markdown fences, or trailing text."
)


def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
    return text


def _load_single_json_object(text: str) -> dict[str, object]:
    payload, end = json.JSONDecoder().raw_decode(text)
    remainder = text[end:].strip()
    if remainder:
        cleaned_remainder = re.sub(r"(?:\\[nrt])+", "", remainder).strip()
        if cleaned_remainder:
            raise ValueError("Model response must contain only one JSON object.")
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")
    return payload


def parse_model_step(raw_response: str) -> ModelStep:
    normalized = _strip_json_fence(raw_response)
    payload = _load_single_json_object(normalized)

    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get("action_input", {})
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")
    if not isinstance(action_input, dict):
        raise ValueError("action_input must be a JSON object.")

    return ModelStep(
        thought=thought,
        action=action,
        action_input=action_input,
        raw_response=raw_response,
    )


class ReActAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT

    def _build_messages(self, task: PublicTask, state: AgentRuntimeState) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        messages.append(ModelMessage(role="user", content=build_task_prompt(task)))
        for step in state.steps:
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            messages.append(
                ModelMessage(role="user", content=build_observation_prompt(step.observation))
            )
        return messages

    def _complete_with_parse_recovery(
        self, base_messages: list[ModelMessage]
    ) -> tuple[ModelStep | None, str, str | None]:
        """Call model and parse; retry on parse errors with a corrective hint.

        Returns (model_step, last_raw_response, last_error). model_step is None when
        all retries failed.
        """
        messages = list(base_messages)
        last_raw_response = ""
        last_error: str | None = None
        for attempt in range(self.config.parse_retries + 1):
            last_raw_response = self.model.complete(messages)
            try:
                return parse_model_step(last_raw_response), last_raw_response, None
            except Exception as exc:  # noqa: BLE001 — surface any parse failure
                last_error = str(exc)
                if attempt == self.config.parse_retries:
                    break
                messages = messages + [
                    ModelMessage(role="assistant", content=last_raw_response),
                    ModelMessage(
                        role="user",
                        content=_PARSE_RECOVERY_HINT.format(error=last_error),
                    ),
                ]
        return None, last_raw_response, last_error

    def _build_partial_result(self, task: PublicTask, state: AgentRuntimeState) -> AgentRunResult:
        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )

    def _flush_checkpoint(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        writer: CheckpointWriter | None,
    ) -> None:
        if writer is None:
            return
        try:
            writer(self._build_partial_result(task, state).to_dict())
        except Exception:  # noqa: BLE001 — checkpoint failure must not crash the agent
            pass

    def run(
        self,
        task: PublicTask,
        *,
        checkpoint_writer: CheckpointWriter | None = None,
    ) -> AgentRunResult:
        state = AgentRuntimeState()
        started_at = perf_counter()
        budget = self.config.wall_budget_seconds
        margin = self.config.safety_margin_seconds
        for step_index in range(1, self.config.max_steps + 1):
            if budget is not None and (perf_counter() - started_at) > max(budget - margin, 0.0):
                state.failure_reason = (
                    f"Wall budget reached after step {step_index - 1} (budget={budget}s, margin={margin}s)."
                )
                break
            base_messages = self._build_messages(task, state)
            model_step, raw_response, parse_error = self._complete_with_parse_recovery(base_messages)
            if model_step is None:
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action="__error__",
                        action_input={},
                        raw_response=raw_response,
                        observation={"ok": False, "error": parse_error or "parse failed"},
                        ok=False,
                    )
                )
                self._flush_checkpoint(task, state, checkpoint_writer)
                continue
            try:
                tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
                step_record = StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    raw_response=raw_response,
                    observation=observation,
                    ok=tool_result.ok,
                )
                state.steps.append(step_record)
                if tool_result.is_terminal:
                    state.answer = tool_result.answer
                    self._flush_checkpoint(task, state, checkpoint_writer)
                    break
                self._flush_checkpoint(task, state, checkpoint_writer)
            except Exception as exc:
                observation = {
                    "ok": False,
                    "error": str(exc),
                }
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought=model_step.thought,
                        action=model_step.action,
                        action_input=model_step.action_input,
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                    )
                )
                self._flush_checkpoint(task, state, checkpoint_writer)

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
