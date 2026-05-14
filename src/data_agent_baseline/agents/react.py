from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Optional

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep, ModelAdapterLLMWrapper
from data_agent_baseline.agents.prompt import (
    REACT_PS_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.observation_memory import ObservationMemory, cap_preserved_observation
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import (
    RECALL_OBSERVATION_TOOL_NAME,
    ToolExecutionResult,
    ToolRegistry,
    recall_observation_spec,
)

try:
    from productive_agents.ctxopt.history_optimizer import HistoryOptimizer as ACONHistoryOptimizer
except ImportError:
    ACONHistoryOptimizer = None

# PROJECT_ROOT는 src 폴더의 상위 디렉토리 (프로젝트 루트)
# react.py: ...src/data_agent_baseline/agents/react.py
# parents[0]: agents, parents[1]: data_agent_baseline, parents[2]: src, parents[3]: PROJECT_ROOT
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 16
    parse_retries: int = 2
    wall_budget_seconds: float | None = None
    safety_margin_seconds: float = 30.0
    # ACON Context Optimization
    enable_context_optimization: bool = False
    # history_summarization_threshold: 이 토큰 이상이면 압축 수행 (-1 = 매번 압축)
    history_summarization_threshold: int = 4000
    # 압축 시 verbatim으로 보존할 가장 최근 step 수
    preserve_last_k_steps: int = 5
    # Observation Memory: 오래된 observation을 외부 메모리에 분리 보관하고 recall_observation으로 조회
    enable_observation_memory: bool = False
    memory_index_head_chars: int = 200
    # preserved-K 윈도우 안의 observation도 N자 초과 시 잘라냄 (recall로 원본 조회 가능).
    # 큰 출력(execute_python 등)으로 컨텍스트가 부풀어 ContextLengthExceeded가 나는 걸 방지.
    # 0 이하면 무제한 (캡 비활성).
    preserved_observation_max_chars: int = 5000


CheckpointWriter = Callable[[dict[str, Any]], None]


_PARSE_RECOVERY_HINT = (
    "Your previous response could not be parsed as a single JSON object: {error}. "
    "Reply with ONLY one JSON object containing keys `thought`, `action`, `action_input`. "
    "Do not include any prose, markdown fences, or trailing text."
)


_FORCE_ANSWER_HINT = (
    "You have not submitted an answer and this is your final attempt. "
    "Reply with ONLY one JSON object whose `action` is `answer` and whose "
    "`action_input` follows the answer tool schema (columns + rows). "
    "Use your best inference from the data and reasoning gathered so far. "
    "If you are uncertain, return your most likely candidate; an empty rows "
    "list is acceptable. Do not call any other tool. Do not include prose."
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
    # Plan-and-Solve: plan 필드 선택적으로 파싱 (있으면 사용, 없으면 공백)
    plan = payload.get("plan", "")
    
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")
    if not isinstance(action_input, dict):
        raise ValueError("action_input must be a JSON object.")
    if not isinstance(plan, str):
        raise ValueError("plan must be a string.")

    return ModelStep(
        thought=thought,
        action=action,
        action_input=action_input,
        raw_response=raw_response,
        plan=plan,
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
        self.system_prompt = system_prompt or REACT_PS_SYSTEM_PROMPT
        self.logger = logging.getLogger(__name__)
        
        # ACON HistoryOptimizer 초기화
        self.history_optimizer: Any = None
        self.compressed_history: str | None = None
        # 마지막 K step 보존
        self.preserve_last_k_steps: int = self.config.preserve_last_k_steps
        self.preserved_steps: list[StepRecord] = []  # 마지막 K step 보존할 steps

        # Observation Memory
        self.observation_memory: ObservationMemory | None = None
        if self.config.enable_observation_memory:
            self.observation_memory = ObservationMemory(
                head_chars=self.config.memory_index_head_chars,
            )
            # recall_observation spec을 도구 광고 목록에 주입 (handler는 run()에서 직접 처리)
            if RECALL_OBSERVATION_TOOL_NAME not in self.tools.specs:
                self.tools.specs[RECALL_OBSERVATION_TOOL_NAME] = recall_observation_spec()
        
        if self.config.enable_context_optimization and ACONHistoryOptimizer is not None:
            try:
                # ACON-main의 HistoryOptimizer를 사용
                # ACON은 config.get("history_prompt_dir", None)를 우선으로 확인
                prompt_dir_path = str(PROJECT_ROOT / "acon-main" / "experiments" / "appworld" / "prompts" / "context_opt")
                
                acon_config = {
                    "model": model.model if hasattr(model, 'model') else "qwen/qwen3.5-35b-a3b",
                    "temperature": model.temperature if hasattr(model, 'temperature') else 1.0,
                    "history_summarization_threshold": self.config.history_summarization_threshold,
                    # ACON이 찾는 키 - config에서 우선 확인
                    "history_prompt_dir": prompt_dir_path,
                    # 프롬프트 템플릿 이름 설정
                    "prompts": {
                        "prompt_system": "system_prompt",
                        "prompt_history_user": "prompt_history_v2",  # v2 버전 사용
                    }
                }
                
                # ModelAdapter를 ACON 호환 형식으로 래핑
                wrapped_model = ModelAdapterLLMWrapper(model)
                
                # ACON의 HistoryOptimizer 생성
                # prompt_dir 파라미터는 무시됨 (config에서 history_prompt_dir이 우선)
                self.history_optimizer = ACONHistoryOptimizer(
                    config=acon_config,
                    debug_mode=False,
                    llm=wrapped_model
                )
            except Exception as e:
                self.logger.warning(f"[ACON] ✗ Failed to initialize HistoryOptimizer: {e}")
                import traceback
                self.logger.warning(traceback.format_exc())
                self.history_optimizer = None

    
    def _has_retry_budget(self, started_at: float) -> bool:
        budget = self.config.wall_budget_seconds
        if budget is None:
            return True
        elapsed = perf_counter() - started_at
        return elapsed <= max(budget - self.config.safety_margin_seconds, 0.0)

    def _can_retry_empty_answer(self, step_index: int, started_at: float) -> bool:
        return step_index < self.config.max_steps and self._has_retry_budget(started_at)



    def _build_messages(self, task: PublicTask, state: AgentRuntimeState) -> list[ModelMessage]:
        """Build messages for model. Only include preserved last K steps."""
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
            enable_observation_memory=self.observation_memory is not None,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        
        # Task + Compressed History를 하나의 user 메시지로
        task_content = build_task_prompt(task)
        if self.compressed_history:
            task_content += f"\n\n<HISTORY_SUMMARY>\n{self.compressed_history}\n</HISTORY_SUMMARY>"

        # 보존된 steps + 그 이후의 새로운 steps 추가
        if self.preserved_steps:
            # 보존된 steps 이후의 모든 새 steps 찾기
            last_preserved_step_idx = self.preserved_steps[-1].step_index
            new_steps_after_compression = [s for s in state.steps if s.step_index > last_preserved_step_idx]
            steps_to_add = self.preserved_steps + new_steps_after_compression
        else:
            # 압축 전: 모든 steps 사용
            steps_to_add = state.steps

        # Observation Memory: preserved 윈도우 밖의 모든 step을 인덱스로 노출
        if self.observation_memory is not None:
            preserved_indices = {s.step_index for s in steps_to_add}
            index_block = self.observation_memory.render_index(preserved_indices)
            if index_block:
                task_content += f"\n\n{index_block}"

        messages.append(ModelMessage(role="user", content=task_content))

        for step in steps_to_add:
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            obs_for_prompt = step.observation
            if (
                self.observation_memory is not None
                and self.config.preserved_observation_max_chars > 0
            ):
                obs_for_prompt = cap_preserved_observation(
                    step.observation,
                    step.step_index,
                    self.config.preserved_observation_max_chars,
                )
            messages.append(
                ModelMessage(role="user", content=build_observation_prompt(obs_for_prompt))
            )
        
        return messages

    def _complete_with_parse_recovery(
        self, task_id: str, base_messages: list[ModelMessage], step_id: int
    ) -> tuple[ModelStep | None, str, str | None]:
        """Call model and parse; retry on parse errors with a corrective hint.

        Returns (model_step, last_raw_response, last_error). model_step is None when
        all retries failed.
        """
        messages = list(base_messages)
        last_raw_response = ""
        last_error: str | None = None
        for attempt in range(self.config.parse_retries + 1):
            #last_raw_response = self.model.complete(messages)
            try:
                last_raw_response = self.model.complete(messages)
            except RuntimeError as e:
                if "missing text content" in str(e):
                    if attempt < self.config.parse_retries:
                        continue
                    else:
                        raise
                raise
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

    def _execute_recall(
        self, action_input: dict[str, Any], at_step: int
    ) -> ToolExecutionResult:
        assert self.observation_memory is not None
        raw = action_input.get("step_index")
        try:
            step_index = int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            self.observation_memory.record_recall(at_step=at_step, recalled_step=-1, ok=False)
            return ToolExecutionResult(
                ok=False,
                content={"error": f"recall_observation requires integer `step_index`, got {raw!r}"},
            )
        entry = self.observation_memory.recall(step_index)
        if entry is None:
            self.observation_memory.record_recall(at_step=at_step, recalled_step=step_index, ok=False)
            return ToolExecutionResult(
                ok=False,
                content={"error": f"No observation stored for step {step_index}."},
            )
        self.observation_memory.record_recall(at_step=at_step, recalled_step=step_index, ok=True)
        return ToolExecutionResult(
            ok=True,
            content={
                "recalled_step_index": entry.step_index,
                "tool": entry.tool,
                "action_input": entry.action_input,
                "observation": entry.observation,
            },
        )

    def _memory_usage_payload(self) -> dict[str, Any] | None:
        if self.observation_memory is None:
            return None
        return {"enabled": True, **self.observation_memory.to_dict()}

    def _build_partial_result(self, task: PublicTask, state: AgentRuntimeState) -> AgentRunResult:
        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
            memory_usage=self._memory_usage_payload(),
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

    def _try_compress_history(self, task: PublicTask, state: AgentRuntimeState, step_id: int) -> None:
        """ACON 방식으로 history 압축: 마지막 K step 이전만 압축"""
        task_id = task.task_id if hasattr(task, 'task_id') else "unknown"
        
        # Step 1: 마지막 K step 분리
        preserve_steps_count = self.preserve_last_k_steps
        
        if len(state.steps) <= preserve_steps_count:
            # 아직 충분한 step이 없으면 압축 불필요
            self.preserved_steps = state.steps
            return
        
        steps_for_compression = state.steps[:-preserve_steps_count]
        self.preserved_steps = state.steps[-preserve_steps_count:]
        
        # Step 2: 보존할 steps 이전 것만 압축
        if self.history_optimizer is None:
            # ACON 미초기화: preserved_steps만 갱신하고 종료
            # (메모리 단독 모드일 때 이 경로로 흐름)
            return
        history_text = self._format_steps_for_compression(steps_for_compression)
        if not history_text.strip():
            return

        # Step 3: 압축 필요 여부 확인
        needs_compression = self.history_optimizer.check_summarization_needed(
            history_text, self.compressed_history
        )
        
        if not needs_compression:
            return
        
        # Step 4: 압축 수행
        try:
            compression_start_idx = steps_for_compression[0].step_index if steps_for_compression else 0
            compression_end_idx = steps_for_compression[-1].step_index if steps_for_compression else 0
            preserve_start_idx = self.preserved_steps[0].step_index if self.preserved_steps else 0
            preserve_end_idx = self.preserved_steps[-1].step_index if self.preserved_steps else 0
            
            compressed = self.history_optimizer.process(
                task=task.question,
                history=history_text,
                prev_history_summary=self.compressed_history,
                raw_history=[],
                opt_args={}
            )
            
            if isinstance(compressed, tuple):
                compressed_text = compressed[0]
            else:
                compressed_text = compressed
            
            if compressed_text:
                self.compressed_history = compressed_text
                
        except Exception as e:
            import traceback
            self.logger.debug(traceback.format_exc())

    # NOTE: Context-column validation removed per user request; rely on prompt guidance only.

    def _format_steps_for_compression(self, steps: list[StepRecord]) -> str:
        """ACON HistoryOptimizer에 전달할 형식으로 steps를 포맷
        
        Plan-and-Solve: plan 필드 포함으로 계획 정보 보존
        """
        formatted = []
        for step in steps:
            step_text = f"Step {step.step_index}:\n"
            step_text += f"  Thought: {step.thought}\n"
            # Plan-and-Solve: 계획 단계 표시 (있으면 포함)
            # 원본 출처: Plan-and-Solve-Prompting의 history 포맷
            if hasattr(step, 'plan') and step.plan:
                step_text += f"  Plan: {step.plan}\n"
            step_text += f"  Action: {step.action}\n"
            step_text += f"  Action Input: {json.dumps(step.action_input, ensure_ascii=False)}\n"
            if step.observation:
                if self.observation_memory is not None:
                    # 메모리 ON: ACON 요약 입력에는 observation 본문 대신 메타만 노출
                    ok_flag = step.observation.get("ok", True)
                    tool_name = step.observation.get("tool", step.action)
                    step_text += (
                        f"  Observation: <stored in memory; tool={tool_name} ok={ok_flag}>\n"
                    )
                else:
                    obs_str = json.dumps(step.observation, ensure_ascii=False)
                    step_text += f"  Observation: {obs_str}\n"
            formatted.append(step_text)
        return "\n".join(formatted)

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
            
            model_step, raw_response, parse_error = self._complete_with_parse_recovery(task.task_id, base_messages, step_index)
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
                        plan="",  # Plan-and-Solve: error 시에도 plan 필드 포함
                    )
                )
                self._flush_checkpoint(task, state, checkpoint_writer)
                continue
            plan_preview = model_step.plan.strip().replace("\n", " / ")
            if len(plan_preview) > 140:
                plan_preview = plan_preview[:140] + "..."

            try:
                if (
                    self.observation_memory is not None
                    and model_step.action == RECALL_OBSERVATION_TOOL_NAME
                ):
                    tool_result = self._execute_recall(model_step.action_input, step_index)
                else:
                    tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
                retry_empty_answer = (
                    tool_result.is_terminal
                    and tool_result.answer is not None
                    and len(tool_result.answer.rows) == 0
                    and self._can_retry_empty_answer(step_index, started_at)
                )
                if retry_empty_answer:
                    observation["retry"] = True
                    observation["retry_reason"] = (
                        "submitted answer has 0 rows; continue reasoning and resubmit only if a non-empty result exists"
                    )
                step_record = StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    raw_response=raw_response,
                    observation=observation,
                    ok=tool_result.ok and not retry_empty_answer,
                    plan=model_step.plan,  # Plan-and-Solve: LLM으로부터 받은 계획 저장
                )
                state.steps.append(step_record)

                # Observation Memory: recall 호출은 저장하지 않음 (재귀 방지)
                if (
                    self.observation_memory is not None
                    and model_step.action != RECALL_OBSERVATION_TOOL_NAME
                ):
                    self.observation_memory.store(
                        step_index=step_index,
                        tool=model_step.action,
                        action_input=model_step.action_input,
                        observation=observation,
                    )

                if tool_result.is_terminal:
                    if retry_empty_answer:
                        self._try_compress_history(task, state, step_index)
                        self._flush_checkpoint(task, state, checkpoint_writer)
                        continue
                    # Accept terminal answer as-is (no programmatic schema validation)
                    state.answer = tool_result.answer
                    self._flush_checkpoint(task, state, checkpoint_writer)
                    break
                
                # Context compression 시도
                self._try_compress_history(task, state, step_index)
                
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
                        plan=model_step.plan,  # Plan-and-Solve: 예외 발생 시에도 계획 저장
                    )
                )
                self._flush_checkpoint(task, state, checkpoint_writer)

        if state.answer is None and state.steps:
            self._attempt_forced_answer(task, state, checkpoint_writer)

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."

        return self._build_partial_result(task, state)

    def _attempt_forced_answer(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        checkpoint_writer: CheckpointWriter | None,
    ) -> None:
        forced_index = (state.steps[-1].step_index if state.steps else 0) + 1
        base_messages = self._build_messages(task, state)
        forced_messages = base_messages + [ModelMessage(role="user", content=_FORCE_ANSWER_HINT)]
        try:
            model_step, raw_response, parse_error = self._complete_with_parse_recovery(task.task_id, forced_messages, forced_index)
        except Exception as exc:  # noqa: BLE001
            state.steps.append(
                StepRecord(
                    step_index=forced_index,
                    thought="",
                    action="__forced_error__",
                    action_input={},
                    raw_response="",
                    observation={"ok": False, "error": str(exc)},
                    ok=False,
                    plan="",  # Plan-and-Solve
                )
            )
            self._flush_checkpoint(task, state, checkpoint_writer)
            return

        if model_step is None:
            state.steps.append(
                StepRecord(
                    step_index=forced_index,
                    thought="",
                    action="__forced_error__",
                    action_input={},
                    raw_response=raw_response,
                    observation={"ok": False, "error": parse_error or "parse failed"},
                    ok=False,
                    plan="",  # Plan-and-Solve
                )
            )
            self._flush_checkpoint(task, state, checkpoint_writer)
            return

        if model_step.action != "answer":
            state.steps.append(
                StepRecord(
                    step_index=forced_index,
                    thought=model_step.thought,
                    action=f"__forced_non_answer__:{model_step.action}",
                    action_input=model_step.action_input,
                    raw_response=raw_response,
                    observation={"ok": False, "error": "Forced attempt did not call the answer tool."},
                    ok=False,
                    plan=model_step.plan,  # Plan-and-Solve
                )
            )
            self._flush_checkpoint(task, state, checkpoint_writer)
            return

        try:
            tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
        except Exception as exc:  # noqa: BLE001
            state.steps.append(
                StepRecord(
                    step_index=forced_index,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    raw_response=raw_response,
                    observation={"ok": False, "error": str(exc)},
                    ok=False,
                    plan=model_step.plan,  # Plan-and-Solve
                )
            )
            self._flush_checkpoint(task, state, checkpoint_writer)
            return

        observation = {"ok": tool_result.ok, "tool": "answer (forced)", "content": tool_result.content}
        state.steps.append(
            StepRecord(
                step_index=forced_index,
                thought=model_step.thought,
                action="answer",
                action_input=model_step.action_input,
                raw_response=raw_response,
                observation=observation,
                ok=tool_result.ok,
                plan=model_step.plan,  # Plan-and-Solve
            )
        )
        if tool_result.is_terminal and tool_result.answer is not None:
            state.answer = tool_result.answer
        self._flush_checkpoint(task, state, checkpoint_writer)
