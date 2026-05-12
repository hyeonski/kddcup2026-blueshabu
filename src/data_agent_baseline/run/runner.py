from __future__ import annotations

import csv
import json
import multiprocessing
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import AppConfig
from data_agent_baseline.scoring.normalize import normalize_value
from data_agent_baseline.tools.registry import ToolRegistry, create_default_tool_registry

CHECKPOINT_FILENAME = "trace.partial.json"
TRACE_FILENAME = "trace.json"


@dataclass(frozen=True, slots=True)
class TaskRunArtifacts:
    task_id: str
    task_output_dir: Path
    prediction_csv_path: Path | None
    trace_path: Path
    succeeded: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_output_dir": str(self.task_output_dir),
            "prediction_csv_path": str(self.prediction_csv_path) if self.prediction_csv_path else None,
            "trace_path": str(self.trace_path),
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
        }


def create_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _short_model(model: str) -> str:
    # "qwen/qwen3.5-35b-a3b" → "qwen3.5-35b-a3b"
    return model.rsplit("/", 1)[-1]


def _fmt_temp(t: float) -> str:
    s = f"{t:g}"
    return s if "." in s else f"{s}.0"


def build_param_suffix(config: AppConfig) -> str:
    """Deterministic param suffix for run_id: encodes T/H/K and ACON/PS flags."""
    parts = [f"T{_fmt_temp(config.agent.temperature)}"]
    if config.agent.enable_context_optimization:
        parts.append(f"H{config.agent.history_summarization_threshold}")
        parts.append(f"K{config.agent.preserve_last_k_steps}")
    return "-".join(parts)


def build_run_id_from_config(config: AppConfig) -> str:
    """Build a fully self-describing run_id from agent config."""
    return f"{_short_model(config.agent.model)}-{build_param_suffix(config)}"


def agent_config_snapshot(config: AppConfig) -> dict[str, Any]:
    """Serialize the agent config fields that influence benchmark outcomes."""
    return {
        "model": config.agent.model,
        "temperature": config.agent.temperature,
        "max_steps": config.agent.max_steps,
        "wall_budget_seconds": config.agent.wall_budget_seconds,
        "safety_margin_seconds": config.agent.safety_margin_seconds,
        "seed": config.agent.seed,
        "provider_only": list(config.agent.provider_only),
        "enable_context_optimization": config.agent.enable_context_optimization,
        "history_summarization_threshold": config.agent.history_summarization_threshold,
        "preserve_last_k_steps": config.agent.preserve_last_k_steps,
        "task_timeout_seconds": config.run.task_timeout_seconds,
    }


def resolve_run_id(run_id: str | None = None) -> str:
    if run_id is None:
        return create_run_id()

    normalized = run_id.strip()
    if not normalized:
        raise ValueError("run_id must not be empty.")
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError("run_id must be a single directory name, not a path.")
    return normalized


def create_run_output_dir(output_root: Path, *, run_id: str | None = None) -> tuple[str, Path]:
    effective_run_id = resolve_run_id(run_id)
    run_output_dir = output_root / effective_run_id
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return effective_run_id, run_output_dir


def build_model_adapter(config: AppConfig):
    return OpenAIModelAdapter(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        temperature=config.agent.temperature,
        provider_only=config.agent.provider_only,
        seed=config.agent.seed,
    )


def _resolve_wall_budget(config: AppConfig) -> float | None:
    if config.agent.wall_budget_seconds is not None:
        return config.agent.wall_budget_seconds
    if config.run.task_timeout_seconds <= 0:
        return None
    derived = config.run.task_timeout_seconds - config.agent.safety_margin_seconds
    return max(derived, 1.0)


def _build_react_config(config: AppConfig) -> ReActAgentConfig:
    return ReActAgentConfig(
        max_steps=config.agent.max_steps,
        wall_budget_seconds=_resolve_wall_budget(config),
        safety_margin_seconds=config.agent.safety_margin_seconds,
        enable_context_optimization=config.agent.enable_context_optimization,
        history_summarization_threshold=config.agent.history_summarization_threshold,
        preserve_last_k_steps=config.agent.preserve_last_k_steps,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp_path, path)


def _make_checkpoint_writer(checkpoint_path: Path) -> Callable[[dict[str, Any]], None]:
    def _writer(payload: dict[str, Any]) -> None:
        _atomic_write_json(checkpoint_path, payload)
    return _writer


def _read_checkpoint(checkpoint_path: Path) -> dict[str, Any] | None:
    if not checkpoint_path.exists():
        return None
    try:
        return json.loads(checkpoint_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([normalize_value(cell) for cell in row])


def _failure_run_result_payload(task_id: str, failure_reason: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "answer": None,
        "steps": [],
        "failure_reason": failure_reason,
        "succeeded": False,
    }


def _run_single_task_core(
    *,
    task_id: str,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    public_dataset = DABenchPublicDataset(config.dataset.root_path)
    task = public_dataset.get_task(task_id)

    agent = ReActAgent(
        model=model or build_model_adapter(config),
        tools=tools or create_default_tool_registry(),
        config=_build_react_config(config),
    )
    writer = _make_checkpoint_writer(checkpoint_path) if checkpoint_path is not None else None
    run_result = agent.run(task, checkpoint_writer=writer)
    return run_result.to_dict()


def _run_single_task_in_subprocess(
    task_id: str,
    config: AppConfig,
    checkpoint_path_str: str | None,
    queue: multiprocessing.Queue[Any],
) -> None:
    try:
        checkpoint_path = Path(checkpoint_path_str) if checkpoint_path_str else None
        queue.put(
            {
                "ok": True,
                "run_result": _run_single_task_core(
                    task_id=task_id, config=config, checkpoint_path=checkpoint_path
                ),
            }
        )
    except BaseException as exc:  # noqa: BLE001
        queue.put(
            {
                "ok": False,
                "error": str(exc),
            }
        )


def _recover_from_checkpoint(
    task_id: str,
    checkpoint_path: Path | None,
    timeout_message: str,
) -> dict[str, Any]:
    if checkpoint_path is None:
        return _failure_run_result_payload(task_id, timeout_message)
    payload = _read_checkpoint(checkpoint_path)
    if payload is None:
        return _failure_run_result_payload(task_id, timeout_message)
    recovered_steps = len(payload.get("steps", []))
    payload["failure_reason"] = (
        f"{timeout_message} Recovered {recovered_steps} steps from checkpoint."
    )
    payload["succeeded"] = False
    return payload


def _run_single_task_with_timeout(
    *,
    task_id: str,
    config: AppConfig,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    timeout_seconds = config.run.task_timeout_seconds
    if timeout_seconds <= 0:
        return _run_single_task_core(task_id=task_id, config=config, checkpoint_path=checkpoint_path)

    queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_run_single_task_in_subprocess,
        args=(task_id, config, str(checkpoint_path) if checkpoint_path else None, queue),
    )
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join()
        return _recover_from_checkpoint(
            task_id,
            checkpoint_path,
            f"Task timed out after {timeout_seconds} seconds.",
        )

    if queue.empty():
        exit_code = process.exitcode
        if exit_code not in (None, 0):
            return _recover_from_checkpoint(
                task_id,
                checkpoint_path,
                f"Task exited unexpectedly with exit code {exit_code}.",
            )
        return _recover_from_checkpoint(
            task_id,
            checkpoint_path,
            "Task exited without returning a result.",
        )

    result = queue.get()
    if result.get("ok"):
        return dict(result["run_result"])
    return _recover_from_checkpoint(
        task_id,
        checkpoint_path,
        f"Task failed with uncaught error: {result['error']}",
    )


def _write_task_outputs(task_id: str, run_output_dir: Path, run_result: dict[str, Any]) -> TaskRunArtifacts:
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = task_output_dir / TRACE_FILENAME
    _write_json(trace_path, run_result)
    checkpoint_path = task_output_dir / CHECKPOINT_FILENAME
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
        except OSError:
            pass

    prediction_csv_path: Path | None = None
    answer = run_result.get("answer")
    if isinstance(answer, dict):
        prediction_csv_path = task_output_dir / "prediction.csv"
        _write_csv(
            prediction_csv_path,
            list(answer.get("columns", [])),
            [list(row) for row in answer.get("rows", [])],
        )

    return TaskRunArtifacts(
        task_id=task_id,
        task_output_dir=task_output_dir,
        prediction_csv_path=prediction_csv_path,
        trace_path=trace_path,
        succeeded=bool(run_result.get("succeeded")),
        failure_reason=run_result.get("failure_reason"),
    )


def run_single_task(
    *,
    task_id: str,
    config: AppConfig,
    run_output_dir: Path,
    model=None,
    tools: ToolRegistry | None = None,
) -> TaskRunArtifacts:
    started_at = perf_counter()
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = task_output_dir / CHECKPOINT_FILENAME
    if model is None and tools is None:
        run_result = _run_single_task_with_timeout(
            task_id=task_id, config=config, checkpoint_path=checkpoint_path
        )
    else:
        run_result = _run_single_task_core(
            task_id=task_id,
            config=config,
            model=model,
            tools=tools,
            checkpoint_path=checkpoint_path,
        )
    run_result["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)
    return _write_task_outputs(task_id, run_output_dir, run_result)


def run_benchmark(
    *,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    limit: int | None = None,
    progress_callback: Callable[[TaskRunArtifacts], None] | None = None,
) -> tuple[Path, list[TaskRunArtifacts]]:
    # Resolve run_id: empty → auto-build from params; "{params}" placeholder → expand.
    raw_run_id = config.run.run_id
    if raw_run_id is None:
        resolved_run_id: str | None = build_run_id_from_config(config)
    elif "{params}" in raw_run_id:
        resolved_run_id = raw_run_id.replace("{params}", build_param_suffix(config))
    else:
        resolved_run_id = raw_run_id
    effective_run_id, run_output_dir = create_run_output_dir(config.run.output_dir, run_id=resolved_run_id)

    dataset = DABenchPublicDataset(config.dataset.root_path)
    tasks = dataset.iter_tasks()
    if limit is not None:
        tasks = tasks[:limit]

    effective_workers = config.run.max_workers
    if effective_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    if model is not None or tools is not None:
        effective_workers = 1

    task_ids = [task.task_id for task in tasks]

    task_artifacts: list[TaskRunArtifacts]
    if effective_workers == 1:
        shared_model = model or build_model_adapter(config)
        shared_tools = tools or create_default_tool_registry()
        task_artifacts = []
        for task_id in task_ids:
            artifact = run_single_task(
                task_id=task_id,
                config=config,
                run_output_dir=run_output_dir,
                model=shared_model,
                tools=shared_tools,
            )
            task_artifacts.append(artifact)
            if progress_callback is not None:
                progress_callback(artifact)
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_index = {
                executor.submit(
                    run_single_task,
                    task_id=task_id,
                    config=config,
                    run_output_dir=run_output_dir,
                ): index
                for index, task_id in enumerate(task_ids)
            }
            indexed_artifacts: list[TaskRunArtifacts | None] = [None] * len(task_ids)
            for future in as_completed(future_to_index):
                artifact = future.result()
                indexed_artifacts[future_to_index[future]] = artifact
                if progress_callback is not None:
                    progress_callback(artifact)
            task_artifacts = [artifact for artifact in indexed_artifacts if artifact is not None]

    summary_path = run_output_dir / "summary.json"
    _write_json(
        summary_path,
        {
            "run_id": effective_run_id,
            "task_count": len(task_artifacts),
            "succeeded_task_count": sum(1 for artifact in task_artifacts if artifact.succeeded),
            "max_workers": effective_workers,
            "agent_config": agent_config_snapshot(config),
            "tasks": [artifact.to_dict() for artifact in task_artifacts],
        },
    )
    return run_output_dir, task_artifacts
