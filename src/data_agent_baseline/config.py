from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ENV_MODEL_API_URL = "MODEL_API_URL"
ENV_MODEL_API_KEY = "MODEL_API_KEY"
ENV_MODEL_NAME = "MODEL_NAME"
ENV_MODEL_PROVIDER_ONLY = "MODEL_PROVIDER_ONLY"
ENV_MODEL_SEED = "MODEL_SEED"


def _resolve_str(env_name: str, yaml_value: object, default: str) -> str:
    env_value = os.environ.get(env_name, "").strip()
    if env_value:
        return env_value
    if yaml_value is None:
        return default
    text = str(yaml_value)
    return text if text else default


def _resolve_provider_only(yaml_value: object, default: tuple[str, ...]) -> tuple[str, ...]:
    """Resolve OpenRouter provider allow-list. env > yaml > default. CSV format."""
    env_value = os.environ.get(ENV_MODEL_PROVIDER_ONLY, "").strip()
    if env_value:
        return tuple(item.strip() for item in env_value.split(",") if item.strip())
    if yaml_value is None:
        return default
    if isinstance(yaml_value, (list, tuple)):
        return tuple(str(item).strip() for item in yaml_value if str(item).strip())
    text = str(yaml_value).strip()
    if not text:
        return default
    return tuple(item.strip() for item in text.split(",") if item.strip())


def _resolve_seed(yaml_value: object, default: int | None) -> int | None:
    """Resolve sampling seed. env > yaml > default."""
    env_value = os.environ.get(ENV_MODEL_SEED, "").strip()
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            return default
    if yaml_value is None:
        return default
    try:
        return int(yaml_value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_dataset_root() -> Path:
    return PROJECT_ROOT / "data" / "public" / "input"


def _default_run_output_dir() -> Path:
    return PROJECT_ROOT / "artifacts" / "runs"


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    root_path: Path = field(default_factory=_default_dataset_root)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    model: str = "gpt-4.1-mini"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    max_steps: int = 16
    temperature: float = 0.0
    wall_budget_seconds: float | None = None
    safety_margin_seconds: float = 30.0
    provider_only: tuple[str, ...] = ()
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class RunConfig:
    output_dir: Path = field(default_factory=_default_run_output_dir)
    run_id: str | None = None
    max_workers: int = 4
    task_timeout_seconds: int = 600


@dataclass(frozen=True, slots=True)
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    run: RunConfig = field(default_factory=RunConfig)


def _path_value(raw_value: str | None, default_value: Path) -> Path:
    if not raw_value:
        return default_value
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def load_app_config(config_path: Path) -> AppConfig:
    payload = yaml.safe_load(config_path.read_text()) or {}
    dataset_defaults = DatasetConfig()
    agent_defaults = AgentConfig()
    run_defaults = RunConfig()

    dataset_payload = payload.get("dataset", {})
    agent_payload = payload.get("agent", {})
    run_payload = payload.get("run", {})

    dataset_config = DatasetConfig(
        root_path=_path_value(dataset_payload.get("root_path"), dataset_defaults.root_path),
    )
    raw_wall_budget = agent_payload.get("wall_budget_seconds", agent_defaults.wall_budget_seconds)
    wall_budget_value = float(raw_wall_budget) if raw_wall_budget is not None else None
    provider_only = _resolve_provider_only(agent_payload.get("provider_only"), agent_defaults.provider_only)
    seed_value = _resolve_seed(agent_payload.get("seed"), agent_defaults.seed)
    agent_config = AgentConfig(
        model=_resolve_str(ENV_MODEL_NAME, agent_payload.get("model"), agent_defaults.model),
        api_base=_resolve_str(ENV_MODEL_API_URL, agent_payload.get("api_base"), agent_defaults.api_base),
        api_key=_resolve_str(ENV_MODEL_API_KEY, agent_payload.get("api_key"), agent_defaults.api_key),
        max_steps=int(agent_payload.get("max_steps", agent_defaults.max_steps)),
        temperature=float(agent_payload.get("temperature", agent_defaults.temperature)),
        wall_budget_seconds=wall_budget_value,
        safety_margin_seconds=float(agent_payload.get("safety_margin_seconds", agent_defaults.safety_margin_seconds)),
        provider_only=provider_only,
        seed=seed_value,
    )
    raw_run_id = run_payload.get("run_id")
    run_id = run_defaults.run_id
    if raw_run_id is not None:
        normalized_run_id = str(raw_run_id).strip()
        run_id = normalized_run_id or None

    run_config = RunConfig(
        output_dir=_path_value(run_payload.get("output_dir"), run_defaults.output_dir),
        run_id=run_id,
        max_workers=int(run_payload.get("max_workers", run_defaults.max_workers)),
        task_timeout_seconds=int(run_payload.get("task_timeout_seconds", run_defaults.task_timeout_seconds)),
    )
    return AppConfig(dataset=dataset_config, agent=agent_config, run=run_config)
