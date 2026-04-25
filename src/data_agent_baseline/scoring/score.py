"""Local scorer mirroring the official leaderboard formula.

Score = max(0, Recall - lambda * (Extra Columns / Predicted Columns))
where:
- Recall = Matched Columns / Gold Columns
- Matched columns are gold/pred column pairs whose normalized, sorted value
  signatures are identical.
- Column order is irrelevant; matching is content-based.

Each gold column is matched to at most one prediction column. Greedy unique
pairing is sufficient because signatures are exact tuples — equal signatures
are interchangeable.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from data_agent_baseline.scoring.normalize import normalize_column_values

DEFAULT_LAMBDA = 0.5


@dataclass(frozen=True, slots=True)
class TaskScore:
    task_id: str
    recall: float
    extras_ratio: float
    score: float
    matched_columns: int
    gold_columns: int
    predicted_columns: int
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RunScore:
    total: float
    task_scores: list[TaskScore] = field(default_factory=list)


def _load_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _column_signatures(header: list[str], rows: list[list[str]]) -> list[tuple[str, ...]]:
    signatures: list[tuple[str, ...]] = []
    for index in range(len(header)):
        column_values = [row[index] if index < len(row) else "" for row in rows]
        normalized = normalize_column_values(column_values)
        signatures.append(tuple(sorted(normalized)))
    return signatures


def score_task(
    *,
    task_id: str,
    prediction_path: Path | None,
    gold_path: Path,
    lambda_: float = DEFAULT_LAMBDA,
) -> TaskScore:
    if not gold_path.exists():
        return TaskScore(
            task_id=task_id,
            recall=0.0,
            extras_ratio=0.0,
            score=0.0,
            matched_columns=0,
            gold_columns=0,
            predicted_columns=0,
            error=f"gold csv missing: {gold_path}",
        )
    gold_header, gold_rows = _load_csv(gold_path)
    gold_sigs = _column_signatures(gold_header, gold_rows)
    gold_count = len(gold_sigs)

    if prediction_path is None or not prediction_path.exists():
        return TaskScore(
            task_id=task_id,
            recall=0.0,
            extras_ratio=0.0,
            score=0.0,
            matched_columns=0,
            gold_columns=gold_count,
            predicted_columns=0,
            error="prediction missing",
        )

    pred_header, pred_rows = _load_csv(prediction_path)
    pred_sigs = _column_signatures(pred_header, pred_rows)
    pred_count = len(pred_sigs)

    used_pred: set[int] = set()
    matched = 0
    for gold_sig in gold_sigs:
        for index, pred_sig in enumerate(pred_sigs):
            if index in used_pred:
                continue
            if pred_sig == gold_sig:
                used_pred.add(index)
                matched += 1
                break

    recall = matched / gold_count if gold_count else 0.0
    extras_ratio = (pred_count - matched) / pred_count if pred_count else 0.0
    score = max(0.0, recall - lambda_ * extras_ratio)
    return TaskScore(
        task_id=task_id,
        recall=recall,
        extras_ratio=extras_ratio,
        score=score,
        matched_columns=matched,
        gold_columns=gold_count,
        predicted_columns=pred_count,
    )


def score_run(
    *,
    run_dir: Path,
    gold_root: Path,
    lambda_: float = DEFAULT_LAMBDA,
) -> RunScore:
    task_scores: list[TaskScore] = []
    task_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("task_"))
    for task_dir in task_dirs:
        task_id = task_dir.name
        prediction_path = task_dir / "prediction.csv"
        if not prediction_path.exists():
            prediction_path = None
        gold_path = gold_root / task_id / "gold.csv"
        task_scores.append(
            score_task(
                task_id=task_id,
                prediction_path=prediction_path,
                gold_path=gold_path,
                lambda_=lambda_,
            )
        )
    if task_scores:
        total = sum(s.score for s in task_scores) / len(task_scores)
    else:
        total = 0.0
    return RunScore(total=total, task_scores=task_scores)
