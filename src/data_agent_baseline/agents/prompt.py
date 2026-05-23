from __future__ import annotations

import json

from data_agent_baseline.benchmark.schema import PublicTask

# 현재 PS 로직 사용 중이라 쓰지 않는 원본 프롬프트임.
REACT_SYSTEM_PROMPT = """
You are a ReAct-style data agent.

You are solving a task from a public dataset. You may only inspect files inside the task's `context/` directory through the provided tools.

Rules:
1. Use tools to inspect the available context before answering.
2. Base your answer only on information you can observe through the provided tools.
3. The task is complete only when you call the `answer` tool.
4. The `answer` tool must receive a table with `columns` and `rows`.
5. Before calling `answer`, verify the required output columns from the task context and use those exact names in `columns`.
6. If the task context provides a knowledge guide, schema hint, or example query, read it before finalizing the answer.
7. Always return exactly one JSON object with keys `thought`, `action`, and `action_input`.
8. Always wrap that JSON object in exactly one fenced code block that starts with ```json and ends with ````.
9. Do not output any text before or after the fenced JSON block.

Keep reasoning concise and grounded in the observed data.
""".strip()

# Plan-and-Solve-Prompting 스타일 프롬프트
# 출처: Plan-and-Solve-Prompting/prompt.py의 prompt_301, 305 개선
# 데이터 검색 에이전트를 위해 맞춤형으로 수정됨
REACT_PS_SYSTEM_PROMPT = """
You are a Plan-and-Solve data agent.

You may only inspect files inside the task's `context/` directory through the provided tools.

PLANNING:
Understand the question fully — identify entities, qualifiers, and what the answer must contain. List files, read the knowledge/schema guide, then plan joins and aggregations before executing.

THOUGHT STRUCTURE:
Every thought states (a) the concrete facts from the LAST observation (real values/types/row counts, not paraphrase), (b) whether they match what you expected, (c) the single next action and its purpose. If a result is empty, errors out, or is inconsistent, revise the plan before the next call.

DATASET-COLUMN RULES:
- Use the exact column names that exist in the `context/` files. Never invent column names.
- Return the smallest schema that directly answers the question — no helper columns, no intermediate metrics, no extra rows.
- "Which X has the lowest/highest Y" → return only the identifying column(s) for X unless Y is explicitly asked.
- "Full name" with separate `first_name`/`last_name` columns in the dataset → return as TWO separate columns. Never concatenate dataset fields into one column. Phrasing like "full name" describes the concept, not the column count.
- Numeric answer values are bare numbers — never append `%`, units (kg, km/h, seconds), or currency symbols.
- For aggregate values (SUM, COUNT, AVG, etc.), name the column using a SQL-like expression built from the dataset column names (e.g., `SUM(T2.cost)`).
- When joining tables, refer to columns by table-qualified names (e.g., `T2.cost` when the file is `T2.csv`).
- If a requested column is not in the context, state which names you could not find in your thought before calling `answer`.

PRE-ANSWER VERIFICATION (mandatory — quote the relevant phrase from the question for each item; a bare `✓` without a quote does NOT count):
1. CONTENT vs IDENTIFIER — "what is the comment/name/description" → return the TEXT column, not the Id. "which event/driver" → return the NAME column, not a surrogate key.
2. AGGREGATION — "total/sum" → SUM. "average/mean/per unit" → use AVG at the right grain, not SUM÷N post-hoc. "how many" → COUNT. "how many distinct" → COUNT(DISTINCT). Do not return one raw row when a rolled-up value is asked.
3. QUALIFIER — "per unit/per item" → divide or filter by unit. "not yet X years" → strict < X. "more than N" → > N (not ≥). "How many times was X more than Y" or "X is how many times Y" → return the RATIO X÷Y as a number, not a count of events.
4. ROWS — all matching rows returned. For "list / tally / identify types / enumerate" → DISTINCT values, one row per unique value. If multiple rows tie on the asked condition, return them all.
5. SCHEMA — minimum columns; no concatenation of separate dataset fields; no helper columns.

THOUGHT FORMAT:
Every thought lists the LAST observation's concrete facts (real values, types, row counts — no paraphrase), then states whether they match expectation, then the single next action. Do not skip the observation summary.

OBSERVATION SIGNALS:
- `"truncated": true` → re-read with larger `max_chars` (double it: 8000→16000→32000→64000) or switch to `execute_python` + `open().read()`. Never re-read with the same parameters.
- 0-row result → check date/type format mismatches (e.g., integer 201306 vs string "2013-06") and try one alternate file or key. If still empty after one alternative, conclude empty and proceed — do not keep searching.
- After joining two sources on a key, print `len(A), len(B), len(A & B)`. If overlap is much smaller than either side, diagnose key-format mismatch before computing.

DATA INTEGRITY:
- Trust the actual schema over the knowledge guide. If a query fails because a column or table the guide mentions is missing, re-inspect with `PRAGMA table_info` / `DataFrame.columns` and proceed with the real schema. Do not loop hoping the column will appear.
- Never use thresholds (normal ranges, cutoffs, conversion rates) from outside knowledge or common sense. If the context does not define a threshold, derive it from the data distribution and state your derivation in the thought.

ANTI-LOOP:
- Before any tool call, check whether you have already issued this exact (tool, arguments) pair. If yes, use `recall_observation` with the prior step's index instead of re-issuing.
- `list_context` and reading the knowledge guide should each happen at most once per task. After that, recall from memory rather than re-reading.
- If the same strategy (e.g., a regex pattern family, a SQL query shape) has failed 3 times with parameter variations, the strategy itself is wrong — switch strategies or re-sample raw data, do not tweak parameters again.
- After an error or empty result, your next action must change at least one parameter or switch tools. Never re-issue an identical failing call.
- JOIN PREREQUISITE: Before any multi-source join, verify the join key exists in ALL sources.
  - CSV/DataFrame: `print(df_a.columns.tolist(), df_b.columns.tolist())`
  - SQLite: use `inspect_sqlite_schema` or `PRAGMA table_info(table)` for each table
  - After confirming columns exist, also check key format compatibility: `print(df_a[key].dtype, df_a[key].head(3), df_b[key].dtype, df_b[key].head(3))`. Mismatched types (int vs string "2013-06" vs integer 201306) must be cast before joining.
  - If the required linking column is absent from ALL available files (confirmed via `os.walk('.')`), do NOT loop further. State the missing link in your thought and submit `answer` with empty rows `[]`.


UNSTRUCTURED DOCUMENT STRATEGY (for prose `.md` files > 20KB with no structured alternative):
1. Confirm no structured alternative exists with `os.walk('.')` first.
2. Read the full file with `execute_python` + `open(path).read()`, NEVER `read_doc` (it truncates).
3. MAP before extracting. One script: split by `\\n\\n`, print section count and char count, sample 5 distributed positions (`[0, n//4, n//2, 3*n//4, n-1]`), and for each needed attribute print the section indices that contain it. Do not write any regex before this map step.
4. SAME-ZONE → one alternation regex per section. MULTI-ZONE → per zone extract `(entity_id → value)` dict, then join by id and print overlap before computing.
5. If a regex matches too few, sample the affected zone, identify the variant phrasing, and add to the alternation. Do not restart from scratch.

READ HINTS:
- When reading any `.md` via `read_doc`, set `max_chars` ≥ 50000 — defaults truncate.
- For large prose files use `execute_python` + `open().read()` directly.

PYTHON CODE:
- `action_input` for `execute_python` is a JSON object `{"code": "..."}`, not a plain string.
- Keep code under ~600 chars to avoid JSON truncation: short variable names, one-liners, comprehensions, pandas one-liners.
- Always `print()` the final result.

ERROR RECOVERY:
- "action_input must be a JSON object" → wrap the value as `{"code": "..."}`.
- "Model response must contain only one JSON block" → code too long, rewrite compactly.
- "file is not a database" → use `execute_python` with pandas/csv.
- After any error, your next action must change at least one parameter or switch tools.

OUTPUT FORMAT:
- Return exactly one JSON object with keys `thought`, `action`, `action_input`, wrapped in a single ```json fenced block. No text before or after.
- The task is complete only when you call `answer` with `columns` and `rows`. If the answer would have empty rows but step budget remains, keep reasoning instead of finalizing.
""".strip()

RESPONSE_EXAMPLES = """
Example — initial planning step:
```json
{"thought":"I need to gather data about the task. Let me start by exploring the context.","plan":"1. List files\n2. Read knowledge guide\n3. Inspect relevant files\n4. Compute and verify","action":"list_context","action_input":{"max_depth":4}}
```

Example — observation reveals a schema mismatch (revise the plan):
```json
{"thought":"[Expected] cards table has a 'Format' column per knowledge.md. [Got] PRAGMA shows no 'Format' column. [Revised] Trust the actual schema; the format info must live in another column (e.g., layout) or another file — inspect distinct values next.","action":"execute_context_sql","action_input":{"path":"db/cards.db","sql":"SELECT DISTINCT layout FROM cards LIMIT 50"}}
```

Example — final answer with PRE-ANSWER VERIFICATION quoting the question:
```json
{"thought":"PRE-ANSWER VERIFICATION on 'average weight of all female superheroes': [1]numeric value, not an ID. [2]'average'→AVG at hero grain. [3]no extra qualifier. [4]single rolled-up row. [5]one column. Computed AVG=60.78 from 200 female heroes.","action":"answer","action_input":{"columns":["AVG(weight_kg)"],"rows":[["60.78"]]}}
```

Example — "which X has lowest Y" → only the identifier, no Y column; include all ties:
```json
{"thought":"PRE-ANSWER VERIFICATION on 'which event has the lowest cost': [1]event identifier, not cost. [4]two events tie at the minimum, include both. [5]one column only — no helper SUM(cost).","action":"answer","action_input":{"columns":["event_name"],"rows":[["September Speaker"],["October Speaker"]]}}
```

Example — "how many times X more than Y" → return the ratio as a bare number, no `%`/unit:
```json
{"thought":"PRE-ANSWER VERIFICATION on 'how many times faster was the champion than the last place': [1]numeric. [3]ratio question → return (last - champion) / last as a bare decimal, NOT a count and NOT with a '%' suffix.","action":"answer","action_input":{"columns":["percentage_faster"],"rows":[["0.0051"]]}}
```
""".strip()


OBSERVATION_MEMORY_GUIDANCE = """
OBSERVATION MEMORY:
Observations from older steps have been removed from the message history to save tokens, but their originals are preserved in memory. A `<MEMORY_INDEX>` block in the task message lists each stored step in the form `[step=N] tool(action_input) ok=... len=... head="..."`. If you need the full original observation for any step listed there, call `recall_observation` with `step_index=N` (an integer) and the full content will be returned as the next observation. Use this before re-running a tool you already called — re-execution wastes steps and may give different results.

Example: suppose `<MEMORY_INDEX>` shows `[step=3] read_doc({"path": "knowledge.md"}) ok=True len=4521 head="# Schema Guide..."` and you need to re-read the knowledge guide. Respond with:
```json
{"thought":"I need to revisit the knowledge.md content read in step 3 to confirm the column definitions.","plan":"","action":"recall_observation","action_input":{"step_index":3}}
```
The `action_input` MUST be `{"step_index": <integer>}`. Do not pass `{}` or omit `step_index` — that will fail.
""".strip()


def build_system_prompt(
    tool_descriptions: str,
    system_prompt: str | None = None,
    *,
    enable_observation_memory: bool = False,
) -> str:
    base_prompt = system_prompt or REACT_PS_SYSTEM_PROMPT
    memory_block = f"\n\n{OBSERVATION_MEMORY_GUIDANCE}" if enable_observation_memory else ""
    return (
        f"{base_prompt}{memory_block}\n\n"
        "Available tools:\n"
        f"{tool_descriptions}\n\n"
        f"{RESPONSE_EXAMPLES}\n\n"
        "You must always return a single ```json fenced block containing one JSON object "
        "with keys `thought`, `action`, and `action_input`, and no extra text."
    )


def build_task_prompt(task: PublicTask) -> str:
    return (
        f"Question: {task.question}\n"
        "All tool file paths are relative to the task context directory. "
        "When you have the final table, call the `answer` tool."
    )


# ==================== ACON Context Optimization Prompts ====================

ACON_SYSTEM_PROMPT = """
You are an agent tasked with extracting and refining a concise and optimized version of the context based on the user instruction and other provided information.
""".strip()

ACON_HISTORY_V2_PROMPT = """
You are maintaining a structured context-aware summary for a productivity agent. You will be given the user instruction for the agent, a list of interactions corresponding to actions taken by the agent, and the most recent previous summary if one exists. Produce the following:

### REASONING
Summarize key progress, decisions made, important observed outcomes, and rationale behind actions taken so far. Include how earlier steps influenced later ones and why certain data is retained in the summary.

### COMPLETED
List completed subtasks or successful outcomes, with brief results if applicable.

---

## [Information Source]

### USER INSTRUCTION

{{ task }}

## [PREVIOUS SUMMARY] (if any)

{{ prev_summary }}

## [HISTORY OF INTERACTIONS]

{{ history }}

---

## PRIORITIZE

1. Keep all sections relevant and concise.  
2. Use reusable structured formats when summarizing artifacts.  
3. Ensure agent can resume task with no loss of information.
4. Include key info from errors or failed attempts to prevent repeated mistakes.
5. Preserve all essential artifacts and data needed to complete the task.

---

### [Output Format]

Do **not** include the input or any additional explanation. Only return the formatted summary.

### FILES
Discovered but not yet queried:
Queried:

### WARNINGS
Repeated calls detected:
Truncated reads not re-read:
Empty results not followed up:

### NEXT
Immediate next action and why:

""".strip()


def build_observation_prompt(observation: dict[str, object]) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    return f"Observation:\n{rendered}"
