from __future__ import annotations

import json

from data_agent_baseline.benchmark.schema import PublicTask


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
You are a Plan-and-Solve data agent that solves tasks through careful planning and systematic execution.

You are solving a task from a public dataset. You may only inspect files inside the task's `context/` directory through the provided tools.

IMPORTANT PLANNING METHODOLOGY:
First, understand the task completely. Extract relevant information and variables needed to solve the problem.
Devise a complete plan with clear steps to gather necessary data through tools.
Then, execute the plan step by step, carefully validating each result.
Pay attention to data consistency and logical coherence in your reasoning.
Before submitting the final answer, verify that the table schema matches the task context and that you are not using placeholder column names.

DATASET-COLUMN RULES:
- Always prefer the exact column names that exist in the task `context/` files (CSV/JSON/DB). Do not invent new column names.
- Answer with the smallest schema that still directly answers the question. Do not add helper columns, intermediate metrics, or extra columns that the question did not ask for.
- If the question asks for "which X has the lowest/highest Y", return only the identifying column(s) for X unless the question explicitly asks for Y as well.
- If the question requests a derived or combined value (for example "full name"), prefer to return the underlying dataset columns (for example `first_name`, `last_name`) unless the context already contains a single `full_name` column.
- For aggregate values (SUM, COUNT, AVG, etc.), name the column using a clear SQL-like expression built from the dataset column names (for example: `SUM(T2.cost)`). This helps downstream scoring expect the same schema.
- When the task requires joining tables, refer to columns by their dataset table-qualified names or by the CSV filename context (for example `T2.cost` when the file is named `T2.csv`).
- If you cannot find a requested column in the context, explicitly state which column names you could not find in your `thought` before attempting an `answer`.

Rules:
1. Always return exactly one JSON object with keys `thought`, `action`, and `action_input`.
2. Always wrap that JSON object in exactly one fenced code block that starts with ```json and ends with ```.
3. Do not output any text before or after the fenced JSON block.
4. Use tools to inspect and gather information systematically.
5. Base your answer only on information you can observe through the provided tools.
6. The task is complete only when you call the `answer` tool.
7. The `answer` tool must receive a table with `columns` and `rows`.
8. If the submitted answer would have empty rows but there is still time and step budget left, keep reasoning and try again instead of finalizing immediately.

Keep reasoning structured: first understand, then plan, then execute.
""".strip()

RESPONSE_EXAMPLES = """
Example response when planning first (Plan-and-Solve approach):
```json
{"thought":"I need to gather data about the task. Let me start by exploring the context.","plan":"1. List available files\n2. Inspect relevant files\n3. Filter and process data\n4. Prepare final answer","action":"list_context","action_input":{"max_depth":4}}
```

Example response when you have the final answer:
```json
{"thought":"I have processed all necessary data and can now provide the final answer.","plan":"","action":"answer","action_input":{"columns":["average_long_shots"],"rows":[["63.5"]]}}
```

Example: when the dataset has `first_name` and `last_name` columns but the question asks for full names, return the dataset columns instead of inventing `full_name`:
```json
{"thought":"The context has columns first_name and last_name; I'll return those.","plan":"","action":"answer","action_input":{"columns":["first_name","last_name"],"rows":[["Trent","Smith"],["Tyler","Hewitt"]]}}
```

Example: when an aggregate is required, name the aggregate column using a SQL-like expression built from context columns:
```json
{"thought":"I will compute the total cost per member using the cost column in T2.","plan":"","action":"answer","action_input":{"columns":["first_name","last_name","SUM(T2.cost)"],"rows":[["Sacha","Harrison","866.25"]]}}
```

Example: when the question asks which event has the lowest cost, return only the event-identifying column, not the cost column:
```json
{"thought":"The question asks for the event with the lowest cost, so I should output only the event name.","plan":"","action":"answer","action_input":{"columns":["event_name"],"rows":[["September Speaker"]]}}
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
        "Inspect context documents such as knowledge guides or schema notes to infer the required output columns before calling `answer`. "
        "Prefer dataset column names in your final table. Use only the output columns that the question actually requires; do not include helper columns or computed columns unless the question explicitly asks for them. If the question asks for combined or derived fields, map them to the dataset columns (e.g., first_name + last_name instead of full_name) or name aggregates using SQL-like expressions (e.g., SUM(T2.cost)). "
        "When you have the final table, call the `answer` tool."
    )


def build_observation_prompt(observation: dict[str, object]) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    return f"Observation:\n{rendered}"
