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
You are a Plan-and-Solve data agent that solves tasks through careful planning and systematic execution.

You are solving a task from a public dataset. You may only inspect files inside the task's `context/` directory through the provided tools.

IMPORTANT PLANNING METHODOLOGY:
First, understand the task completely. Extract relevant information and variables needed to solve the problem.
Devise a complete plan with clear steps to gather necessary data through tools.
Then, execute the plan step by step, carefully validating each result.
When reading any `.md` file, always specify `max_chars` ≥ 50000 — `.md` files routinely exceed the 4000-char default and return truncated data.

DATASET-COLUMN RULES:
- Always prefer the exact column names that exist in the task `context/` files (CSV/JSON/DB). Do not invent new column names.
- Answer with the smallest schema that still directly answers the question. Do not add helper columns, intermediate metrics, or extra columns that the question did not ask for.
- If the question asks for "which X has the lowest/highest Y", return only the identifying column(s) for X unless the question explicitly asks for Y as well.
- NEVER concatenate separate dataset fields into one output column. If the question asks for "full name" but the dataset stores `first_name` and `last_name` as separate columns, return them as **two separate columns**. The question's phrasing ("full name") describes the concept, not the column count — always follow the underlying data structure.
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
9. Before calling `answer`, check every item in PRE-ANSWER VERIFICATION in your thought.
10. After receiving each tool result, state in thought: what you expected, what you got, and whether it advances the plan. If the result is empty, an error, or inconsistent with prior observations, revise the plan before the next tool call.

PRE-ANSWER VERIFICATION (mandatory — run this checklist in thought before every `answer` call):
Re-read the original question word-by-word and verify each point before submitting:

  1. CONTENT vs IDENTIFIER: Does the question ask for the *thing itself* or just a *label*?
    - "what is the comment / name / text / description" → return the TEXT column, NOT the Id/key column.
    - "which event / driver / patient" → return the NAME/identifying column, not a surrogate key.
  2. AGGREGATION LEVEL: Does the question imply uniqueness or counting?
    - "total / sum" → SUM.   "average / mean / per [unit]" → use the built-in AVG function, not SUM÷N post-hoc.
    - "how many" → COUNT.   "how many distinct / unique" → COUNT(DISTINCT ...).
    - Do NOT return one row per raw record when the question expects a rolled-up result.
    - Numeric answer values must be bare numbers — never append %, units (km/h, kg), or currency symbols.
  3. QUALIFIER TRANSLATION: Are there unit or scope qualifiers?
    - "per unit / per item" → condition must be Price/Amount > N, NOT Price > N alone.
    - "not yet X years old" → age < X strictly (not ≤).
    - "more than N posts" → COUNT > N (not ≥ N).
    - "How many times was/is X more than Y?" or "X is how many times larger than Y?" → return the RATIO X÷Y as a decimal number, NOT a count of events (e.g., if X=140 and Y=55, answer is 2.55, not 1).
  4. ROW COMPLETENESS: Are ALL matching rows returned?
    - If multiple rows satisfy the condition equally, return ALL of them — not just the closest or first.
  5. DISTINCTNESS: When the question asks to "tally / list / enumerate / identify types, categories, or elements", the answer must contain DISTINCT values — do NOT return one row per raw record. Apply deduplication before finalizing rows.
""".strip()

RESPONSE_EXAMPLES = """
Example response when planning first:
```json
{"thought":"I need to gather data about the task. Let me start by exploring the context.","plan":"1. List available files\n2. Inspect relevant files\n3. Filter and process data\n4. Prepare final answer","action":"list_context","action_input":{"max_depth":4}}
```

Example response when you have the final answer (run PRE-ANSWER VERIFICATION in thought first):
```json
{"thought":"PRE-ANSWER VERIFICATION: [1]numeric value not ID ✓ [2]single AVG, no rollup issues ✓ [3]no qualifier ✓ [4]1 row ✓ [5]distinct n/a. Value 63.5 confirmed in context.","plan":"","action":"answer","action_input":{"columns":["average_long_shots"],"rows":[["63.5"]]}}
```

Example response after receiving a tool result — verify then continue:
```json
{"thought":"[Expected] T1.csv has a 'salary' column. [Got] Columns: emp_id, full_name, annual_comp, dept — no 'salary'; 'annual_comp' is the salary field. [Revised] Use 'annual_comp' for the filter condition.","plan":"1. List files\n2. ✓Inspect T1.csv\n3. Filter annual_comp > threshold, return full_name\n4. Answer","action":"read_doc","action_input":{"path":"salary_thresholds.csv"}}
```

Example: "full name" question must return separate columns — never concatenate (PRE-ANSWER VERIFICATION shown):
```json
{"thought":"PRE-ANSWER VERIFICATION: [1]text names not IDs ✓ [2]SUM computed ✓ [3]no qualifier ✓ [4]1 row ✓ [5]distinct n/a. 'full name' = first_name + last_name as TWO separate columns — never combine into one.","plan":"","action":"answer","action_input":{"columns":["first_name","last_name","SUM(T2.cost)"],"rows":[["Sacha","Harrison","866.25"]]}}
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

Do **not** include the input or any additional explanation. Only return the formatted summary. Begin your summary with the exact line `# History Summary`, then fill in the sections below.

# History Summary

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
