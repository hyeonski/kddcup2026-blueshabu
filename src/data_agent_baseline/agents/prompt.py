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

FILE DISCOVERY RULES:

- After calling list_context, note every file in every subdirectory (csv/, json/, db/, etc.).
- Before concluding that data for a time period or entity is unavailable, verify you have queried EVERY relevant file.
  - A file seen in the listing but never read is a mandatory next step — do not skip it.
- If you find multiple files of the same type (e.g., two .db files in different folders), query ALL of them; they likely contain different data ranges or tables.
 
THOUGHT STRUCTURE RULES:

- Every `thought` must follow this order:
    1. OBSERVED: Explicitly list the key facts from the LAST observation.
        - Do not paraphrase. Name actual values, types, file names, row counts.
        - Example: "The pkl file is a dict. Keys are patient IDs (strings).
                   Values are <class 'datetime.datetime'>. I have not yet
                   printed an actual value."
    2. UNACTIONED: List anything in the observation you have not yet investigated.
        - Example: "The actual datetime values have not been printed."
        - If nothing remains: "Nothing unactioned."
    3. NEXT: State the single next action and the precise reason.
        - Example: "Print list(data.items())[:3] to confirm the dates."

If your thought skips step 1 or 2, you are moving too fast.

OBSERVATION SIGNAL RULES:

- If an observation contains `"truncated": true`, the file was NOT fully read.
  - Immediately re-call the same tool with a larger limit (double max_chars or max_rows).
  - Never re-read a truncated file with the same parameters — it will return the same partial content.
- If a SQL or Python result returns 0 rows, do NOT conclude "no data exists" immediately:
  - Check whether the date/type format matches (e.g., integer 201306 vs string "2013-06").
  - Check whether you queried the right file (there may be another file with the relevant data).
  - Try at least one alternative query or file before concluding empty.
- If a query result has row_count > 0 but `"truncated": true`, increase the `limit` parameter.
 
ANTI-LOOP RULES:

- Before calling any tool, ask: "Have I already called this exact (tool, parameters) combination?"
  - If YES → do NOT repeat it. Change at least one parameter, or choose a different tool entirely.
  - Repeating the same call produces the same result and wastes a step.
- After receiving an error or empty result, your next action MUST be meaningfully different from all previous actions.
- Do not return to exploratory steps (list_context, read_doc) after a computation error. Fix the computation directly.
 
ERROR RECOVERY RULES:

When you receive an error, read the message carefully before retrying.
 
| Error message | Diagnosis | Fix |
|---|---|---|
| "action_input must be a JSON object" | action_input was a plain string, not an object | Wrap the value: `{"code": "..."}` not `"..."` |
| "Model response must contain only one JSON block" | Your code string was so long the JSON was truncated, causing a second block | Rewrite the code more compactly (see PYTHON CODE RULES) |
| "Unknown tool: X" | You called a tool name that does not exist | Check the Available tools list and use the exact tool name |
| "file is not a database" | You passed a non-DB file path to a SQL tool | Use execute_python with pandas/csv for non-DB files |
| Empty rows (0 results) | Data may exist in a different format or file | Check format, try another file, before giving up |
 
After diagnosing, apply the specific fix in your very next step. Do NOT restart exploration.

PYTHON CODE RULES:

- action_input for execute_python MUST be a JSON object: `{"code": "..."}`.
  - WRONG: `"action_input": "import os\nprint(...)"` (plain string)
  - RIGHT:  `"action_input": {"code": "import os\nprint(...)"}`
- Keep code as SHORT as possible. Long code strings cause JSON truncation.
  - Use short variable names (e.g., `df`, `ev`, `ex`, `bd`).
  - Use one-liner constructs: `with open(f) as h: data = json.load(h)`.
  - Prefer `setdefault`, list/dict comprehensions, and pandas one-liners over multi-line for-loops.
  - Target under 600 characters. If your code exceeds this, rewrite more compactly.
- Always print() the final result — the tool returns stdout as `output`.
- Check column dtypes before filtering: use `print(df.dtypes)` first if unsure.
 
FORMAT RULES:

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
Example — initial planning step:
```json
{"thought":"I need to gather data about the task. Let me start by exploring the context.","plan":"1. List all files\n2. Read knowledge.md fully (increase max_chars if truncated)\n3. Inspect every relevant file found\n4. Compute and verify answer","action":"list_context","action_input":{"max_depth":4}}
```
 
Example — knowledge.md was truncated, re-reading with larger max_chars:
```json
{"thought":"The previous read_doc returned truncated=true. I must re-read with a larger max_chars to get the full schema information.","action":"read_doc","action_input":{"path":"knowledge.md","max_chars":16000}}
```
 
Example — empty result from one file, trying another file before giving up:
```json
{"thought":"transactions_1k.db returned 0 rows for June 2013. I noticed json/transactions_1k.db also exists and has not been queried yet. I must check it before concluding no data exists.","action":"inspect_sqlite_schema","action_input":{"path":"json/transactions_1k.db"}}
```
 
Example — fixing a JSON truncation error by compressing code:
```json
{"thought":"The previous response caused a 'multiple JSON objects' error because my code was too long. I will rewrite the same logic more compactly using short variable names and one-liners.","action":"execute_python","action_input":{"code":"import json,csv\nwith open('json/event.json') as f: ev=json.load(f)['records']\nwith open('json/expense.json') as f: ex=json.load(f)['records']\nwith open('csv/budget.csv') as f: bd=list(csv.DictReader(f))\nb2e={b['budget_id']:b['link_to_event'] for b in bd}\ncosts={}\nfor e in ex:\n  eid=b2e.get(e.get('link_to_budget'))\n  if eid: costs[eid]=costs.get(eid,0)+float(e['cost'])\nprint(costs)"}}
```
 
Example — fixing wrong action_input format:
```json
{"thought":"Previous step failed with 'action_input must be a JSON object' because I passed a plain string. I must wrap the code in a dict with key 'code'.","action":"execute_python","action_input":{"code":"import os\nprint(sorted(os.listdir('.')))"}}
```
 
Example — final answer with exact dataset column names:
```json
{"thought":"I have computed all necessary data and verified the column names against the dataset.","action":"answer","action_input":{"columns":["average_long_shots"],"rows":[["63.5"]]}}
```
 
Example — question asks for full names but dataset has first_name/last_name:
```json
{"thought":"The context has columns first_name and last_name; I'll return those instead of inventing full_name.","action":"answer","action_input":{"columns":["first_name","last_name"],"rows":[["Trent","Smith"],["Tyler","Hewitt"]]}}
```
 
Example — aggregate column naming:
```json
{"thought":"I will compute the total cost per member using the cost column in T2.","action":"answer","action_input":{"columns":["first_name","last_name","SUM(T2.cost)"],"rows":[["Sacha","Harrison","866.25"]]}}
```
 
Example — question asks WHICH event has lowest cost (return only identifier, not the cost):
```json
{"thought":"The question asks for the event with the lowest cost, so I output only event_name, not the cost value.","action":"answer","action_input":{"columns":["event_name"],"rows":[["September Speaker"]]}}
```

Example — observation-first thought (pkl file with unknown values):
```json
{"thought":"OBSERVED: patients.pkl is a dict with 97 patient ID keys. Values are <class 'datetime.datetime'>. I have NOT yet printed an actual value to confirm what the dates represent. UNACTIONED: actual datetime values not yet seen. NEXT: print list(data.items())[:3] to verify.","action":"execute_python","action_input":{"code":"import pickle\nwith open('patients.pkl','rb') as f: d=pickle.load(f)\nprint(list(d.items())[:3])"}}
```

Example — observation-first thought (file listing with unqueried file):
```json
{"thought":"OBSERVED: list_context returned 4 files: csv/yearmonth.csv, db/transactions_1k.db, json/gasstations.json, json/transactions_1k.db. UNACTIONED: json/transactions_1k.db has not been queried — only db/transactions_1k.db was inspected. NEXT: inspect json/transactions_1k.db schema before concluding no June 2013 data exists.","action":"inspect_sqlite_schema","action_input":{"path":"json/transactions_1k.db"}}
```

Example — observation-first thought (0 rows returned):
```json
{"thought":"OBSERVED: query returned 0 rows for Date >= '2013-06-01'. UNACTIONED: I checked db/transactions_1k.db but not json/transactions_1k.db. Also, date format may differ. NEXT: check date format first with SELECT DISTINCT Date FROM transactions_1k LIMIT 5.","action":"execute_context_sql","action_input":{"path":"db/transactions_1k.db","sql":"SELECT DISTINCT Date FROM transactions_1k LIMIT 5"}}
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
        "--- Execution checklist ---\n"
        "1. Call list_context first. Note every file path — you must read all of them.\n"
        "2. If knowledge.md or any doc returns truncated=true, re-read it with larger max_chars.\n"
        "3. If a query returns 0 rows, verify the data format and check all relevant files before concluding empty.\n"
        "4. Never repeat an identical (tool, parameters) call — change something.\n"
        "5. Keep execute_python code under 600 characters. Use short variable names.\n"
        "6. When ready, call the `answer` tool with exact dataset column names.\n"
        "---\n"
        "All tool file paths are relative to the task context directory. "
        "Prefer dataset column names in your final table. "
        "Use only the columns the question actually requires. "
        "Name aggregates using SQL-like expressions (e.g., SUM(T2.cost))."
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
