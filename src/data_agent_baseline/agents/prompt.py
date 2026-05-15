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
- If the list_context observation ends mid-item (cut off mid-sentence or mid-entry), the listing is INCOMPLETE.
  - Re-run list_context or confirm all files with: `execute_python` → `import os; print([os.path.join(r,f) for r,d,fs in os.walk('.') for f in fs if not f.endswith('Identifier')])`
  - A hidden structured file (JSON/DB/CSV) may be missing from your view if you don't verify.
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
  - Check the file's size (bytes) from the list_context output. Set max_chars = size + 500 to read it in one shot.
  - If size was not recorded, double max_chars on each retry: 8000 → 16000 → 32000 → 64000.
  - Keep retrying until truncated=false. NEVER give up on knowledge.md — it contains critical thresholds.
  - Never re-read a truncated file with the same parameters — it will return the same partial content.
- If a SQL or Python result returns 0 rows, do NOT conclude "no data exists" immediately:
  - Check whether the date/type format matches (e.g., integer 201306 vs string "2013-06").
  - Check whether you queried the right file (there may be another file with the relevant data).
  - Try at least one alternative query or file before concluding empty.
- If a query result has row_count > 0 but `"truncated": true`, increase the `limit` parameter.
- JOIN COMPLETENESS CHECK: When joining two extracted data sources on a common key
  (e.g., entity ID, patient ID), always verify the overlap count before computing a result:
    print(f"A: {len(A)}, B: {len(B)}, Joined: {len(set(A) & set(B))}")
  If the joined count is much smaller than either source (e.g., A=64, B=58, Joined=3),
  the extraction is INCOMPLETE — do NOT compute a final answer yet.
  Instead, print sample unmatched keys from both sides to diagnose why IDs don't align.
 
ANTI-LOOP RULES:

- Before calling any tool, ask: "Have I already called this exact (tool, parameters) combination?"
  - If YES → do NOT repeat it. Change at least one parameter, or choose a different tool entirely.
  - Repeating the same call produces the same result and wastes a step.
- After receiving an error or empty result, your next action MUST be meaningfully different from all previous actions.
- Do not return to exploratory steps (list_context, read_doc) after a computation error. Fix the computation directly.
- STRATEGY-LEVEL LOOP: If the same approach type (e.g., "regex extraction from prose") has failed 3+ times with different variations, STOP. The approach itself is wrong — not just the parameters. Switch to a completely different strategy (e.g., different section delimiter, print raw samples to re-examine the format from scratch).

UNSTRUCTURED DOCUMENT STRATEGY:
When the context contains a large prose markdown file (no structured CSV/JSON/DB):

1. CONFIRM NO STRUCTURED FILE EXISTS FIRST:
   Run `os.walk('.')` to list all files. Only fall back to prose parsing after confirming
   there is truly no CSV/JSON/DB alternative.

2. USE execute_python + open() — NEVER read_doc FOR FILES > 20KB:
   - read_doc has a character limit and will always truncate large prose files.
   - `open('doc/file.md').read()` has no such limit and reads the full file into memory.
   - Use read_doc only for quick < 5KB inspection; use open() for all programmatic processing.

3. UNDERSTAND FORMAT BEFORE EXTRACTING:
   In the first Python call, ALWAYS run this exact sequence:
   a. `parts = content.split('\n\n'); print(len(parts))`
      - If len(parts) > 10 → each paragraph is one entity. Use '\n\n' as your delimiter. Do NOT search for a fancier split.
      - If len(parts) ≤ 10 → the file is structured differently; try a more specific delimiter.
   b. Print 2–3 representative entity paragraphs (skip index 0, which is usually the title/header):
        `[print(p[:300]) for p in parts[1:4]]`
   c. Identify the EXACT phrases used (e.g., "height is recorded as X.0 centimeters",
      "publisher 13"). Do NOT write any regex before seeing these raw samples.

4. EXTRACT ALL IN ONE COMPREHENSIVE SCRIPT:
   Once the format is confirmed (step 3), write ONE Python script that:
   a. Reads the full file with open() — treat this as the last open() of this file.
   b. Splits using the delimiter confirmed in step 3a (usually '\n\n').
   c. For each entity section, tries MULTIPLE attribute patterns at once using alternation:
        `re.search(r'pattern_A|pattern_B|pattern_C', section)`
      Do NOT write separate scripts for pattern_A, pattern_B, pattern_C — try all in one pass.
   d. Collects ALL (attr1, attr2, …) tuples and computes the final answer in the same script.
   If the result is wrong or incomplete, add more pattern variants to THIS script in the next call.
   Do NOT iterate: open file → try regex_A → open file again → try regex_B → open file again → …

5. WHEN REGEX RETURNS 0 MATCHES:
   - Print raw text of 2–3 sections where you expected a match.
   - Confirm the delimiter actually splits the document correctly (check len(parts)).
   - Add the observed new phrase variant to your pattern list, then re-run comprehensively.

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

PRE-ANSWER VERIFICATION (mandatory — run this checklist in thought before every `answer` call):

Re-read the original question word-by-word and verify each point before submitting:

1. CONTENT vs IDENTIFIER: Does the question ask for the *thing itself* or just a *label*?
   - "what is the comment / name / text / description" → return the TEXT column, NOT the Id/key column.
   - "which event / driver / patient" → return the NAME/identifying column, not a surrogate key.
2. AGGREGATION LEVEL: Does the question imply uniqueness or counting?
   - "tally / list the elements / how many distinct" → apply DISTINCT or GROUP BY before submitting.
   - Do NOT return one row per raw record when the question expects a rolled-up result.
3. QUALIFIER TRANSLATION: Are there unit or scope qualifiers?
   - "per unit / per item" → condition must be Price/Amount > N, NOT Price > N alone.
   - "not yet X years old" → age < X strictly (not ≤).
   - "more than N posts" → COUNT > N (not ≥ N).
4. COLUMN ORDER: Match the order terms appear in the question.
   - "average upvotes AND average age" → UpVotes column first, Age second.
5. ROW COMPLETENESS: Are ALL matching rows returned?
   - If multiple rows satisfy the condition equally, return ALL of them — not just the closest or first.
6. EXTRA COLUMNS: Strip any column the question did NOT ask for.
   - Do not include CustomerID, molecule_id, etc. when the question only asks for Consumption or element.

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

Example — list_context observation cut off mid-item, confirming all files before proceeding:
```json
{"thought":"OBSERVED: list_context ended mid-item at {'path': 'json', 'kind'... — the file listing is INCOMPLETE. UNACTIONED: unknown files may still exist. NEXT: confirm all files with os.walk before touching any data.","action":"execute_python","action_input":{"code":"import os\nprint([os.path.join(r,f) for r,d,fs in os.walk('.') for f in fs if not f.endswith('Identifier')])"}}
```

Example — knowledge.md was truncated: check file size first, then set max_chars accordingly:
```json
{"thought":"OBSERVED: read_doc returned truncated=true for knowledge.md at max_chars=8000. list_context showed knowledge.md is 5459 bytes. UNACTIONED: full content not yet read. NEXT: set max_chars=6000 (size+500) to read it in one shot.","action":"read_doc","action_input":{"path":"knowledge.md","max_chars":6000}}
```

Example — knowledge.md still truncated after first retry, escalating further:
```json
{"thought":"OBSERVED: read_doc returned truncated=true again at max_chars=16000. UNACTIONED: full content not yet read — I cannot proceed without the domain thresholds inside. NEXT: double to 32000 and retry. I will keep doubling until truncated=false.","action":"read_doc","action_input":{"path":"knowledge.md","max_chars":32000}}
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

Example — PRE-ANSWER VERIFICATION catches wrong column before submission:
```json
{"thought":"PRE-ANSWER VERIFICATION: Question says 'what is the comment with the highest score'. (1) CONTENT vs IDENTIFIER: 'what is the comment' asks for the comment CONTENT, not its Id — I must return the Text column, not Id. (2) AGGREGATION: one highest-score row is correct. (3) QUALIFIERS: none. (4) ORDER: single column. (5) COMPLETENESS: one row. (6) EXTRA COLUMNS: none. Result: I had computed Id=90813 but must instead return the Text value for that row.","action":"execute_python","action_input":{"code":"import csv,sqlite3\ncon=sqlite3.connect('db/posts.db')\nids=set(r[0] for r in con.execute('SELECT Id FROM posts WHERE ViewCount BETWEEN 100 AND 150'))\ncon.close()\nbest=None;bscore=-1\nwith open('csv/comments.csv') as f:\n  for r in csv.DictReader(f):\n    if int(r['PostId']) in ids and int(r['Score'])>bscore:\n      bscore=int(r['Score']);best=r['Text']\nprint(best)"}}
```

Example — PRE-ANSWER VERIFICATION catches missing DISTINCT before submission:
```json
{"thought":"PRE-ANSWER VERIFICATION: Question says 'tally the toxicology element of the 4th atom'. (1) CONTENT vs IDENTIFIER: element is already a text value. (2) AGGREGATION: 'tally' means enumerate DISTINCT elements — I returned 84 molecule-element pairs but the question wants distinct elements only. I must apply DISTINCT. Fixing now.","action":"execute_python","action_input":{"code":"carcinogens=set(open('carcinogens.txt').read().split())\nelems=set()\nfor line in open('molecule_4th_atom.txt'):\n  parts=line.strip().split('\\t')\n  if len(parts)==2 and parts[0] in carcinogens: elems.add(parts[1])\nprint(sorted(elems))"}}
```

Example — large prose file (>20KB): skip read_doc, confirm files then understand format first:
```json
{"thought":"OBSERVED: doc/superhero.md is 177990 bytes — too large for read_doc. os.walk confirmed no structured CSV/JSON/DB exists. UNACTIONED: format not yet understood. NEXT: print 3 raw sections to learn the exact delimiter and attribute phrases before writing any regex.","action":"execute_python","action_input":{"code":"with open('doc/superhero.md') as f: c=f.read()\nparts=[p for p in c.split('\\n\\n') if len(p)>50]\nprint(f'Sections: {len(parts)}')\nfor p in parts[:3]: print('---'); print(p[:300])"}}
```

Example — after understanding prose format, extract ALL attributes in one comprehensive script:
```json
{"thought":"OBSERVED: sections split by 'An entry has been recorded for'; height uses variants 'height is X cm' / 'height is recorded as X.0 centimeters'; publisher uses 'publisher N'. NEXT: extract all (height, publisher_id) pairs in one script using all variants simultaneously.","action":"execute_python","action_input":{"code":"import re,json\nwith open('doc/superhero.md') as f: c=f.read()\nwith open('json/publisher.json') as f: pub={r['id']:r['publisher_name'] for r in json.load(f)['records']}\nHPAT=re.compile(r'height(?:[^\\d]{0,30})(\\d{2,3}(?:\\.\\d)?)\\s*(?:cm|centimeter)',re.I)\nPPAT=re.compile(r'publisher\\s+(\\d+)',re.I)\nparts=c.split('An entry has been recorded for')\ntotal=marvel=0\nfor p in parts:\n  h=HPAT.search(p);pb=PPAT.search(p)\n  if h and pb:\n    ht=float(h.group(1));pid=int(pb.group(1))\n    if 150<=ht<=180: total+=1;marvel+=(pub.get(pid)=='Marvel Comics')\nprint(total,marvel,round(100*marvel/total,2) if total else 'N/A')"}}
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
        "1. Call list_context first. Note every file path AND each file's size in bytes.\n"
        "   - If the observation ends mid-item, confirm all files with os.walk via execute_python.\n"
        "2. Read knowledge.md FULLY before touching any data file.\n"
        "   - From list_context, note the knowledge.md size. Set max_chars = size + 500.\n"
        "   - If still truncated, double max_chars and retry (8000→16000→32000→64000).\n"
        "   - NEVER skip or partially read knowledge.md — it defines thresholds and schema.\n"
        "3. If a query returns 0 rows, verify the data format and check all relevant files before concluding empty.\n"
        "4. Never repeat an identical (tool, parameters) call — change something.\n"
        "5. Keep execute_python code under 600 characters. Use short variable names.\n"
        "6. BEFORE calling answer, run the PRE-ANSWER VERIFICATION checklist:\n"
        "   a. Re-read the question. Does 'what is the X' mean the TEXT of X, or just its Id?\n"
        "   b. Does the question say 'tally / distinct / unique'? → apply DISTINCT or GROUP BY.\n"
        "   c. Does the question say 'per unit'? → divide price by quantity, not use price alone.\n"
        "   d. Do column names and their ORDER match the question's phrasing?\n"
        "   e. Are ALL matching rows included — not just the first/closest one?\n"
        "   f. Remove any columns the question did NOT ask for.\n"
        "7. Call the `answer` tool only after the above checklist passes.\n"
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
