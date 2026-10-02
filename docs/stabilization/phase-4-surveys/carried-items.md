# Phase 4 survey: dw-stabilization worktree (read-only)

Paths relative to /Users/don/src/dkackman/dw-stabilization. community_pipelines ignored. Line numbers are as of HEAD e768378c.

## Helper semantics to keep in mind
- `is_ref(kind, value)`: isinstance str + startswith; accepts a str or a tuple.
- `ref_name(kind, value)`: returns the remainder or None; single prefix only; does NOT `.strip()`. Many hand sites do `.removeprefix(X).strip()`, so the replacement is `ref_name(...)` + `.strip()` (or add a stripping variant). Call this out in the plan.
- `make_ref(kind, name)`: single prefix only.
- Tuples: SUBSTITUTED=(VARIABLE,ITEM), UNRESOLVED=(VARIABLE,ITEM,PREVIOUS_RESULT,GATHER), DEFERRED=UNRESOLVED+(ASSET,OUTPUT,PROMPT,CONSTANT,BUILTIN). `references.py` has no stripping helper and no CONSTRAINT tuple.

## Part A. Prefix spellings outside dw/references.py

### A3. Module-level alias constants (16 lines, 13 modules). Each is the root cause of several A1/A2 sites.
| file:line | code | replacement |
|---|---|---|
| dw/assets.py:30 | `ASSET_PREFIX = references.ASSET` | delete; importers (reference_names, server/exports, server/outputs, server/admission) use `references.ASSET` / helpers |
| dw/runs.py:57 | `OUTPUT_PREFIX = references.OUTPUT` | delete; importers: reference_names, realize, server/exports, server/outputs |
| dw/prompts.py:28 | `PROMPT_PREFIX = references.PROMPT` | delete; importers: arguments, realize, server/catalog, server/admission, reference_names |
| dw/realize.py:42 | `BUILTIN_PREFIX = references.BUILTIN` | delete; importer: plan.py:29 |
| dw/realize.py:43 | `VARIABLE_PREFIX = references.VARIABLE` | delete; importers: plan.py:30, server/jobs.py:46 |
| dw/prompts.py:33-40 | `RESERVED_TEXT_PREFIXES = (PREVIOUS_RESULT, VARIABLE, CONSTANT, ASSET, OUTPUT, PROMPT_PREFIX)` | no helper fits: a bespoke 6-tuple (it is not DEFERRED, which also holds ITEM/GATHER/BUILTIN). Either add a named tuple to references.py (e.g. RESERVED_TEXT) or keep and use is_ref(RESERVED_TEXT_PREFIXES, text). Also joined into messages at prompts.py:159 and server/routes/library.py:573 |
| dw/shots.py:42 | `SHOT_REFERENCE_PREFIX = f"{ref_prefixes.PREVIOUS_RESULT}shot@"` | no helper fits: a composite prefix plus the literal "shot@" (cf. MEMBER_SEPARATOR "@" in references.py). Use `is_ref(PREVIOUS_RESULT, r)` and `ref_name(...).startswith("shot@")`, or `make_ref(PREVIOUS_RESULT, "shot@")` |
| dw/reference_names.py:35 | `_UNRESOLVED_PREFIXES = references.UNRESOLVED` | `references.UNRESOLVED` |
| dw/video_extensions.py:30 | `_UNRESOLVED_PREFIXES = references.UNRESOLVED` | same |
| dw/probe_paths.py:21 | `UNRESOLVED_PREFIXES = references.UNRESOLVED` (exported in `__all__` line 66) | same; the public export means checking importers (grep found none outside the module) |
| dw/reference_limits.py:49 | `_UNRESOLVED_PREFIXES = references.UNRESOLVED` | same |
| dw/adapter_compatibility.py:58 | `_UNRESOLVED_PREFIXES = references.UNRESOLVED` | same |
| dw/subfolders.py:31 | `_UNRESOLVED_PREFIXES = references.SUBSTITUTED` | `references.SUBSTITUTED` |
| dw/content_types.py:73 | `_UNRESOLVED_PREFIXES = references.SUBSTITUTED` | same |
| dw/kernel_availability.py:36 | `_UNRESOLVED_PREFIXES = references.SUBSTITUTED` | same |
| dw/step_value_checks.py:49 | `_UNRESOLVED_PREFIXES = references.SUBSTITUTED` | same |

The name `_UNRESOLVED_PREFIXES` is misleading: five of the eight modules bind it to SUBSTITUTED, three to UNRESOLVED.
No module rebuilds `(VARIABLE, ITEM)` or UNRESOLVED inline; the duplication is through aliases only.

Inline ad-hoc tuples passed to is_ref (legit use of the helper, but none matches a named tuple; 5 sites):
- dw/argument_media.py:138 and :294 `references.is_ref((references.PREVIOUS_RESULT, references.VARIABLE), x)`
- dw/arguments.py:935 same pair
- dw/video_extensions.py:46 `(CONSTANT, PROMPT)`
- dw/server/catalog_shape.py:171 `(VARIABLE, GATHER)`
A named tuple (e.g. LAZY_MEDIA = (PREVIOUS_RESULT, VARIABLE)) would remove the repetition; optional.

### A1 + A2. startswith / removeprefix / slice / replace by hand
"->" is the replacement. `S` = SUBSTITUTED, `U` = UNRESOLVED.
Many are startswith + removeprefix/slice pairs of one logical site; pairs are marked (pair).

startswith sites (31 lines):
- dw/plan.py:186 `if isinstance(reference, str) and reference.startswith(VARIABLE_PREFIX):` + :187 `name = reference.removeprefix(VARIABLE_PREFIX)` (pair) -> `name = ref_name(VARIABLE, reference)`; `if name is not None`
- dw/plan.py:230 `if isinstance(written_seed, str) and written_seed.startswith(VARIABLE_PREFIX):` + :231 removeprefix (pair) -> ref_name
- dw/plan.py:652 `if isinstance(path, str) and not path.startswith(BUILTIN_PREFIX):` -> `not is_ref(BUILTIN, path)` (note: `path` must be str: is_ref already checks)
- dw/realize.py:104 `...definition_seed.startswith(VARIABLE_PREFIX):` + :105 `seed_variable = definition_seed.removeprefix(VARIABLE_PREFIX)` (pair) -> ref_name
- dw/realize.py:152 `if string.startswith(PROMPT_PREFIX):` -> `is_ref(PROMPT, string)`
- dw/realize.py:218 `...not path.startswith(BUILTIN_PREFIX):` -> `not is_ref(BUILTIN, path)`
- dw/realize.py:121 `if value.startswith(prefix) and value not in found:` inside `strings_with_prefix(tree, prefix)`: generic over a parameter, not a spelling; -> `is_ref(prefix, value)` is possible. Not counted as a site.
- dw/server/jobs.py:310 `if not isinstance(seed, str) or not seed.startswith(VARIABLE_PREFIX):` + :312 `name = seed.removeprefix(VARIABLE_PREFIX)` (pair) -> ref_name
- dw/server/admission.py:277 `elif leaf.startswith(PROMPT_PREFIX):` -> `is_ref(PROMPT, leaf)`
- dw/server/catalog.py:50 `if value.startswith(PROMPT_PREFIX):` + :51 `references.add(value.removeprefix(PROMPT_PREFIX).strip())` (pair) -> `ref_name(PROMPT, value)` + `.strip()` (note: the local variable is named `references`, shadowing the module name there)
- dw/vram_estimate.py:61 `...value.startswith(ref_prefixes.VARIABLE):` + :62 `variables.get(value[len(ref_prefixes.VARIABLE) :])` (pair) -> ref_name
- dw/vram_estimate.py:256 `if value.startswith(ref_prefixes.VARIABLE):` + :257 `yield value[len(ref_prefixes.VARIABLE) :]` (pair) -> ref_name
- dw/vram_estimate.py:298 `...entries.startswith(ref_prefixes.VARIABLE):` + :299 `name = entries[len(ref_prefixes.VARIABLE) :]` (pair) -> ref_name
- dw/adapter_compatibility.py:183 `...not reference.startswith(references.VARIABLE):` + :185 `variable = reference.removeprefix(references.VARIABLE)` (pair) -> ref_name
- dw/adapter_compatibility.py:143-145 `weight_name.startswith(_UNRESOLVED_PREFIXES)` -> `is_ref(U, weight_name)`
- dw/variable_constraints.py:423 `...reference.startswith(references.CONSTRAINT):` + :425 `constraints[reference[len(references.CONSTRAINT) :]]` (pair) -> ref_name(CONSTRAINT, ...)
- dw/variable_constraints.py:453 `and value.startswith(references.CONSTRAINT)` + :454 `and value[len(references.CONSTRAINT) :] not in constraints` (pair) -> ref_name
- dw/shots.py:255 `if isinstance(reference, str) and reference.startswith(SHOT_REFERENCE_PREFIX):` + :257 `member = reference[len(ref_prefixes.PREVIOUS_RESULT) :]` (pair) -> no helper fits for the composite prefix (see A3); the slice -> `ref_name(PREVIOUS_RESULT, reference)`
- dw/shots.py:259-261 `elif isinstance(reference, str) and reference.startswith(ref_prefixes.PREVIOUS_RESULT):` + :263 `step = reference[len(ref_prefixes.PREVIOUS_RESULT) :]` (pair) -> ref_name
- dw/workflow_run.py:178-180 `candidate.startswith(references.PREVIOUS_RESULT)` + :181 `field["entry"] = candidate[len(references.PREVIOUS_RESULT) :]` (pair) -> ref_name
- dw/locations.py:533 `return value.startswith(references.DEFERRED)` -> `is_ref(references.DEFERRED, value)` (is_ref also guards non-str; check the function's guard)
- dw/prompts.py:156 `if text.startswith(RESERVED_TEXT_PREFIXES):` and dw/server/routes/library.py:569 `if str(request.prompt.get("text", "")).startswith(RESERVED_TEXT_PREFIXES):` -> `is_ref(RESERVED_TEXT_PREFIXES, ...)`; the tuple itself: no helper fits (see A3)
- Alias-tuple startswith, all -> `is_ref(S/U, value)`:
  dw/subfolders.py:89, dw/content_types.py:146, dw/kernel_availability.py:145 (all S, `isinstance` guard sits beside it), dw/step_value_checks.py:74 (S), dw/video_extensions.py:44 (U), dw/probe_paths.py:41 (U), dw/reference_limits.py:102 (U), dw/reference_names.py:100 (U)
- dw/reference_names.py:97 `if not value.startswith(prefix):` + :99 `rest = value[len(prefix) :].strip()` (pair; loops over `_KINDS` of (OUTPUT_PREFIX, ASSET_PREFIX, PROMPT_PREFIX)) -> `ref_name(prefix, value)`; the `_KINDS` tuple entries hold alias constants (lines 48-50) -> `references.OUTPUT/ASSET/PROMPT`

removeprefix / slice / replace sites not paired above:
- dw/assets.py:113 `name = validate_asset_reference(reference.removeprefix(ASSET_PREFIX).strip())` -> ref_name(ASSET, reference).strip()
- dw/runs.py:236 `...reference.removeprefix(OUTPUT_PREFIX).strip()` -> ref_name(OUTPUT, ...)
- dw/prompts.py:97 `...reference.removeprefix(PROMPT_PREFIX).strip()` -> ref_name(PROMPT, ...)
- dw/arguments.py:315 `name = validate_constant_name(reference.removeprefix(references.CONSTANT).strip())` -> ref_name(CONSTANT, ...)
- dw/realize.py:173 `name = reference.removeprefix(PROMPT_PREFIX).strip()` ; :189 `name = reference.removeprefix(OUTPUT_PREFIX).strip()`
- dw/reference_names.py:44 `return reference.removeprefix(OUTPUT_PREFIX).strip()`
- dw/server/admission.py:268 `name = leaf.removeprefix(ASSET_PREFIX).strip()`
- dw/server/exports.py:284 `reference.removeprefix(ASSET_PREFIX).strip()` ; :309 `name = reference.removeprefix(OUTPUT_PREFIX).strip()`
- dw/server/outputs.py:173 `return name.removeprefix(OUTPUT_PREFIX).strip()` ; :245 `reference.removeprefix(ASSET_PREFIX).strip(),`
- dw/workflow.py:450 and :921 `builtin_name = path.replace(references.BUILTIN, "")` -> `ref_name(BUILTIN, path)`. This one is a latent bug: `replace` strips the substring anywhere, not just the prefix; `ref_name` strips only the prefix. Both sit in the duplicated builtin-resolution blocks (see B3).
No `split(":", 1)` or `partition(":")` strips a reference prefix. (`vram_estimate.py:161` `.split(":")[0]` is a device string; `validation.py:189/736` partitions are warning-separator / dotted paths; `media_frames.py:191` and `server/routes/media.py:298` use the `frame:` prefix, which is not a reference prefix and has no constant.)
No `previous_results.py:165` hit: `name[len(result_name)+1:]` strips a step name, not a prefix. `for_each.py:283` `reference[len(group):]` likewise.

### A4. Building a reference by hand
- dw/elision.py:210 `reference = references.VARIABLE + name` -> make_ref(VARIABLE, name)
- dw/for_each.py:203 `return references.PREVIOUS_RESULT + _rewrite_reference(` -> make_ref(PREVIOUS_RESULT, ...)
- dw/for_each.py:263 `references.PREVIOUS_RESULT + member_name(group, key)` -> make_ref
- dw/for_each.py:292 (inside f-string, message) `f"'{references.GATHER}{group}' for every member's result, ...` -> `make_ref(GATHER, group)`
- dw/realize.py:201 `return f"{OUTPUT_PREFIX}{relative}"` -> make_ref(OUTPUT, relative)
- dw/server/routes/assets.py:363 `logger.info(f"Kept output {body.name} as asset:{asset_name}")` and :433 `logger.info(f"Deleted asset:{relative} ({path})")` -> literal text "asset:" in log lines (a spelled prefix, not a reference value) -> `make_ref(ASSET, ...)`. The rest of routes/assets.py already uses `make_ref(ASSET, ...)` (161, 219, 319, 365, 438); docstrings at 85/176/281/380... only mention "asset:" in prose.
- dw/server/enhancers.py:29 `"workflow": "builtin:h3_context_ir.json",` -> a whole reference literal in the PRESETS dict. -> `make_ref(BUILTIN, "h3_context_ir.json")` (constants can't be built at class level without an import; module already can import references)
- dw/arguments.py:327-328 message `f"'{references.CONSTANT}' reads a value, ..."` -> uses the constant for text only; no change needed (not a reference construction).
Prose-only strings that mention a prefix in error text (no handling; leave): assets.py:124, runs.py:253, locations.py:155/554 (no actual prefix), serve.py:89, step_value_checks.py:299, tasks/assess.py:222, tasks/task.py:552, type_helpers.py:209, validation.py:445, variables.py:389, video_extensions.py:58, workflow.py:457/929, dw_mcp/assets.py:157/185, dw_mcp/server.py:58, result.py:482.

### A5. Regexes
- Python: none embed a prefix (grep of `re.compile/match/search/sub` in dw, dw_mcp: nothing prefix-related; `security.py:652` CONSTANT_NAME_PATTERN is a name rule).
- JSON schema (not Python, outside the metric): dw/workflow_schema.json:103, :201, :211, :531 `"pattern": "^variable:"` and :400 `"pattern": "^constraint:[a-zA-Z_][a-zA-Z0-9_-]*$"`. 5 spellings. No helper (a schema is data); a test that pins the schema patterns to `references.VARIABLE`/`CONSTRAINT` is the way to guard them.

### Totals (lines, outside dw/references.py)
- Form 1 startswith: 31 lines (+1 generic parameter site, not counted)
- Form 2 removeprefix 18 + slice 9 (incl. reference_names:99) + replace 2 = 29 lines
- Form 3 aliases: 16 lines (+5 inline ad-hoc tuples, optional)
- Form 4 building: 3 concat/f-string in engine (elision, for_each x2) + realize:201 + for_each:292 + routes/assets x2 + enhancers x1 = 8
- Form 5 regex: 0 in Python, 5 in JSON schema
- Sum: 84 lines in dw/ (about 70 logical sites once the ~15 startswith+strip/slice pairs are merged), spread over 31 Python modules: adapter_compatibility, arguments, assets, content_types, elision, for_each, kernel_availability, locations, plan, probe_paths, prompts, realize, reference_limits, reference_names, runs, shots, step_value_checks, subfolders, variable_constraints, video_extensions, vram_estimate, workflow, workflow_run, server/admission, server/catalog, server/enhancers, server/exports, server/jobs, server/outputs, server/routes/assets, server/routes/library. dw_mcp has none. Tests were not surveyed.
- Biggest clusters: realize.py 9, plan.py 5, vram_estimate.py 6, for_each.py 3, shots.py 5, variable_constraints.py 4, server/exports 2, server/outputs 2.
- Importers of alias constants (the aliases cannot be deleted until each is changed): ASSET_PREFIX: reference_names, server/exports, server/outputs, server/admission; OUTPUT_PREFIX: reference_names, realize, server/exports, server/outputs; PROMPT_PREFIX: arguments (already uses is_ref at :289), reference_names, realize, server/catalog, server/admission; BUILTIN_PREFIX/VARIABLE_PREFIX: plan, server/jobs; RESERVED_TEXT_PREFIXES: server/routes/library.

## scripts/arch_metrics.py: how `prefix_literals` is counted
- `REFERENCE_PREFIXES` (lines 28-42) is a frozenset of the 10 prefix strings; `PREFIX_OWNERS = {"dw/references.py"}` (line 43).
- `measure()` (lines 185-210): for every `.py` under packages dw and dw_mcp (excludes community_pipelines etc.), it `ast.walk`s the parse tree and increments `prefix_literals` for each `ast.Constant` whose `value in REFERENCE_PREFIXES` (exact string equality) in a file not in PREFIX_OWNERS (lines 202-207). Only `ast.Constant` nodes are examined, nothing else.
- What it catches: a literal that is exactly `"asset:"`, `"variable:"`, ... in any position: `startswith("asset:")`, `"asset:" + name`, a bare alias assignment, and an f-string whose constant fragment is exactly the prefix (`f"asset:{x}"` yields a Constant fragment "asset:").
- What it misses:
  - Form 1, 2, 3: every site uses a constant name, never a literal, so all 31 + 29 + 16 lines above are invisible to it by design. The metric cannot see "spelling" handling, only literals.
  - Form 4: concat/f-string via constants (elision, for_each, realize) are invisible; f-strings where the fragment is not exactly the prefix are missed: routes/assets.py:363 (fragment " as asset:"), :433 (fragment "Deleted asset:"); a whole reference literal `"builtin:h3_context_ir.json"` (enhancers.py:29) is missed, as is any `"asset:foo.png"` literal (anything with a suffix) or a message embedding a prefix. `"frame:"` is not in the set.
  - Form 5: regexes with the prefix inside a longer pattern are missed; and the JSON schema (not .py) is out of scope entirely.
  - tests/ and scripts/ are not scanned (packages are only dw and dw_mcp).
- Suggested metric upgrade: add a second counter `prefix_handling` that flags (a) `ast.Call` with attr startswith/removeprefix/replace/partition whose argument resolves to a REFERENCE constant/alias/tuple name, (b) `ast.Subscript` slices `[len(<refs const>):]`, (c) `ast.BinOp(+)` / `JoinedStr` with a refs const operand, (d) module-level `Assign` whose value is `references.<X>`; plus `prefix_literals` extended to a substring/startswith check (`any(p in s for p in REFERENCE_PREFIXES)` on non-docstring Constants, with an allowlist for prose). My AST scan script (not saved) found the 65 call/slice/concat/fstring hits plus 16 aliases that make up the totals above.

## Part B. Carried items

### B1. `for_each._copy_leaf` (dw/for_each.py:217-227), callers :214, :238, :251
Code:
```
def _copy_leaf(value):
    try:
        return copy.deepcopy(value)
    except Exception:
        return value
```
Correction to the premise: it does NOT share leaves. Inside a member (`member is not None`, `_rewrite` tail at :214, `_item` at :238 and :251) every non-str, non-container leaf, and every `item:` field value, is `copy.deepcopy`'d, with a fall-back to sharing when deepcopy raises. Outside a member the leaf is returned as is (comment :205-213). The ROADMAP (phase 2d follow-up, docs/stabilization/ROADMAP.md:371) states it correctly: "`for_each._copy_leaf` and the step-cache snapshot still copy media leaves". The leaves are the ones `realize_args` has already turned into loaded images, decoded frame lists, torch tensors, `from_file()` dataclasses (see the comment at :205-213 and deep_equal in step_cache.py:313+), so each for_each member deep-copies the template's media: N members hold N copies of the media, plus another copy per member's snapshot (B2). Risk of the current behaviour: memory multiplication and time, and a silent fall-back to sharing for anything not deep-copyable (so behaviour differs by leaf type). The fix would change copying of leaves to sharing, i.e. use `copy_containers` semantics (containers rebuilt, leaves shared; `_rewrite` already rebuilds dict/list containers itself so only the scalar/media leaf and any tuple remain). Risk of that fix: any task that mutates a leaf in place (the workflow.py:195-203 comment names `conform_artifact` stamping fps onto what `select` handed back) would now edit the object every member shares; tuples as leaves currently get a deep copy of their contents. Import note: `copy_containers` lives in dw/step_cache.py, which imports torch/numpy/PIL; `for_each` currently imports only `references`, so a direct import adds a heavy edge and possibly a cycle (step_cache is imported by realize, workflow_run, pipeline). Moving `copy_containers` to `dw/references.py`-style leaf module (or a tiny `dw/copying.py`) avoids it.

### B2. Step-cache snapshot and `copy_containers`
- Defined: `copy_containers` at dw/step_cache.py:285-310 (dicts, lists, tuples exact types rebuilt, everything else, including subclasses, shared). Test pins it: tests/test_step_cache.py:859.
- Current users: dw/realize.py:86, :92 (definition + variables copy, import at :37); dw/workflow_run.py:290 (`recorded_variables = copy_containers(variables)`, import :52); dw/pipeline_processors/pipeline.py:119 (import :42).
- The snapshot: `cache_lookup` in dw/workflow_run.py:395-436:
  ```
  step_data_snapshot = None
  if is_cacheable:
      try:
          step_data_snapshot = copy.deepcopy(step_data)       # line 398
          if parent_saves_this: step_data_snapshot["__saved_by_parent__"] = True
          ... step_data_snapshot["__borrowed_pipelines__"] = borrowed
      except Exception as ex:
          logger.debug(... "not copyable ... skipping the step cache for it"); is_cacheable = False
  ```
  The snapshot is then the lookup key (`step_cache.get(workflow_id, step_data_snapshot, step_seed, ...)` at :421-425) and, via `cache_entry` (:712) and `step_cache.put` (:727-730), the stored key. Matching uses `deep_equal` (step_cache.py:313+, with `a is b` short-circuit and value comparison of Image/tensor/array).
- Current behaviour: every cacheable step deep-copies `step_data`, which at that point already holds the realized media arguments, so each step run copies all its images/frames/tensors once for the lookup and keeps that copy alive in the cache entry. A leaf that cannot be copied disables caching for that step.
- Fix: `copy_containers(step_data)` in place of `copy.deepcopy` gives an independent key structure (the later in-place edits the comment at :380-387 worries about are entry replacements/key assignments, which the container copy isolates) with shared media leaves; the `except` branch ("not copyable") becomes unreachable for leaves, so remove it or narrow it. Risks: (1) a leaf mutated in place after the snapshot silently changes the key (and `deep_equal`'s `a is b` fast path would then report a false hit); (2) the cache entry now pins the same media objects the live run holds (retention semantics change; LRU byte accounting in step_cache.py ~:530-600 would need a look); (3) `copy_containers` shares subclass containers (OrderedDict etc.), so a definition with those is not isolated. Also the sub-workflow path `workflow.py:619` (`copy.deepcopy(value) if name in declared`) and workflow_run.py:450/497 `copy.deepcopy(workflow.workflow_definition)` are the same family (not named in the carry list).

### B3. Sub-workflow path resolution: every place
Core resolver: `resolve_sub_workflow(path, base_dir, confine_to)` at dw/library.py:616-717 (raises SubWorkflowNotFound(path, tried), library.py:593/717).
Places that resolve a path:
1. `Workflow.resolve_sub_workflow_path(path)` dw/workflow.py:439-473. Code: builtin branch (:449-467) with `path.replace(references.BUILTIN, "")` and name checks, `confine_to = builtin_root()`, existence check raising SubWorkflowNotFound; else `if confine_to is None and not os.path.isabs(path): confine_to = catalog_root_dir(self.file_spec)` then `resolve_sub_workflow(path, os.path.dirname(self.file_spec), confine_to)` (:469) and `validate_workflow_path(resolved, confine_to)`. Returns (path, root).
2. `Workflow._sub_workflow_action(step_definition, default_seed)` dw/workflow.py:908-998 (reached from `create_step_action` at :776, dispatch `if "workflow" in step_definition: return self._sub_workflow_action(...)` at :811). It re-implements the SAME resolution inline (:919-968): builtin branch at :920-941 duplicating the name check and message (differs: `confine_to = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflows")` instead of `builtin_root()`, no isfile check), `resolve_sub_workflow(path, os.path.dirname(self.file_spec), confine_to)` at :960, `validate_workflow_path` + `workflow_from_file(validated_path, self.output_dir, confine_to)`. It does not call resolve_sub_workflow_path, although that method's docstring says it is "the same resolution create_step_action does". The two copies already diverge in builtin confinement and in the replace-based name extraction (A2).
3. `Workflow.open_sub_workflow(path)` dw/workflow.py:475-481: calls `self.resolve_sub_workflow_path(path)` (:481) then `workflow_from_file(...)`. Used by validation.
4. `validation.sub_workflow_errors` dw/validation.py:755-807: per sub-workflow step it resolves twice: `workflow.resolve_sub_workflow_path(path)` at :776 (to get `resolved` for the cycle check and a precise error), then `workflow.open_sub_workflow(path)` at :793, which resolves again at workflow.py:481. Message ownership is the reason: the first call yields resolution errors verbatim; the second wraps any failure as "Sub-workflow '<path>': ...". Fix is to have `open_sub_workflow` take/return the resolved path (or split into `resolve` + `open_resolved(resolved, root)`) so one resolution feeds both.
5. `validation.sub_workflow_argument_warnings` dw/validation.py:809-838: `workflow.open_sub_workflow(reference["path"])` at :821 for every step with arguments; a third resolution per step, swallowing the error (":824 an unresolvable path is an error, reported by sub_workflow_errors"). Registered check "sub_workflow_warnings" (validation.py:545).
6. `read_sub_workflow` dw/realize.py:232-249 (called by `_digest`, `_record_sub_workflows` scan at :218-226): `resolve_sub_workflow(path, base_dir or ".", workflow_dir)` at :244 + `validate_workflow_path(candidate, root.root if root else None)`. Its docstring says it resolves "the way `Workflow.create_step_action` resolves it", yet it skips the `catalog_root_dir` fallback (:468 in workflow.py) and builtins (:218 skips them).
7. dw/server/routes/jobs.py:557-560 (observed-cost lookup for a composed child): `resolve_sub_workflow(path, base_dir or ".", candidate.workflow_dir)`, errors swallowed; again no catalog_root_dir fallback or builtin handling.
Also: `create_step_action` on a warm/cached path can reach only #2. `dw/library.py:128` documents the confinement root. So seven sites, four distinct implementations of the preamble (workflow.py x2, realize x1, routes/jobs x1) around the one real resolver.

### B4. `Workflow._run_dir`
- Declared as class attributes: dw/workflow.py:167 `_run_dir = None`, :172 `_run_dir_inherited = False`, :177 `_run_version = None` (comment :162-176).
- Set: dw/workflow_run.py:557 `workflow._run_dir = None` (flat layout, inside `_claim_run_dir`) and :566 `workflow._run_dir, workflow._run_version = claim_run_dir(...)`; both only when `not workflow._run_dir_inherited` (open_run :584-585). Sub-workflow: dw/workflow.py:993-994 `workflow._run_dir = self._run_dir; workflow._run_dir_inherited = self._run_dir is not None` (child is a fresh Workflow each time, inside `_sub_workflow_action`).
- Read: dw/workflow.py:289-290 (`step_output_dir`: `if self._run_dir: return self._run_dir`), :714 (via `workflow_run.owns_run_dir`, which reads workflow_run.py:145-148); dw/workflow_run.py:148, :217, :222 (`_run_version`), :252, :254, :573-574, :591, :602, :620/622, :970 (`owns_run_dir`) and the `finally` in `run()` (workflow.py:714).
- Not reset at the top of `run` (workflow.py:623-662): `run()` resets `pipeline_ownership` (:655), `manifest` (:656), `_elided_steps` (:659), and creates a new `RunRecord`, but never `_run_dir`, `_run_version` or `_run_dir_inherited`. A persistent worker reuses the Workflow across jobs (the comment at :650-654 says so). Consequence: if `prepare_run` raises (workflow.py:662) before `open_run` reaches `_claim_run_dir`, the `finally` at :713-715 sees the previous run's `_run_dir` and `owns_run_dir` true, so `write_run_manifest` rewrites the previous run's manifest.json with the failed run's record; `step_output_dir` also keeps returning the old directory before the claim. In the flat layout `_claim_run_dir` sets None itself, so only the nested layout is exposed. Fix: at the top of `run()` (before `prepare_run`) reset `self._run_dir = None; self._run_version = None` when `not self._run_dir_inherited` (a child's values are set by the parent and must stay).
- Test touchpoints: tests/test_runs.py:628 sets `workflow._run_dir` directly; tests/test_events.py:536/551/566 and tests/test_shots.py:1127/1133 read it after a run.

### B5. kernels-hub "Cannot find a build variant"
- Not produced by dw: it comes from the third-party `kernels` package (venv: kernels/resolver.py:103, :132, :221; the per-variant lines come from `variants_trace_str(trace)` in kernels/variants.py:615-628, which sorts via `_sort_variants` but, per the roadmap note, the lines come out in set-driven order across processes).
- dw only embeds the text: dw/kernel_availability.py:125-126
  ```
  except Exception as e:
      raise _KernelFault(f"'{value}' {KERNEL_FAULT_MARKER}: {e}") from e
  ```
  so the validation error the author sees carries the variant lines in whatever order the hub library printed.
- The only normaliser lives in scripts/surface_snapshot.py:212-229, `stable_message(value)`: if "Cannot find a build variant" is in the string it sorts the lines that start with "torch" in place; recursion over list/dict (used at :256 on warning-check output). tests/test_kernel_availability.py:31/103 uses a stub FileNotFoundError("no build variant ...").
- Fix: sort the `torch...` lines in `_fault_for_name` (or a small `stable_variant_message(e)` in kernel_availability.py) so the message is deterministic for users and agents, then drop `stable_message` from the snapshot script (or keep it as belt and braces). The line detection must keep to the "torch" prefix convention or the `variant_str: reason` shape. Checking the installed kernels version is needed because the format could change.

### B6. `argument_template`
- Schema: dw/workflow_schema.json:106-109
  `"argument_template": { "description": "Engine-injected: the arguments a parent workflow passed to this one when it ran it as a sub-workflow. Written by create_step_action from the step's 'arguments' block, not authored - a workflow file carrying one is read, but a sub-workflow step is how they are meant to be supplied.", "type": "object" }`
- Current behaviour: `create_step_action` writes nothing into the definition. It dispatches to `_sub_workflow_action` (workflow.py:811), which sets `workflow._handed_arguments = workflow_reference.get("arguments", {})` at :977 (attribute declared :209 with the rationale at :195-209: "kept here rather than written into workflow_definition, where every validate() and run() of the child deep-copied them again"). `Workflow.argument_template` (workflow.py:229-233) is now a property returning `self._handed_arguments` when not None, else `self.workflow_definition.get("argument_template", {})` (so an authored `argument_template` in a file is still read, as the description's last clause says). Readers: dw/step.py:95 `get_iterations(step_action.argument_template, ...)` (step_action here is the pipeline/task action's own property: pipeline_processors/pipeline.py:179, tasks/task.py:849, not Workflow's) and dw/pipeline_ownership.py:276, pipeline.py:409/443 mutate the action's template, not the workflow's. docs/RELEASING.md:88 notes "a composed child's steps no longer carry argument_template".
- Fix: reword the description to say the handed arguments are held on the child workflow object by `_sub_workflow_action` at run time and never written into the definition; an authored value in a file is still read as the fallback. Mind that the schema file is part of the `workflow-schema` MCP surface (scripts/surface_snapshot.py:196 `workflow_schema()` snapshots it, so the description change shows in the surface diff).

### B7. `gain_audio` region end vs `slice_audio` (after #557)
`slice_audio` (dw/tasks/audio_utils.py:142-248) delegates to `slice_region` (dw/task_domains.py:406-441), which rounds a frame-addressed end once:
```
start = frames_to_samples(start_frame or 0, fps, sample_rate)
if num_frames is not None:
    end = frames_to_samples((start_frame or 0) + num_frames, fps, sample_rate)
    return start, end - start
```
`gain_audio` (audio_utils.py:251-377), the frame branch at :332-343, still has the two-halves computation:
```
elif start_frame is not None or num_frames is not None:
    if fps is None: raise ValueError("gain_audio needs 'fps' to address a region in frames")
    start = frames_to_samples(start_frame or 0, fps, sample_rate)
    length = (
        max(total - start, 0) if num_frames is None
        else frames_to_samples(num_frames, fps, sample_rate)      # <- rounded separately from start
    )
```
with `frames_to_samples(frames, fps, sample_rate) = int(round(frames / fps * sample_rate))` (task_domains.py:401). Because `round(a)+round(b) != round(a+b)` in general, `start + length` can land a sample off the exact `round((start_frame+num_frames)/fps*sr)`; slice_audio's end is exact. The seconds branches (:317-325) match slice_region's. Other differences: gain_audio checks `fps is None` while slice_region uses `not fps`; gain_audio has an "everything" branch (`start=0; length=total`, :345-347) where slice_region returns None. Fix: have gain_audio call `slice_region(...)` (it already imports from task_domains; `region is None` plus "no region args at all" distinguishes the whole-track case), keeping its "needs fps" message. The clip at `region_end = max(region_start, min(start + max(length, 0), total))` (:350-351) stays. tests: check tests for gain_audio frame regions (not surveyed).

### B8. `concat_videos` vs `dissolve_videos` with a track that has no sample rate
Shared function: dw/tasks/joins.py:70-133 `reconcile_sample_rates(command, videos, names, waveforms, sample_rate=None, skip_unrated=True)`.
```
rates = [video.sample_rate for video, waveform in zip(videos, waveforms)
         if waveform is not None and (video.sample_rate or not skip_unrated)]
sample_rate = sample_rate or (max(set(rates)) if rates else None)
...
return [ waveform if waveform is None
              or (skip_unrated and not video.sample_rate)     # <- unrated track passed through unchanged
              or video.sample_rate == sample_rate
         else resample_waveform(waveform, video.sample_rate, sample_rate)
         for video, waveform in zip(videos, waveforms)], sample_rate
```
- concat_videos: dw/tasks/concat_videos.py `_prepare_inputs` (:~215-228) calls `reconcile_sample_rates("concat_videos", videos, names, _input_waveforms(videos), sample_rate)` with the default `skip_unrated=True`. A track whose AudioVideo carries `sample_rate` None/0 is excluded from `rates` (so it cannot influence the target) and is returned unresampled, i.e. joined as though already at the target rate, which plays at the wrong speed/pitch (the class of fault `as_track` and #140 refuse elsewhere; see dw/dsp.py:223-230). `audio_native_rate = getattr(video, "sample_rate", None)` (:~167) is also None for it.
- dissolve_videos: dw/tasks/dissolve_videos.py:297-304 `_dissolve_audio` passes `skip_unrated=False`. The unrated track enters `rates` as `None`/0 ... then `resample_waveform(waveform, None, target)` runs, and `dw/dsp.py:226-230` raises `ValueError("resample_waveform needs a sample_rate above zero, got None")`, so it refuses. Pinned by tests/test_dissolve_videos.py:94-106 ("refuses the unrated track").
- The docstring at joins.py:82-86 calls skip_unrated "concat_videos' long-standing rule". Fix options: make concat refuse like dissolve (flip to `skip_unrated=False`; behaviour change: surface note, and check the templates/tests that rely on pipeline-reported None rates, tests/test_result.py:109 comment mentions a pipeline reporting no sample rate), or resample using a fall-back rate with a warning; then delete the `skip_unrated` parameter.

### B9. The 8 one-line `Workflow` delegators (dw/workflow.py)
Registry: dw/validation.py ERROR_CHECKS/WARNING_CHECKS; warning checks are run in the registry by `admit()` directly, so the methods are not on the production path except where noted.
| method (def line) | body | delegates to |
|---|---|---|
| validation_context :498-502 | `return validation.workflow_context(self, arguments, composing, ceiling_index)` | dw/validation.py:607 `workflow_context` |
| sub_workflow_warnings :484-496 | `return validation.run_warning_check(self, "sub_workflow_warnings", arguments)` | validation.py:707 `run_warning_check` (check body `sub_workflow_argument_warnings` :809) |
| adapter_warnings :520-529 | `return validation.run_warning_check(self, "adapter_warnings", arguments)` | run_warning_check (check -> adapter_compatibility.adapter_warnings, validation.py:536) |
| inherited_vram_warnings :531-544 | `if not index: return []` then `validation.run_warning_check(self, "inherited_vram_warnings", arguments, ceiling_index=index)` (not strictly one line: has a guard) | run_warning_check (-> vram_inheritance.inherited_vram_warnings, validation.py:572-577) |
| slice_past_end_warnings :546-554 | `return validation.run_warning_check(self, "slice_past_end_warnings", arguments)` | run_warning_check (-> slice_preflight.slice_past_end_warnings, :554) |
| shot_span_warnings :556-565 | `return validation.run_warning_check(self, "shot_span_warnings", arguments)` | run_warning_check (-> shot_span_preflight.shot_span_warnings, :562) |
| null_variable_argument_warnings :567-583 | `return validation.run_warning_check(self, "null_variable_argument_warnings", arguments)` | run_warning_check (check at validation.py:541) |
| cache_hits :604-607 | `return workflow_run.cache_hits(self, arguments)` | dw/workflow_run.py:439 `cache_hits` |

Callers, production:
- validation_context: dw/validation.py:662 (`workflow_errors`, `context = workflow.validation_context(arguments, composing)`), :712 (`run_warning_check`, `workflow.validation_context(arguments, **context_fields)`), dw/server/admission.py:158 (`candidate.validation_context(checked, ceiling_index=ceiling_index)`).
- null_variable_argument_warnings: dw/server/routes/library.py:298 (`warnings += candidate.null_variable_argument_warnings()`).
- cache_hits: dw/worker.py:513 (`cached = workflow.cache_hits(command.get("arguments") or {})`).
- sub_workflow_warnings, adapter_warnings, inherited_vram_warnings, slice_past_end_warnings, shot_span_warnings: NO production callers (the registry runs the module functions through run_warning_check/admit). Only tests and scripts/surface_snapshot.py:204-208 `WARNING_CHECKS` tuple (adapter_warnings, slice_past_end_warnings, shot_span_warnings, null_variable_argument_warnings, sub_workflow_warnings) called by name at scripts/surface_snapshot.py:256 `getattr(workflow, name)()`; inherited_vram_warnings and cache_hits are not in that script.
Callers, tests:
- validation_context: tests/test_validation.py:332, :349; tests/test_admission.py:499 (parametrised `(Workflow, "validation_context")` monkeypatch target, so removing the method breaks that test's target and it must change to `validation.workflow_context`).
- sub_workflow_warnings: tests/test_validation.py:301 (WARNING_ORDER list entry, string only), :374; tests/test_workflow.py:1452, :1478, :1521.
- adapter_warnings (method): tests/test_validation.py:370; tests/test_lora_disable.py:124, :131 (`h3_workflow(tmp_path).adapter_warnings(...)`); (module-function uses at test_lora_disable.py:143, test_h3_adapters.py:136-178 are the other function).
- inherited_vram_warnings (method): tests/test_validation.py:377-378 (`workflow.inherited_vram_warnings(None, {"identity": {}})`); tests/test_vram_inheritance.py:125 (`workflow.inherited_vram_warnings(arguments, index)`); tests/test_admission.py:237 (registry swap by name, string); tests/test_vram_inheritance.py:295 (message text).
- slice_past_end_warnings (method): tests/test_validation.py:375; tests/test_slice_preflight.py:186 (`workflow.slice_past_end_warnings()`). Module-function calls elsewhere are not the method.
- shot_span_warnings (method): tests/test_validation.py:376; tests/test_shot_span_preflight.py:172; tests/test_admission.py:220/233/307 are registry-swap strings.
- null_variable_argument_warnings: tests/test_validation.py:371-372.
- cache_hits: tests/test_realize.py:451; tests/test_workflow_step_cache.py:656, :1009, :1019, :1036, :1047, :1056; tests/test_worker_execute.py:287 and :322 (fake workflow classes define their own `cache_hits`, so the worker's call at dw/worker.py:513 must keep an object-with-`cache_hits` seam or those fakes change).
Note for removal planning: replacing the methods with a module call means 3 production changes (validation.py:662/712, admission.py:158, routes/library.py:298, worker.py:513 = 5 call lines) and ~25 test lines; `Workflow.validation_errors` (:504) and `validate` (:585) are further delegators not in this list.
