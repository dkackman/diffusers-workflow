# Architecture: the seam map

Where each concept lives. One row per concept: the module that owns it, the rule that holds
across the seam in one sentence, and what enforces the rule. The detail is in the owning
module's docstring, so read that before editing. The map only says which one to open.

- **Owner**: repo-relative paths. A function name follows a colon (`dw/runs.py`: `run_versions`).
- **Enforced by**: a test (a file, or `file::Class::test_name`), a ratchet in `scripts/arch_metrics.py`
  (its key in `docs/stabilization/baseline.json`), the CodeQL pack, or the validation registry. A
  `—` means nothing mechanical holds the rule, so a change there is checked only by review.
- `tests/test_architecture_map.py` checks that every path named here exists and that every test
  named here is defined in its file.

## Engine core

| Concept | Owner | Rule | Enforced by |
| --- | --- | --- | --- |
| Workflow | `dw/workflow.py`: `Workflow`, `workflow_from_file` | A workflow is loaded and checked here, then run by `Workflow.run`, which hands each phase to `dw/workflow_run.py`. | `tests/test_workflow.py` |
| One execution, in phases | `dw/workflow_run.py` | `Workflow.run` calls `prepare_run`, `open_run`, `begin_steps` and then `run_step` per step, and the cache probe `cache_hits` shares `prepare_definition` and `cache_lookup` with the run, so the probe answers for the run that would happen. | `tests/test_workflow_step_cache.py::TestCacheHits::test_after_a_run_the_probe_names_what_the_next_run_reuses` |
| Step | `dw/step.py`: `Step`, `dw/workflow.py`: `Workflow.create_step_action` | A step is exactly one of a pipeline, a task or a sub-workflow (the schema's `oneOf`), `create_step_action` builds its action, and `dw/workflow_run.py` runs it. | `tests/test_step.py` |
| Pipeline loading | `dw/pipeline_processors/pipeline.py`: `Pipeline`, `dw/pipeline_processors/components.py`: `load_component`, `configure_components` | Components load from `from_pretrained_arguments`, and a quantization config is built in `dw/pipeline_processors/config_objects.py` from a `config_type` that the `_type` key convention loads by name, so a new backend needs no code. | `tests/test_config_objects.py` |
| Placement and offload | `dw/pipeline_processors/placement.py`: `place_component` | Every device a workflow names passes through `resolve_device()` (`dw/__init__.py`) before placement reads its backend, so the MPS accommodations also fire for a translated device. | `tests/test_device_portability.py::TestPlacement::test_a_translated_device_gets_its_backends_offload_downgrade` |
| On-demand residency | `dw/pipeline_processors/placement.py`: `apply_on_demand_placement` | The wrappers keep the wrapped signature (`functools.wraps`; H3's denoiser reads `signature(transformer.forward)`), they cannot be combined with `group_offload` on one component, and either one stops the wholesale `pipeline.to(device)`. | `tests/test_modular_pipeline.py::TestOnDemandResidency::test_the_wrapped_signature_survives`, `tests/test_modular_pipeline.py::TestOnDemandResidency::test_group_offload_and_on_demand_together_are_rejected` |
| LoRAs | `dw/pipeline_processors/adapters.py`: `active_loras`, `load_loras` | A `loras` entry whose `model_name` is null is switched off, because a template's `loras` list is fixed JSON and a variable can null a value but cannot remove an entry. | `tests/test_lora_disable.py::TestLoadLoras::test_an_all_null_entry_loads_nothing` |
| Step progress | `dw/pipeline_processors/progress.py`: `reported_progress_bars` | A pipeline with no step callback (every `ModularPipeline`) reports its denoise steps through its progress bars instead. | `tests/test_modular_progress.py` |
| Adding a task | `dw/tasks/task.py`: `register_command` | A task is a function registered with `@register_command`, and the signature of its implementation is its argument schema. | `tests/test_task_discovery.py::TestDescribeTask::test_signature_becomes_the_schema` |
| Task argument domains | `dw/task_domains.py` | A numeric domain that a signature cannot express is declared in its table, which is checked at validation and again at run time (`check_arguments`). | `tests/test_task_domains.py` |
| Variables | `dw/variables.py`: `set_variables`, `replace_variables`, `resolve_variable_values`, `argument_errors` | `argument_errors` folds a caller's arguments in exactly as `set_variables` does at the start of a run, so a bad name or value is a 400 before anything is queued. | `tests/test_variables.py::test_argument_errors_reports_a_dict_passed_for_a_string_variable`, `tests/test_variables.py::TestResolveVariableValues::test_an_undeclared_name_is_the_usual_error` |
| List variables from the CLI | `dw/run.py`, `dw/variables.py`: `get_value` | A `name=value` string given for a list variable is split on commas, so a list of objects (such as a template's `shots`) can only be passed as JSON over the API or MCP. | `tests/test_variables.py::test_set_variables_list_default_splits_on_comma` |
| `for_each` expansion | `dw/for_each.py`: `expand_for_each` | Expansion runs after substitution and before the reference check, and it names each member `<step>@<entry>`, so the step loop, the cache and the manifest only ever see ordinary steps. | `tests/test_for_each.py::TestNaming::test_member_name_joins_with_at` |
| Previous results | `dw/previous_results.py`: `get_iterations`, `previous_result_reference_errors` | References multiply into a cartesian product, and a literal reference to a step that is not earlier is a validation error at its JSON path. | `tests/test_previous_results.py::TestGetIterations::test_multiple_references_create_cartesian_product`, `tests/test_previous_results.py::TestStaticReferenceChecking::test_a_step_cannot_reference_itself_or_a_later_one` |
| Type conversion | `dw/arguments.py`: `realize_args`, `dw/type_helpers.py` | A key ending `_type` or `_dtype`, or named `dtype`, loads a Python object at realize time, and a `{}`-wrapped value stays a string (see `docs/WORKFLOW_GUIDE.md`, "Types and escaping"). | `tests/test_type_helpers.py::TestGetType::test_get_type_from_diffusers`, `tests/test_arguments.py::TestRealizeArgs::test_realize_escaped_type_reference` |

## References and libraries

| Concept | Owner | Rule | Enforced by |
| --- | --- | --- | --- |
| Reference prefixes | `dw/references.py` | Every prefix (`variable:`, `asset:`, `output:` and the rest) is spelled only here, and every module that tests for, strips or builds one calls this module. | `scripts/arch_metrics.py` ratchets `prefix_literals` and `prefix_handling`; `tests/test_references.py::test_references_imports_nothing_from_dw` |
| Explicit references resolve first | `dw/arguments.py`: `_realize_explicit_reference`, `dw/assets.py`, `dw/runs.py`: `resolve_output_reference`, `dw/prompts.py` | `asset:`, `output:`, `constant:` and `prompt:` resolve in `realize_args` before any key-name convention, and an `asset:` or `output:` path is confined to its library or output root by a `dw/security.py` validator. | `tests/test_assets.py::TestReferences::test_a_symlink_out_of_the_library_is_refused`, `tests/test_security_symlinks.py::TestOutputs::test_an_output_reference_does_not_follow_a_linked_run_directory` |
| Reference name shape | `dw/reference_names.py`: `reference_name_errors` | The shape of every `asset:`, `prompt:` and `output:` name is checked before the queue, but whether the file exists is checked against a workspace later. | `tests/test_reference_names.py` |
| Library search paths | `dw/library.py`: `LibraryPath`, `library_path` | Reads go through the roots front to back, so an earlier name shadows a later one, and writes go only to the front root, so saving something opened from a read-only root writes a copy. | `tests/test_library_path.py`, `tests/test_server_library_path.py::TestOneListingEnvelope::test_deleting_a_read_only_entry_gives_one_message` |
| Packaged builtins and sub-workflows | `dw/library.py`: `builtin_root`, `resolve_sub_workflow` | `builtin:` names the packaged `dw/workflows/` (not the top-level `workflows/` examples), and a sub-workflow path resolves beside its parent first, then by catalog name along the search path, and is confined to the root it resolves in. | `tests/test_sub_workflow_resolver.py` |
| Workspace | `dw/workspace.py`: `resolve_workspace`, `set_workspace` | The order is `--workspace`, then `DW_WORKSPACE`, then the `workspace` setting, then a working directory that looks like a workspace, then `~/diffusers-workspace`, and resolving creates nothing. | `tests/test_workspace.py::TestResolution::test_a_flag_wins_over_everything`, `tests/test_workspace.py::TestResolution::test_a_bare_working_directory_falls_back_to_the_home_workspace` |
| Realized workflow | `dw/realize.py`: `realize_workflow`, `dw/runs.py`: `write_realized_workflow` | Each run directory holds `workflow.json`, which pins every input that can change between runs: arguments, seed, stored prompt text and `output:.../latest/...`. | `tests/test_realize.py::TestVariablesAndSeed::test_arguments_become_the_variable_defaults` |

## Runs and outputs

| Concept | Owner | Rule | Enforced by |
| --- | --- | --- | --- |
| Run directories | `dw/runs.py`: `open_run`, `workflow_identity`, `new_run_id` | Each execution writes into `<output_dir>/<identity>/<run id>/` with a `manifest.json`, unless the flat layout (`output_layout`) is chosen. | `tests/test_runs.py` |
| Run versions | `dw/runs.py`: `open_run`, `run_versions`, `record_run_versions` | A run takes its number once, when it opens, as one more than the highest number recorded by any sibling, so deleting a middle run leaves a gap rather than renumbering. | `tests/test_runs.py::TestRunVersions::test_a_deleted_middle_run_leaves_a_gap_rather_than_renumbering`, `tests/test_runs.py::TestRunVersions::test_runs_started_in_the_same_second_still_number_upward` |
| Run-version surfaces | `dw/runs.py` (owner), `dw/server/routes/gallery.py`, `dw/server/outputs.py`, `dw_mcp/tools_catalog.py`: `list_gallery`, `ui/src/lib/pages/GalleryPage.svelte` | The number is computed only in `dw/runs.py`, and everywhere else it is read back, as a gallery field, as `output:<identity>/v<N>/<file>`, or as a job's `run_version`. | `tests/test_runs.py::TestRunVersions::test_an_output_reference_can_name_a_run_by_its_version` |
| Result subfolders | `dw/subfolders.py`: `subfolder_errors`, `dw/security.py`: `SUBFOLDER_PATTERN` | A step's `result.subfolder` is checked for shape after expansion and at run time, and it is confined to the run directory by `validate_output_path`. | `tests/test_subfolders.py` |
| Template subfolder roles | `workflows/templates/`, `dw/workflows/` | Every saving step of a template is marked `final` or `intermediate` with at least one `final`, and packaged builtins stay unmarked because assigning a role is the parent's job. | `tests/test_template_subfolders.py::test_every_saving_step_of_a_template_names_its_role`, `tests/test_template_subfolders.py::test_the_packaged_builtins_stay_unmarked` |
| Shots in a joined video | `dw/shots.py` | Every `AudioVideo` constructor carries, rescales, re-measures or drops `shots`, and the sample spans are measured from the joined waveform, not derived from frames. | `tests/test_shots.py` |
| Audio level of a deliverable | `dw/audio_qc.py`: `warn_without_headroom`, `warn_if_written_above_full_scale` | Each check warns and changes nothing, and a muxed video keeps the pre-encode headroom prediction to fall back on only when the post-write probe cannot measure the file. | `tests/test_result.py` |
| Template audio-level placement | `workflows/templates/minimax/music-video.json`, `workflows/templates/minimax/music.json` | `normalize_audio` (-3 dBFS) acts only on the track that goes into the final mux or file, so the slices that condition the shots are left untouched. | — |

## Validation, plan and cost

| Concept | Owner | Rule | Enforced by |
| --- | --- | --- | --- |
| Validation registry | `dw/validation.py`: `ERROR_CHECKS`, `WARNING_CHECKS`, `run_checks` | Adding a check means adding one `Check` to a registry, which runs it in order, and a check that raises becomes one `internal` finding while the rest still run. | `tests/test_validation.py::TestExceptionPolicy::test_a_raising_check_is_one_internal_error_and_the_rest_still_run` |
| Step-value checks | `dw/step_value_checks.py` | The bodies of `fps_errors`, `null_media_errors` and `select_errors` live here, and their only caller is the error registry in `dw/validation.py`. | the validation registry |
| Admission | `dw/server/admission.py`: `admit` | Validate, submit, rerun and enhance load and check a request once, and `JobManager.submit` queues what was admitted without checking it again. | `tests/test_admission.py::test_a_submit_expands_once` |
| Variable constraints | `dw/variable_constraints.py` | A model's rule about a value is declared in the workflow's `variable_constraints`, and it is checked at validation and at run time (`apply_constraints`) and reported in the catalog. | `tests/test_variable_constraints.py` |
| Plan | `dw/plan.py` | The plan comes from the same resolvers the run uses, and a bound `acknowledged_cost` is refused with 409 when the fingerprint or the required downloads changed. | `tests/test_server.py::TestBoundRerun::test_a_rerun_bound_to_a_stale_plan_is_refused` |
| Per-repo gating | `dw/plan.py`: `_collect_sources`, `downloads_required` | Each gated repo, including a `loras` entry's `model_name`, is probed and reported on its own, because access to one gated repo does not grant another. | `tests/test_plan.py::TestDownloadsRequired::test_a_gated_repo_this_token_lacks_access_to_is_blocked` |
| Observed cost | `dw/server/observed_cost.py`, `dw/plan.py`: `_tempered` | `observed` comes only from this box's finished jobs and never writes `cost`, and `plan.estimate` quotes the cold observed median ahead of the curated figure, blending toward that figure below three runs. | `tests/test_observed_cost.py::TestColdIsNotWarm::test_the_two_are_reported_separately_each_with_its_runs`, `tests/test_plan.py::TestLowConfidenceObservedEstimate::test_a_single_run_blends_toward_the_curated_figure` |
| Observed-cost residue | `dw/server/observed_cost.py`: `declared_drivers`, `dw/server/routes/library.py`: `get_workflow` | A cost driver that names no declared variable is dropped, and the raw workflow GET is served verbatim (with no `observed`) because the editor saves what it reads. | `tests/test_observed_cost.py::TestComparability::test_a_driver_naming_no_variable_is_dropped`, `tests/test_observed_cost.py::TestTheCatalogsDriversAreReal::test_every_declared_driver_is_a_variable_of_its_workflow` (the verbatim GET: —) |
| VRAM projection | `dw/vram_estimate.py` | A template's `vram_estimate` is projected for each pipeline step after `for_each` expansion, and only the largest step over the ceiling is reported. | `tests/test_vram_estimate.py` |
| H3 Ref2VA VRAM numbers | `workflows/templates/minimax/` (`vram_estimate`) | Every Ref2VA template declares `base_gb` 16.0, `bytes_per_voxel` 28.71 and `gb_per_reference` 1.0, a classification from field runs rather than a fitted curve. | `tests/test_h3_vram_ceiling.py::test_every_ref2va_minimax_template_declares_gb_per_reference` |
| Inherited VRAM ceiling | `dw/vram_inheritance.py`, `dw/server/deps.py`: `ceiling_index` | A workflow with no `vram_estimate` is matched by pipeline identity against an index of single-identity templates, cached on the listing's mtimes, and is warned, never refused. | `tests/test_vram_inheritance.py::test_every_template_declaring_one_identity_declares_the_same_numbers` |
| H3 adapter partition | `dw/adapter_compatibility.py` | An FL2VA LoRA on a reference (`ref2va`) step is refused, and a LoRA whose file name says neither `ref2v` nor `fl2v` is warned. | `tests/test_h3_adapters.py` |
| IC-LoRA reference scale | `workflows/templates/ltx2/` | The three conditioning templates run at `reference_downscale_factor: 1`, and the two upscalers at 2. | `tests/test_ltx2_ic_loras.py::TestEachTemplateMatchesItsCard::test_the_reference_is_encoded_at_the_output_resolution`, `tests/test_ltx2_ic_loras.py::TestTheGenerativeUpscaleMatchesTheSameCard::test_it_loads_the_upscaler_at_factor_two` |
| IC-LoRA numbers | `workflows/templates/ltx2/` | Every number in the IC-LoRA templates is taken from the vendor card. | `tests/test_ltx2_ic_loras.py::TestEachTemplateMatchesItsCard::test_the_strength_is_the_cards_default`, `tests/test_ltx2_ic_loras.py::TestEachTemplateMatchesItsCard::test_the_defaults_are_the_trained_bucket` |
| IC-LoRA prompt genre | `prompts/ltx2/` (tag `ic-lora`) | An IC-LoRA stored prompt is checked against its trained caption form, not against the 150-220-word T2V paragraph rule. | `tests/test_ltx_prompt_library.py::test_an_ic_lora_prompt_is_in_its_trained_form` |

## Execution and caching

| Concept | Owner | Rule | Enforced by |
| --- | --- | --- | --- |
| Step cache | `dw/step_cache.py`, `dw/workflow_run.py`: `cache_lookup` | A seeded rerun with unchanged inputs is served from the cache (`reused: true`, nothing written), a workflow with no `seed` skips the cache, and `rerun(new_seed=true)` is how to get a different result. | `tests/test_workflow_step_cache.py::test_cache_hit_marks_its_manifest_entry_and_event_reused`, `tests/test_workflow_step_cache.py::TestCacheHits::test_an_unseeded_workflow_has_no_hits`, `tests/test_rerun_new_seed.py::test_a_rerun_with_a_new_seed_draws_one_into_that_variable` |
| Worker protocol | `dw/worker_protocol.py` | Every command and reply is a frozen dataclass that travels as a wire dict, and this module imports neither `dw/worker.py` nor `dw/worker_manager.py`. | `tests/test_worker_messages.py::test_from_wire_inverts_to_wire`, `tests/test_worker_messages.py::test_an_unknown_reply_type_is_kept_whole_rather_than_raised` |
| Persistent worker | `dw/worker.py`, `dw/worker_manager.py`, `dw/serve.py` | Jobs run in one spawned worker process that keeps models loaded between runs, so a change to engine code needs a server restart. | `tests/test_worker_manager.py` |
| Failed-run reporting | `dw/worker.py`, `dw/worker_protocol.py`: `Failed`, `Cancelled` | A failed or cancelled run's reply still carries the manifest of the steps that ran. | `tests/test_worker_execute.py::test_failure_carries_the_manifest_of_the_steps_that_ran`, `tests/test_worker_execute.py::test_cancellation_carries_the_manifest_too` |
| Run context and events | `dw/events.py`: `RunContext`, `emit_warning` | A run's context travels through contextvars, and a run-time warning reaches the job through `emit_warning`, not just the log. | `tests/test_events.py` |

## Media and DSP

| Concept | Owner | Rule | Enforced by |
| --- | --- | --- | --- |
| Opening media | `dw/media.py` | Every `av.open` in the engine is in this module, which uses PyAV only and imports no step types. | `tests/test_media_layering.py::test_av_open_appears_only_in_media`, `tests/test_media_layering.py::test_media_imports_no_step_types` |
| Signal processing | `dw/dsp.py` | Pure numpy, scipy and pyloudnorm: it measures and transforms a waveform, decides nothing, and imports nothing from `dw`. | `tests/test_media_layering.py::test_dsp_imports_nothing_from_dw` |
| Assessment probes | `dw/tasks/assess.py`, `dw/assessment_rules.py`, `dw/server/assess.py` | A probe measures a finished file and lists `findings` against the rules table, and nothing in the engine acts on a finding. | `tests/test_assessment_rules.py` |

## Security and architecture guardrails

| Concept | Owner | Rule | Enforced by |
| --- | --- | --- | --- |
| Path, URL and argument validators | `dw/security.py` | Filesystem access goes through `validate_path` / `validate_workflow_path` / `validate_output_path` (with a base), URLs through `validate_url`, and subprocess arguments through `sanitize_command_args`. | `tests/test_security.py::test_path_validation`, CodeQL (`.github/codeql/dw-security/`) |
| Locations from a workflow | `dw/locations.py` | A media location in a workflow's arguments is checked by one policy (a local path is confined, a URL is filtered for SSRF), because the workflow JSON is untrusted. | `tests/test_security_ssrf.py` |
| Trust gate | `dw/trust.py` | An untrusted workflow (the default) may name only allowlisted, constructible classes and no remote code, and a dotted type it may not use is a validation error before anything is imported. | `tests/test_security_trust_gate.py::TestValidationRefusesBeforeImport::test_a_dotted_type_is_a_validation_error` |
| CodeQL path-injection model | `.github/codeql/dw-security/`, `.github/workflows/codeql.yml`, `dw/security.py` | The local pack models the validators in `dw/security.py` as sanitizers (`validate_path` only when it is given a base), so a validator that moves is re-modelled in the same commit. | the CodeQL query dw/path-injection |
| Archives never follow links | `dw/server/outputs.py`: `zip_download`, `dw/security.py` | A gallery or asset listing drops a symlink that leaves its root, and an archive skips one. | `tests/test_security_symlinks.py::TestOutputs::test_the_archive_route_does_not_follow_the_link`, `tests/test_security_symlinks.py::TestOutputs::test_the_gallery_listing_does_not_enumerate_the_link` |
| Engine/server import direction | `dw/`, `dw/server/`, `dw/workspace.py`: `EXPORTS_SUBDIR` | No engine module imports `dw.server` except the entry point `dw/serve.py`, and a constant both sides need lives engine-side. | `scripts/arch_metrics.py` ratchet `import_cycles` |
| Module and function size | `scripts/arch_metrics.py` | A module stays at or under 1,100 lines (with a warning above 1,000), and a function stays at or under 150 lines. | `scripts/arch_metrics.py` ratchets `modules_over_size_ceiling` and `functions_over_150_lines` |

## Server

| Concept | Owner | Rule | Enforced by |
| --- | --- | --- | --- |
| App factory | `dw/server/app.py`: `create_app` | `create_app` builds the JobManager and `app.state`, installs the middleware and registers the routers, and all state lives in the JobManager. | `tests/test_server.py` |
| Routers | `dw/server/routes/*.py`, `dw/server/routes/__init__.py`: `ROUTERS` | There is one router per resource, registered in `ROUTERS` order because a greedy `{name:path}` route must come after its more specific siblings. | `tests/test_server_downloads.py::test_download_workflow_sets_content_disposition_attachment` |
| Job queue | `dw/server/jobs.py`: `JobManager` | One runner thread runs jobs FIFO on the single worker, and a job carries its own `output_dir`, `asset_dir` and `workflow_dir`, so it stays in its workspace. | `tests/test_server_jobs.py` |
| Job-to-run link | `dw/server/job_record.py`, `dw/server/jobs.py`: `JobManager.realized` | A job records `run_id`, `run_dir` and `run_version`, and `JobManager.realized` reads that run's `workflow.json`, confined to the output root. | `tests/test_server_jobs.py::test_realized_reads_the_file_the_run_wrote`, `tests/test_server_jobs.py::test_realized_refuses_a_run_dir_that_escapes_the_output_root` |
| Job history | `dw/server/job_history.py` | Finished jobs persist in `jobs.sqlite` (WAL mode), so the Jobs view survives a restart. | `tests/test_server.py::test_job_history_survives_restart_and_reruns`, `tests/test_job_history_wal.py::test_the_jobs_database_uses_wal_mode` |
| HTTP security | `dw/server/http_security.py` | Four middlewares read the bind and token values from `app.state` at request time, and only a route marked `query_token_ok` accepts `?token=`. | `tests/test_security_auth.py::TestTheTokenGate::test_every_api_spelling_needs_the_token` |

## MCP

| Concept | Owner | Rule | Enforced by |
| --- | --- | --- | --- |
| `dw_mcp` stays torch-free | `dw_mcp/` | `dw_mcp` is a top-level package that talks to `dw.serve` over HTTP and imports no `dw` module, because `dw/__init__.py` pulls in torch. | `tests/test_mcp_server.py::TestStartupWeight::test_the_server_starts_without_importing_the_engine` |
| Tool surface | `dw_mcp/server.py`, `dw_mcp/tools_*.py` | Only these modules import the MCP SDK, each tool body is a one-line call into a handler, and the registration order is the listing order an agent reads. | `tests/test_mcp_server.py::test_the_wiring_table_covers_every_registered_tool`, `tests/test_mcp_server.py::test_the_stated_tool_count_is_the_registered_one` |
| Surface text budget | `dw_mcp/server.py`, `dw_mcp/tools_*.py` | The instructions and each tool description stay at or under 2,048 characters (Claude Code truncates past that), and the whole surface stays within `SURFACE_BUDGET`. | `tests/test_mcp_server.py::test_no_text_the_agent_reads_is_cut_off_by_the_client`, `tests/test_mcp_server.py::test_the_tool_surface_fits_the_budget` |
| Spending needs consent | `dw_mcp/diagnose.py` | `run_workflow` and `rerun_job` refuse until `acknowledged_cost` is set, and submitting returns at once while progress is polled from the event log. | `tests/test_mcp_diagnose.py::test_run_refuses_without_an_acknowledged_cost`, `tests/test_mcp_diagnose.py::test_rerun_refuses_without_an_acknowledged_cost` |
| API errors | `dw_mcp/client.py` | An API failure becomes a message a person can act on here, and nowhere else. | `tests/test_mcp_client.py::test_a_400_surfaces_the_servers_detail_verbatim` |

## UI

| Concept | Owner | Rule | Enforced by |
| --- | --- | --- | --- |
| The UI reads engine fields | `ui/src/lib/plan.ts`: `describePlan`, `ui/src/lib/results.ts`: `sectionBySubfolder`, `ui/src/lib/pages/` | The UI reads `plan`, `version` and `subfolder` as fields the server sends and derives nothing of its own, and it never sends `acknowledged_cost`. | `ui/src/lib/plan.test.ts`, `ui/src/lib/results.test.ts` |
