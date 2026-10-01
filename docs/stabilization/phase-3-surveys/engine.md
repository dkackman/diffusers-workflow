> Survey taken at `9e25c49a` (2026-09-30) for the Phase 3 plan. Line numbers are that commit's: re-verify before relying on them.

# Engine survey for the refactor plan (worktree /Users/don/src/dkackman/dw-stabilization, branch stabilization/phase-3-plan)

All line numbers are from the worktree as surveyed. The helper scripts (ast outline, import-edge/SCC, minimum-feedback-edge-set brute force, test coupling) were session scratch and are not kept; `scripts/arch_metrics.py` reproduces the cycle list.

Line-budget facts that frame everything:
- Files over 1,000 lines in dw/ + dw_mcp/ (excluding community_pipelines): result.py 1830, workflow.py 2229, pipeline.py 2415, arguments.py 1235, introspection.py 1242, security.py 1038 (the six targets), plus NOT named in the brief: dw/tasks/audio_utils.py 2229, dw/server/app.py 4521, dw/server/jobs.py 1567, dw_mcp/server.py 1344.
- Functions >=140 lines repo-wide (so "every function under 150" is wider than the six files): result.save_artifact 348 (811-1158), workflow.Workflow.run 567 (1203-1769), workflow.create_step_action 278 (1952-2229), pipeline.load_component 147 (1849-1995, under the limit), Pipeline.load 140 (281-420, under), chain.run_chain 140, media_info.probe_media 184, plan.estimate 201, serve.main 259, server/app.create_app 3781 (nested routes), app.gallery_frames 165, tasks/concat_videos.concat_videos 333, tasks/join_into_song 154, tasks/loop_bed.find_loop_bed 362, tasks/voice_attribution.attribute_voices 175, teacache._create_flux_teacache_forward 197 and teacache_forward 186, worker._handle_execute 165, dw_mcp/server.build_server 1280. The six named files only contain save_artifact, run, create_step_action as >=150.

---------------------------------------------------------------------------
## 1. dw/result.py (1830 lines) concern map

Module top (1-55): imports (torch, numpy, soundfile, diffusers.utils export_to_video/export_to_gif/encode_video/is_av_available at 9-14; `.events`, `.security`), constants MAX_BASE_NAME_LENGTH 28, DEFAULT_AUDIO_SAMPLE_RATE 29, DEFAULT_VIDEO_FPS 33.

| Concern | Span(s) | Lines | Members |
|---|---|---|---|
| Artifact size description (log text) | 36-48 | 13 | `_artifact_size`, `_file_size_mb` (328-332, 5) |
| Headroom / clipping / near-silent warnings (audio QC) | 56-75, 116-158, 167-325 | ~250 | `HEADROOM_WARN_DBFS` 56, `_peak_dbfs` 59-75 (17), `warn_without_headroom` 116-158 (43), `CLIPPED_WARN_DBFS` 167, `_UNPROBED` 175, `_probe_written_media` 178-188 (11), `warn_if_written_above_full_scale` 191-248 (58), `NEAR_SILENT_*` 255/260, `warn_if_written_near_silent` 263-325 (63). Pure functions over waveform/file path, emit_warning only. Clean extraction: `dw/audio_qc.py` (or into media_info/loudness siblings, see section 7). |
| Alpha flatten (image) | 80-113 | 34 | `ALPHA_CONTENT_TYPES` 80, `flatten_alpha_for` 83-113 (31) |
| Frame encoding prep | 335-368 | 34 | `frames_for_encoding` |
| Output path selection | 371-400 | 30 | `output_file_path` 371-386, `_dedupe_existing_path` 389-400 |
| Audio/video format tables | 406-455 | ~50 | `AUDIO_FORMATS` 406-417, `LOSSY_AUDIO_CONTENT_TYPES` 423-429, `AUDIO_WRITE_ARGUMENTS` 432, `AUDIO_WRITE_CHUNK_FRAMES` 435, `MUXED_VIDEO_CONTENT_TYPE` 438, `_NO_PROPERTY` 442, `MODULAR_VIDEO_KEYS`/`AUDIO_KEYS`/`SAMPLE_RATE_KEYS` 453-455 |
| Value types carried between steps | 458-518 | 61 | `class AudioVideo` 458-487 (30, `__init__` 465-487), `class AudioTrack` 490-518 (29, `__init__` 506-518). These are what tasks/video_utils.py:18 and tasks/audio_utils.py:493 import from result (the upward import). |
| Result class state + artifact bookkeeping | 521-706 | ~186 | `__init__` 528-567 (state: result_definition, _consumed_by_normalizer, result_list, metadata, saved_files, saved_shots, selected, _no_headroom_warned, _predicted_peak_dbfs, _artifact_cache), `set_metadata` 569, `add_result` 577-597, `retainable` 600-616, `get_artifacts` 618, `_artifacts_for` 631-645, `get_artifact_properties` 647-706 (60) |
| Result.save (dir validation, name resolution, per-result loop) | 708-809 | 102 | validates dir via validate_output_path, `file_base_name` validation (security), lazy import `refuse_active_content_type` (766), `guess_extension`, `emit_phase("saving")`, loops result_list, json branch inline (~785-795), delegates to save_artifact, collects `saved_shots` |
| Result.save_artifact | 811-1158 | 348 | see breakdown below |
| Video fps / conform | 1160-1285 | 126 | `video_fps` 1160-1190 (31), `conform_artifact` 1192-1255 (64; stamps fps, `_fit_audio_to_frames`+`_sample_axis` lazy from tasks.video_utils at 1238, `measured_num_samples` from shots at 1251), `conform_artifacts` 1257-1285 (29) |
| Mux write | 1287-1370 | 84 | `save_audio_video` (encode_video mux, av availability fallback) |
| Audio write args | 1372-1392 | 21 | `get_audio_write_arguments` |
| Image metadata embed | 1394-1432, 1435-1465 | 70 | `_save_image_with_metadata` 1394-1432 (39), module fn `read_embedded_metadata` 1435-1465 (31) |
| Extracting artifacts out of pipeline outputs | 1468-1663 | ~196 | `_frames_from_attributes` 1468, `_audios_from_attribute` 1481, `OUTPUT_FIELD_EXTRACTORS` 1503, `get_artifact_list` 1512-1551 (40), `output_field_names` 1554, `modular_artifacts` 1563-1606 (44), `first_item` 1609, `frames_with_audio` 1622-1642 (21), `pair_audio_with_frames` 1645-1663 (19) |
| Audio array helpers + writer | 1666-1805 | 140 | `as_waveform_array` 1666, `as_audio_track` 1686, `_as_stereo` 1711, `write_audio` 1739-1762 (24), `normalize_audio` 1765-1805 (41; NB this is result.normalize_audio, distinct from tasks.audio_utils.normalize_audio) |
| Extension guessing | 1808-1830 | 23 | `guess_extension` |

Natural target modules (result.py would keep only `Result`): (a) `dw/media_types.py` (or `audio_video.py`): AudioVideo, AudioTrack, MODULAR_*_KEYS, `_NO_PROPERTY`, DEFAULT_* constants, with zero dw dependencies; (b) `dw/audio_qc.py` (warnings block, ~250 lines); (c) `dw/output_extract.py` (get_artifact_list/modular_artifacts/frames_with_audio/pair_audio..., ~196); (d) `dw/audio_write.py` (AUDIO_FORMATS, write_audio, as_*, normalize_audio, get_audio_write_arguments, ~190); (e) `dw/image_write.py` (flatten_alpha_for, metadata embed, read_embedded_metadata, frames_for_encoding ~110).

### save_artifact section by section (811-1158)
1. 811-826 signature + docstring.
2. 827-829 `None` artifact -> warn, return [].
3. 831-843 scalar (int/float/bool) guard -> ValueError (#212 comment block).
4. 844-880 dict artifact: recursive save per key; raw `torch.Tensor` under non-audio content_type skipped with `emit_warning(kind="non_media_artifact_skipped")` (#507) (this is where the H3 'latents' comment is).
5. 882-900 path resolution (`output_file_path`), `emit_log("writing ...")` (#97), `started = time.monotonic()`, resets `self._no_headroom_warned`/`_predicted_peak_dbfs`.
6. 901-1006 the write dispatch (big try block), by content_type:
   - 902-906 video: AudioVideo -> `self.save_audio_video`, else `export_to_video`.
   - 907-908 image/gif -> `export_to_gif`.
   - 909-967 audio (59 lines): `normalize_audio(artifact)`; declared-vs-carried sample_rate and the #205 relabel warning (lazy import `tasks.audio_utils._warn_on_rate_override` at 926); batched waveforms recurse per track (AudioTrack rebuilt) and return early; headroom warning `warn_without_headroom` -> `self._no_headroom_warned`; `write_audio`.
   - 968-970 json write.
   - 971-989 text write (+ ValueError for non-str, #498).
   - 990-1004 objects with `.save` (PIL): alpha flatten, optional `_save_image_with_metadata`.
   - 1005-1006 else ValueError.
7. 1007-1012 `except Exception` -> log and re-raise.
8. 1014-1150 post-write verification for audio/video only (`if content_type.startswith("audio") or ...` at 1020):
   - 1025 `probed_info = _probe_written_media(output_path)` (one decode shared, #262).
   - ~1038-1096 shot-map re-measure for video with `artifact.shots`: lazy `shots.measured_num_samples` (1045), lazy `tasks.video_utils.AUDIO_FIT_TOLERANCE_SECONDS` (1065), computes shortfall vs frame grid, either `emit_log` (1072) or `emit_warning(kind="joined_audio_short_after_mux")`.
   - 1097-1119 `warn_if_written_above_full_scale(...)` with already_warned logic.
   - ~1120-1141 late `audio_no_headroom` warning when the probe failed but the prediction exists.
   - 1142-1150 `warn_if_written_near_silent(... source_already_quiet ...)`.
9. 1152-1158 final `emit_log("wrote ...")`, `return [output_path]`.

Cut proposal (each <150): `_save_dict_artifact` (844-880, ~40), `_write_audio(self, artifact, output_path, content_type, output_dir, ...)` (909-967, ~60; needs the recursion for batched waveforms so returns a list or a sentinel), `_write_by_content_type` dispatcher (901-1006, ~60 after audio is extracted), `_verify_written(self, artifact, output_path, content_type)` (1020-1150) itself split into `_remeasure_shots_after_mux` (~60), `_warn_written_level` (~55). Instance state touched: `self.result_definition`, `self.metadata`, `self._consumed_by_normalizer`, `self._no_headroom_warned`, `self._predicted_peak_dbfs` (the last two are written in warn paths and read in step 8; a small `_LevelState` dataclass would make the verify step pure).

Cycle-relevant imports in result.py: top-level `.events`, `.security`; lazy: `.content_types` (766), `.tasks.audio_utils` (926), `.shots` (1045, 1251), `.tasks.video_utils` (1065, 1238).

---------------------------------------------------------------------------
## 2. dw/pipeline_processors/pipeline.py (2,415 lines)

Top (1-70): imports, `_CACHE_CONTEXT_NAME` 35, `optional_component_names` 37-54 (generic diffusers component names incl. `transformer_2`, `text_encoder_3`, `prompt_enhancer_head`), `_NON_COMPONENT_KEYS`.

| Concern | Span | Lines | Members |
|---|---|---|---|
| Component-name / copy helpers | 71-114 | 44 | `declared_component_names` 71-89, `_loading_copy` 92-114 |
| Pipeline class: lifecycle/loading | 117-463 | ~347 | `Pipeline.__init__` 123-158, `resolve_reused_components` 187-208, `populate_from_pretrained_arguments` 210-241, `check_trusted`+`_check_trusted_block` 243-279, `load` 281-420 (140), `_discard_failed_load` 422-439, `publish_shared_components` 441-461 |
| Pipeline class: running | 464-651 | ~188 | `run` 464-523 (60), `_run_once` 525-563, `_execute_pipeline` 565-587, `_call_pipeline` 589-614, `_takes_step_callback` 616, `_with_step_callback` 628-651 (progress callback wrapping) |
| Pipeline class: optional components/config | 653-749 | 97 | `load_optional_component` 653-692, `configure_loaded_components` 694-749 (vae/unet/transformer blocks via self.configuration) |
| Per-component configuration | 752-860 | 109 | `configure_components` (group offload, tiling, attn processor, device, residency branches) |
| Tiling / attention processor | 863-934 | 72 | `enable_tiling` 863-896, `set_attn_processor` 899-934 |
| Module surgery (truncate layers) | 937-1047 | 111 | `_resolve_submodule`, `truncate_module_lists` 965-1021, `replace_modules_with_identity` |
| On-demand residency | 1055-1140 | 86 | `apply_on_demand_placement` (wraps forward/encode/decode; functools.wraps) |
| torch.compile | 1143-1179 | 37 | `apply_compile` |
| Component lookup / safety checker | 1185-1254 | 70 | `get_component`, `warn_if_safety_checker_blanked` |
| Audio sample rate attach | 1258-1308 | 51 | `_SAMPLE_RATE_SOURCES` 1258, `_component_sample_rate`, `attach_audio_sample_rate` |
| LoRAs / adapters | 1311-1467 | 157 | `set_adapter_alpha` 1311-1364, `_lora_layers`, `active_loras` 1395, `load_loras` 1407-1456, `load_ip_adapter` 1459 |
| Scheduler | 1470-1527 | 58 | `load_and_configure_scheduler` |
| MPS accommodations | 1530-1590 | 61 | `apply_mps_rope_precision` 1530-1556, `attention_slicing_requested`, `auto_cpu_offload_enabled/_active` |
| Components manager / offload placement | 1593-1846 | ~254 | `create_components_manager` 1593-1638, `has_component_group_offload`, `loading_device` 1671, `get_block_configs` 1698-1748 (modular block configs), `place_component` 1751-1846 (96) |
| Component loading + diagnostics | 1849-2040 | 192 | `load_component` 1849-1995 (147), `_diagnose_image_crf_error` 1998-2026, `_hub_auth_status` 2029-2040 |
| Quantized-matmul optimisation | 2043-2090 | 48 | `apply_sdnq_optimizations` |
| Cache transformer (teacache/cache hooks) | 2093-2111, 2345-2415 | ~57 | `get_cache_transformer`, `stateful_cache_context` 2345-2375, `enable_cache_on_transformer` 2378-2415 |
| Progress-bar / block reporting | 2114-2341 | 228 | `_ReportingProgressBar` 2114-2158, `_progress_bar_holders`, `reported_progress_bars` 2193-2249, `_runs_its_blocks_in_sequence`, `reported_blocks` 2270-2341 |

Natural split: pipeline.py keeps `Pipeline` (117-749 + helpers 71-114, ~700 lines); new modules: `placement.py` (on_demand, place_component, components manager, loading_device, MPS offload helpers, ~450), `components.py` (configure_components, tiling, attn, truncate, compile, get_component, load_component, diagnostics, sdnq, ~600), `adapters.py` (loras, alpha, ip_adapter, scheduler ~215), `reporting.py` (progress bars + block reporting ~230), `caching.py` (cache transformer+context ~57). Only 3 names are patched in tests (section 6).

### Model-family-specific names in engine code (result.py, pipeline.py, workflow.py)
Code (behaviour-bearing), as opposed to comments:
- pipeline.py:2107 `for name in ("transformer", "transformer_ref")` in `get_cache_transformer` - `transformer_ref` is MiniMax-H3's ref2va denoiser (docstring 2097-2098).
- pipeline.py:1258-1264 `_SAMPLE_RATE_SOURCES = (("vocoder","output_sampling_rate"),("vocoder","sampling_rate"),("vae","sampling_rate"))` - attribute names per audio family (LTX-2 vs AudioLDM2/StableAudio); and pipeline.py:1269 `component_name == "vae" and not has_audios`.
- pipeline.py:37-54 `optional_component_names` incl. "transformer_2" (Wan 2.2 style), "text_encoder_3" (SD3/Flux), "prompt_enhancer_head" - family-flavoured but generic component slots.
- pipeline.py:1543-1556 `apply_mps_rope_precision`: keyed on a `double_precision` attribute, not a name (comment cites Wan/Lumina2/LTX-2.5) - behaviour is family-agnostic.
- result.py:453-455 `MODULAR_VIDEO_KEYS=("videos",)`, `MODULAR_AUDIO_KEYS=("audio",)`, `MODULAR_SAMPLE_RATE_KEYS=("sampling_rate","audio_sample_rate")` - the modular-pipeline output key names (comment 447 names minimax_h3/ltx2); `OUTPUT_FIELD_EXTRACTORS` 1503 and `modular_artifacts` 1563-1606 read them.
- result.py:853 and :851-858 comment-only reference to H3 'latents'; the code there skips ANY raw tensor under a non-audio content type (family-agnostic).
- workflow.py: nothing behavioural; 217/448-449 'ltx' appears only as an example directory name in docstrings.
Comments/docstrings only (no code branch): result.py:89 (Qwen-Image alpha), 205 (H3 mux), 259 (Bark), 461, 501, 1471 (LTX-2); pipeline.py:620, 816, 822, 867, 903-904 (LTX-2.5 diffusion_decoder), 969-970 (H3 hidden_states[50] of Qwen3-VL in `truncate_module_lists` docstring), 1125, 1226 (SD 1.5 safety checker), 1283, 1318, 1372, 1480 (H3 dual schedules), 1533-1536, 1555, 1567-1568 (SD 1.5/SDXL attention slicing), 1702 (H3 declares three block configs), 2196, 2278, 2296.
Elsewhere in dw/ (for context, counts of minimax/ref2va/fl2va/ltx mentions): adapter_compatibility.py 11 (H3-specific by design, file-name convention), arguments.py 8, security.py 7, runs.py 5, tasks/video_utils.py 4, reference_limits.py 4, vram_inheritance.py 2, etc. Most of those are docstring references; the behaviour-bearing ones are adapter_compatibility.py, reference_limits.py, the H3 block in vram_estimate/variable_constraints (declared in workflow JSON, not engine code).
Conclusion: only `transformer_ref` (pipeline.py:2107) and the modular output key tuples (result.py:453-455) are genuine family names in code; the rest of the assessment's "model-family names in engine code" in these three files is commentary. They can move to configuration (a `cache_transformer_names` tuple in pipeline definition / a constants module).

---------------------------------------------------------------------------
## 3. dw/workflow.py (2,229 lines)

Top: imports 1-113 (many: `.arguments`, `.events`, `.previous_results`, `.adapter_compatibility`, `.elision`, `.variable_constraints`, `.shots`, `.subfolders`, `.validation`, `.vram_estimate`, `.step`, `.step_cache`, `.runs`, `.realize`, `.schema`, `.for_each`, `.variables`, `.pipeline_processors.pipeline`, `.tasks.model_cache`, `.tasks.task`, `.host_memory`, `.security`, `.workflow_sources`), `SEED_BITS` 118.

Module-level (114-325): `ConstantError` 121-127; factories `workflow_from_file` 130-158, `workflow_from_definition` 161-188, `workflow_from_snapshot` 191-210; `workflow_output_subfolder` 213-234; `catalog_root_dir` 237-248; `_allocated_mb` 251, `_release_host_caches` 262, `_relative_shots` 277; `selected_field` 286-308; `release_unreferenced_results` 311-325.

`class Workflow` 328-2229 (1,902 lines - the class itself exceeds the 1,000 limit):
| Concern | Span | Lines |
|---|---|---|
| Construction, naming, output dirs | 383-481 | `__init__` 383-392, `step_save_name` 406, `_parent_progress_fields` 418, `step_file_prefix` 430, `effective_output_dir` 439-461, `step_output_dir` 463-481 |
| Variable folding + for_each expansion | 483-604 | `_fold` 483-533, `_expand` 536-552, `_folded_and_expanded` 554-575, `expanded_definition` 577-599, `folded_variables` 601 |
| Sub-workflow resolution/validation | 606-738 | `resolve_sub_workflow_path` 606-639, `sub_workflow_errors` 641-692, `sub_workflow_warnings` 694-706, `sub_workflow_argument_warnings` 708-738 |
| Validation | 740-964 | `validation_context` 740-773, `validation_errors` 775-836 (62), `_warning_check` 838, `adapter_warnings` 847, `inherited_vram_warnings` 858, `slice_past_end_warnings` 873, `shot_span_warnings` 883, `null_variable_argument_warnings` 894, `_undeclared_variable_errors` 910-945, `validate` 947-964 (these ~225 lines belong in the existing `dw/validation.py`, 879 lines, which already holds `ERROR_CHECKS`/`WARNING_CHECKS`) |
| Prepare + step cache | 966-1201 | `_prepare_definition` 966-1055 (90), `_cache_lookup` 1057-1138 (82), `cache_hits` 1140-1187, `_owned_arguments` 1189-1201 |
| run | 1203-1769 | 567 |
| Run manifest | 1771-1837 | `_write_run_manifest` 67 (belongs in runs.py, 894 lines - careful, runs.py is also large) |
| Pipeline release/borrow | 1839-1950 | `_step_pipeline_key` 1839, `_finish_release` 1845-1872, `_load_deferred_borrows` 1874-1927, `_load_key` 1929-1950 (~110) |
| create_step_action | 1952-2229 | 278 |

### run (1203-1769) section by section
1. 1203-1217 signature + docstring.
2. 1218-1262 (45) context/bookkeeping setup: RunContext activate + `enter_run`, `activate_output_root`, resets of `_pipeline_keys_by_step`, `_running_pipeline_keys`, `_deferred_pipelines`, `_prior_step_keys`, `manifest`, `_elided_steps`; locals status/run_id/resolved_seed/realized_name/annotations/started_at.
3. 1263-1318 (56) definition prep: `_owned_arguments` if composed, deepcopy, `base_dir`, `_prepare_definition`, `warn_adapters`, `warn_elided`, cache-enabled flag, seed resolution (`secrets.randbits(SEED_BITS)`).
4. 1319-1393 (75) run directory: `new_run_id`/`open_run` (or flat layout), `realize_workflow`/`write_realized_workflow`, initial `_write_run_manifest("running")`, emits `run_start`.
5. 1394-1443 (50) state init: `StepResults`, `shared_components`, pipelines cache, empty-steps early return, `realize_args(steps, base_dir)`, `step_pipeline_keys`, emit `workflow_start`, `hits_this_run`.
6. 1444-1712 (270) the per-step loop body - the thing to extract as `_run_step(...)`:
   - 1444-1456 cancel check, `step_start` emit;
   - 1458-1470 `Step(...)` construction with `normalized_downstream` (consumed_by_normalizer);
   - 1472-1497 `_cache_lookup`, `parent_saves_this`;
   - 1498-1527 `_load_deferred_borrows`, `create_step_action`, sub-workflow parent-progress / `_final_save_owned_by_parent` wiring;
   - 1529-1538 reuse vs `step.run`;
   - 1539-1578 sub_manifest capture + `release_pipeline` handling (`_finish_release`);
   - 1579-1610 save (`result.conform_artifacts()` if parent owns the save, else `result.save(step_output_dir, step_save_name)`), `step_cache.put`;
   - 1611-1660 manifest entry (subfolder, reused, selected, shots + `shot_name_collision` warning);
   - 1661-1677 sub-manifest merge, `step_end` emit;
   - 1687-1699 manifest rewrite ("running");
   - 1700-1724 `release_unreferenced_results`, `release_models` (clear_model_cache, `_release_host_caches`), gc + `empty_device_cache`.
7. 1725-1731 workflow_end emit, status completed, return last result list.
8. 1733-1752 except blocks (WorkflowCancelled, security errors, generic) - log + re-raise.
9. 1753-1769 finally: final `_write_run_manifest(status)`, `deactivate_output_root`, `exit_run`, `deactivate_context`.
Cut proposal: `_begin_run` (2-4, returns a small RunState dataclass holding run_id, started_at, seed, realized_name, annotations), `_init_run_state` (5), `_run_step` (6, itself split: `_prepare_step`, `_execute_or_reuse`, `_release_after_step`, `_save_and_cache`, `_record_step`, `_cleanup_after_step`), `_finish_run` (8-9).

### create_step_action (1952-2229) section by section
1. 1952-1971 signature, docstring.
2. 1972-2062 (~90) `"pipeline" in step_definition` branch:
   - 1973-1985 cache key, `_step_pipeline_key`, `touch_pipeline`; cache-hit-and-not-resident -> record `_deferred_pipelines[step]`, return None.
   - 1987-2019 (~33) resident: wrap cached model in a new `Pipeline(...)` with fresh generator seeded from `seed`, `emit_phase("cached")`.
   - 2020-2050 (~30) redefined-step eviction: `prior_key` vs `cache_key`, `still_shared` check, pop, gc, `empty_device_cache`, `_release_host_caches`, `pipeline_released` event (#reason superseded).
   - 2051-2062 fresh load: `Pipeline(...)`, `check_trusted`, `emit_phase("loading")`, `load(shared_components)`, store.
3. ~2063-2090 (~28) `"pipeline_reference"` branch: lookup `_pipeline_keys_by_step`, deferred-on-hit -> None, error if not resident, return `Pipeline` over the referenced model.
4. ~2091-2218 (~128) `"workflow"` branch: builtin name validation (bare `<name>.json`, `builtin_root()`), else `resolve_sub_workflow` with confine_to logic, `validate_workflow_path`, `workflow_from_file`, set `argument_template`, seed, `_cache_enabled_by_parent`, `_composed`, `_run_dir`, `_run_dir_inherited`, `validate()`. (`resolve_sub_workflow_path` 606-639 already duplicates part of this resolution.)
5. 2219-2229 task branch: `Task(task_definition, device, seed=...)`.
Cut proposal: `_pipeline_action` (2, split into `_reuse_resident_pipeline`, `_evict_superseded`, `_load_fresh_pipeline`), `_reference_action` (3), `_sub_workflow_action` (4, sharing `resolve_sub_workflow_path`), `_task_action` (5); dispatcher stays ~15 lines.

Observation for the splitter: `Workflow` has ~25 attributes set ad hoc across run (`_deferred_pipelines`, `_running_pipeline_keys`, `_prior_step_keys`, `_pipeline_keys_by_step`, `_cache_enabled_this_run`, `_elided_steps`, `_composed`, `_run_dir`, `_run_dir_inherited`, `_final_save_owned_by_parent`, `_parent_progress`, `_cache_enabled_by_parent`; `hasattr`/`getattr(self,"_deferred_pipelines")` guards at 1969-1971, 2004, 2074). The pipeline-ownership cluster (`_step_pipeline_key`, `_finish_release`, `_load_deferred_borrows`, `_load_key`, step action pipeline branch) is a natural `PipelineRegistry`/`borrow.py` class holding those 4 dicts.

---------------------------------------------------------------------------
## 4. arguments.py, introspection.py, security.py

### dw/arguments.py (1,235)
Top-level imports include `.runs` (12) and `.locations` (25) plus `references`. Constants 27-100: `EscapedString` 34-43, `is_escaped` 46, `FROM_FILE_KEY`/`FROM_PREVIOUS_RESULT_KEY`/`FROM_ARGUMENTS_KEY` 52-59, `PREVIOUS_RESULT_PREFIX = references.PREVIOUS_RESULT` 64, `CONSTANT_PREFIX` 69, `_Omitted`/`OMITTED` 72-88, `ALLOWED_FROM_FILE_EXTENSIONS` 93, `MEDIA_KINDS` 100.
| Concern | Span | Lines |
|---|---|---|
| Reference realization walk | `realize_args` 104-235 (132), `is_path_reference` 238, `resolve_path_references` 244-267, `is_constant_reference` 270, `is_prompt_reference` 275, `fetch_constant` 280-321, `realize_constants` 324-346 | ~245 |
| Object construction from dicts (type system) | `is_media_reference` 349, `fetch_media` 360, `object_type_key` 384, `_names_no_media` 412, `realize_object` 427-534 (108), `split_from_file_arguments` 537, `apply_field_overrides` 569, `validate_deferred_object` 597, `validate_constructed_object` 633, `names_a_previous_result` 676, `construct_object` 685, `build_objects` 711, `object_from_result` 753-792 | ~450 |
| Media argument resolution | `_carried_audio` 795, `media_arguments` 804-880 (77), `validate_media_location` 883, `resolve_relative_path` 914, `_describe_value_source` 925, `_fetch_image_with_context` 936, `_fetch_video_with_context` 950, `fetch_image` 963-1039 (77), `_with_frame_rate` 1042, `_fetch_remote_video` 1069, `fetch_video` 1091-1179 (89) | ~385 |
| Lazy frame commands | `_LAZY_FRAME_COMMANDS` 1186, `_realize_lazy_frame_arguments` 1198-1235 | ~50 |
Natural split line: at 349 - `arguments.py` keeps reference/constant realization + object construction (lines 1-795, ~800 after trimming), and a new `dw/media_arguments.py` (or `media_loading.py`) takes 795-1235 (media_arguments, fetch_image, fetch_video, lazy frame args, ~440 lines). It is also the lazy importer of `tasks.audio_utils`/`tasks.video_utils`/`runs` (824-825, 1055-1056, 1075, 1209), so moving it isolates those cycle edges. Constants (FROM_*_KEY, PREVIOUS_RESULT_PREFIX) should move to `references.py` (already its home for prefixes) so `for_each`, `variables`, `shots` stop importing `arguments`.

### dw/introspection.py (1,242)
Constants 21-69 (`COMPONENT_LOADING_KNOBS` 40-69). 
| Concern | Span | Lines |
|---|---|---|
| Pipeline/class listing + loading | `_filtered_exports` 72, `list_pipelines` 81, `list_classes` 90, `load_allowed_class` 115-143, `load_pipeline_class` 147 | ~75 |
| Class/docstring description | `_json_safe_default` 150, `_parse_docstring_args` 158-242 (85), `_callable_parameters` 245-291, `describe_class` 294-346, `describe_pipeline` 349, `unknown_call_arguments` 354, `unknown_pipeline_components` 373 | ~235 |
| Task description + signature errors | `list_tasks` 396, `_first_paragraph` 424, `describe_task` 429-541 (113), `unknown_task_arguments` 544, `..._message` 563, `missing_task_arguments` 572, `..._message` 597, `null_variable_task_argument_message` 608, `_null_fed_variable` 625, `task_signature_errors` 650-776 (127) | ~380 |
| Type-reference validation | `_TYPE_REFERENCE_KEYS` 779, `_DOTTED_NAME_PATTERN` 784, `_type_reference_candidates` 789, `_type_reference_error` 798, `_is_type_key` 832, `_loose_type_reference_error` 844, `_constant_reference_error` 865, `_walk_type_references` 883, `component_type_errors` 909-959, `component_name_errors` 962-1037 | ~260 |
| Inert-argument warnings | `_resolved_value` 1040, `_inert_crossfade_warnings` 1056, `_inert_seam_fade_warnings` 1074, `_inert_bleed_gain_warnings` 1102, `_inert_bleed_single_input_warnings` 1132, `_inert_match_levels_dbfs_warnings` 1152, `workflow_argument_warnings` 1177-1242 | ~205 |
Natural split line: introspection stays describe_* / list_* (72-653 + 349-647, ~600) ; move 779-1053 to `dw/type_reference_validation.py` (or into existing `dw/validation.py`? it is 879 lines - better a new module) and 1056-1242 (inert-argument warnings, task/audio-specific, ~190) to `dw/argument_warnings.py`. Imports at top: only `references` and `.variables.undeclared_variable_references`; lazy imports of `tasks.task` (407, 442, 493, 532, 692) are the cycle edge.

### dw/security.py (1,038)
| Concern | Span | Lines |
|---|---|---|
| Limits, extension sets, error classes | 12-75 (constants 15-43, `SecurityError` 46, `PathTraversalError` 52, `InvalidInputError` 58, `UntrustedWorkflowError` 64-75) | ~65 |
| Trust model (dotted-name / remote-code gating) | 82-344: `TRUST_WORKFLOWS_ENV_VAR` 82, `TRUSTED_TOP_LEVEL_PACKAGES` 92, `CONSTRUCTIBLE_*` 118-159, `_constructible_bases` 172, `is_constructible_class` 184, `require_constructible_class` 222, `set_trust_workflows` 243, `workflows_are_trusted` 253, `require_trusted_dotted_name` 267, `require_trusted_pre_load_modules` 297, `REMOTE_CODE_ARGUMENTS` 315, `require_trusted_from_pretrained_arguments` 318 | ~265 |
| Path / file / URL / subprocess validators | `validate_path` 347-427 (81), `contained` 430, `validate_file_extension` 444, `validate_workflow_path` 464, `validate_prompt_path` 470, `validate_output_path` 476, `validate_url` 481, `sanitize_command_args` 518, `validate_variable_name` 554, `validate_json_size` 967, `validate_string_input` 985, `safe_join_path` 1015 | ~320 |
| Reference-name shape validators | `PROMPT_REFERENCE_*` 587-589, `_name_fault` 592-619, `_validate_name` 622-657, `validate_prompt_reference` 660, `ASSET_REFERENCE_*` 697-699, `validate_asset_reference` 702, `OUTPUT_REFERENCE_*` 741-743, `validate_output_reference` 746, `SUBFOLDER_PATTERN`/`validate_subfolder` 780-808, `validate_file_base_name` 811, `validate_content_type` 830, `WORKSPACE_NAME_PATTERN`/`validate_workspace_name` 858-892, `CONSTANT_NAME_PATTERN`/`validate_constant_name` 896-931, `COMMIT_HASH_PATTERN`/`validate_commit_hash` 939-964 | ~330 |
Natural split: `security.py` keeps errors + path/url/subprocess validators (lines 12-75 + 347-579 + 967-1038, ~370); `dw/security_trust.py` takes the trust model (82-344, ~265); `dw/name_validation.py` takes the reference-name/workspace/constant/commit validators (580-964 minus validate_variable_name, ~380). Needs re-exports from `security` (27 test files import from it; many `dw/` modules do). CodeQL local query pack models `dw.security` validators as sanitizers (.github/codeql/dw-security/) - a moved validator must be re-modelled there, or re-exported from `security` and modelled by name; check the pack before moving `validate_path`/`validate_output_path`.

---------------------------------------------------------------------------
## 5. Import cycles

Method: ast walk over all dw/**/*.py (top-level and lazy imports, TYPE_CHECKING not special-cased), Tarjan SCC - reproduces the brief's four cycles and finds one more: `dw.server.app <-> dw.server.mcp_mount`.

### 5a. The 11-module SCC: every edge among the 11 modules (from -> to [names], file:line, kind)
arguments:
- arguments -> runs [fetch_output, is_output_reference] arguments.py:12 top
- arguments -> locations [safe_get, validate_media_path] arguments.py:25 top
- arguments -> tasks.audio_utils [as_channels_samples] arguments.py:824 lazy
- arguments -> tasks.video_utils [frames_as_pil_list] arguments.py:825 lazy
- arguments -> runs [shots_beside] arguments.py:1055 lazy
- arguments -> tasks.video_utils [FrameList, file_fps] arguments.py:1056 lazy and :1075 lazy
- arguments -> tasks.video_utils [VideoFileReference] arguments.py:1209 lazy
content_types:
- content_types -> for_each [MEMBER_SEPARATOR, render_path] content_types.py:29 top (for_each itself re-exports these from `references`: for_each.py:32-33)
- content_types -> result [AUDIO_FORMATS, MUXED_VIDEO_CONTENT_TYPE] content_types.py:30 top
for_each:
- for_each -> arguments [FROM_PREVIOUS_RESULT_KEY, PREVIOUS_RESULT_PREFIX] for_each.py:34 top
- for_each -> variables [argument_errors, set_variables] for_each.py:37 top
locations:
- locations -> runs [output_root] locations.py:92 lazy
media_frames:
- media_frames -> tasks.video_utils [_compose_grid, _default_columns, _evenly_spaced_indices, _format_timestamp, _grid_tile] media_frames.py:16 top (5 private helpers)
result:
- result -> content_types [refuse_active_content_type] result.py:766 lazy
- result -> tasks.audio_utils [_warn_on_rate_override] result.py:926 lazy
- result -> shots [measured_num_samples] result.py:1045 and :1251 lazy
- result -> tasks.video_utils [AUDIO_FIT_TOLERANCE_SECONDS] result.py:1065 lazy; [_fit_audio_to_frames, _sample_axis] result.py:1238 lazy
runs:
- runs -> shots [shots_for_file] runs.py:747 and :838 lazy
shots:
- shots -> arguments [PREVIOUS_RESULT_PREFIX] shots.py:38 top
tasks.audio_utils:
- -> tasks.video_utils [load_audio_video] audio_utils.py:445 lazy
- -> locations [safe_get] :456 lazy; [validate_media_path] :464 lazy
- -> result [AudioTrack] audio_utils.py:493 lazy
tasks.video_utils:
- -> result [AudioVideo] video_utils.py:18 top
- -> media_frames [frames_at] video_utils.py:72 lazy
- -> locations [safe_get, validate_media_path] video_utils.py:495 lazy
- -> runs [shots_beside] video_utils.py:516 lazy
variables:
- variables -> arguments [FROM_ARGUMENTS_KEY, FROM_FILE_KEY, FROM_PREVIOUS_RESULT_KEY] variables.py:4 top
(24 distinct module pairs; ~46 import statements.)

### 5b. Minimal edge set that breaks the SCC
Brute force over the 24 module-pair edges: the minimum feedback edge set is **5 pairs** (12 equivalent solutions found). Canonical one:
1. `shots -> arguments` (PREVIOUS_RESULT_PREFIX, shots.py:38). Fix: `shots` uses `references.PREVIOUS_RESULT` directly (arguments.py:64 is literally `references.PREVIOUS_RESULT`). Same trivial fix applies to `for_each -> arguments` and `variables -> arguments` (move `FROM_FILE_KEY`, `FROM_PREVIOUS_RESULT_KEY`, `FROM_ARGUMENTS_KEY`, plus `PREVIOUS_RESULT_PREFIX`/`CONSTANT_PREFIX` aliases into `references.py`, keep re-exports in arguments) - not strictly required for minimality, but it removes arguments from the lower layer entirely.
2. `content_types -> result` (AUDIO_FORMATS, MUXED_VIDEO_CONTENT_TYPE; 2 names). Fix: move the format tables (result.py:406-417 `AUDIO_FORMATS`, :438 `MUXED_VIDEO_CONTENT_TYPE`, and naturally `LOSSY_AUDIO_CONTENT_TYPES` 423) into content_types.py (or a new leaf `dw/media_formats.py`) and have result import them. Also switch content_types's `for_each` import to `references` (MEMBER_SEPARATOR and render_path are defined in references.py:87 already) - removes `content_types -> for_each`.
3. `tasks.video_utils -> result` (AudioVideo, video_utils.py:18 top) and 4. `tasks.audio_utils -> result` (AudioTrack, audio_utils.py:493 lazy). Fix: move `AudioVideo` (result.py:458-487) and `AudioTrack` (490-518) with their `MODULAR_*` key constants and `_NO_PROPERTY` into a leaf module (`dw/media_types.py`) with no dw imports; result, both task modules, and arguments import from it. 59 lines move; result re-exports the two names (37+10 test imports use `from dw.result import AudioVideo/AudioTrack`, so keep the re-export).
5. `media_frames -> tasks.video_utils` (5 private helpers `_compose_grid`, `_default_columns`, `_evenly_spaced_indices`, `_format_timestamp`, `_grid_tile`; video_utils.py:281-335 ~55 lines) - OR equivalently cut `video_utils -> media_frames` (lazy, video_utils.py:72, one name `frames_at`). Cheapest: move those 5 helpers (video_utils.py:281-335) into media_frames.py (their only other consumer is `frame_grid` in video_utils, which would then import them from media_frames lazily-or-top, since media_frames no longer imports video_utils).
Then `result -> tasks.audio_utils` and `result -> tasks.video_utils` remain as edges but are no longer in any cycle (leaves the layering "result above tasks-utils", which still violates the assessment's 'result imports upward into tasks'). To fully invert: move the four helpers result uses out of tasks into a lower module - `_warn_on_rate_override` (audio_utils.py:1906, needs emit_warning only) , `AUDIO_FIT_TOLERANCE_SECONDS` (video_utils.py:582), `_fit_audio_to_frames` (585-~622), `_sample_axis` (623-~637), ~60 lines total - e.g. into `media_audio.py`/`media_types.py`; `measured_num_samples` (shots.py:214) is already in a low module. After that result.py has no dw.tasks imports at all.
Simulated result: cutting exactly {shots>arguments, for_each>arguments, variables>arguments, content_types>result, content_types>for_each, video_utils>result, audio_utils>result, media_frames>video_utils, result>video_utils, result>audio_utils} leaves none of the 11-module SCC (verified with scc.py). Cutting only the 5-edge canonical set also does (verified by mfas.py: the 12 solutions each leave the graph acyclic).
Other remaining lazy edges (`arguments -> runs/locations/tasks.*`, `runs -> shots`, `locations -> runs`, `audio_utils/video_utils -> locations/runs`) then all point downward and become acyclic; they may optionally be promoted to top-level imports.

### 5c. The three 2-module cycles
1. `dw.introspection <-> dw.tasks.task`:
   - introspection -> task [`_COMMAND_INFO`, `_COMMAND_REGISTRY`, `_VIDEO_PROCESSOR_COMMANDS`, `task_command_info`] introspection.py:407, 442, 493, 532, 692 (all lazy; it needs the populated command registry; importing task is what registers handlers).
   - task -> introspection [`missing_task_argument_message`, `missing_task_arguments`] tasks/task.py:873 lazy inside `Task._check_required_arguments` (~867-885).
   Smallest fix: cut the single `task -> introspection` edge. Options: (a) have the caller of `Task` run the check (dw/step.py already sits above both; `Step.run`/`Workflow.create_step_action` call `missing_task_arguments(command, arguments.keys())` before `Task.run`); (b) inject the checker into Task; (c) move `describe_task`'s registry reading to a `tasks/registry.py` (registry dicts `_COMMAND_REGISTRY` task.py:25, `_COMMAND_INFO` 33, `register_command` 36-122, `task_command_info` 125-142, `_VIDEO_PROCESSOR_*` ~785) so both import it - but registry population is a side effect of importing task.py's handlers, so (c) still needs introspection to import task lazily; (a) is cleaner. Only `missing_task_arguments` and `missing_task_argument_message` (introspection.py:572-605, ~34 lines) are at stake.
2. `dw.security <-> dw.workspace`:
   - security -> workspace [`RESERVED_WORKSPACE_NAMES`] security.py:875 lazy (in `validate_workspace_name`).
   - workspace -> security [`validate_workspace_name`] workspace.py:465, 486 lazy.
   Fix: move `RESERVED_WORKSPACE_NAMES` (workspace.py:95 = `SUBDIRS + (EXPORTS_SUBDIR, COMMON_SUBDIR)`) - or just the tuple of literal names 'workflows','prompts','assets','outputs','exports','common' - into security.py (or a leaf `dw/workspace_names.py`), keep re-export in workspace. One constant, 1 edge.
3. `dw.vram_estimate <-> dw.vram_inheritance`:
   - vram_inheritance -> vram_estimate [`KEY`, `vram_estimate_errors`] vram_inheritance.py:31-32 top.
   - vram_estimate -> vram_inheritance [`pipeline_identity`] vram_estimate.py:155, 200 lazy.
   Fix: move `pipeline_identity` (vram_inheritance.py:45-~75 plus its `_resolved` helper) into vram_estimate.py (or a leaf `dw/pipeline_identity.py`), import it from vram_inheritance.
4. Bonus cycle not in the brief: `dw.server.app <-> dw.server.mcp_mount`: app -> mcp_mount [`build_mcp_app`] app.py:782 lazy; mcp_mount -> app [`LOOPBACK_HOSTS`, `WILDCARD_HOSTS`] mcp_mount.py:25 lazy. Fix: move the two host-set constants to a leaf (e.g. dw/server/hosts.py or dw/netinfo) . scripts/arch_metrics.py counts dw + dw_mcp (server cycle included if grimp sees it - confirm against baseline.json).

---------------------------------------------------------------------------
## 6. Test coupling (tests/ has 200 .py files; 171 import `dw.*`; counts below are string targets in `patch("dw....")`/`setattr("dw....")` plus module-object patches)

| Module | String patch targets (count, files) | Test files importing from module | Most-imported names |
|---|---|---|---|
| dw.result | 41 in 4 files: `dw.result.export_to_video` 13, `dw.result.encode_video` 13, `dw.result.is_av_available` 13, `dw.result.warn_if_written_above_full_scale` 2 | 31 | AudioVideo 37, Result 17, AudioTrack 10, frames_for_encoding 4, warn_if_written_near_silent 4, get_artifact_list 3, warn_if_written_above_full_scale 3, normalize_audio 2, read_embedded_metadata 2; 3 files use `import dw.result as result_module` (no attribute patches found via that alias) |
| dw.pipeline_processors.pipeline | 7 in 2 files: `.load_component` 4, `.empty_device_cache` 2, `.load_loras` 1; plus `patch.object(Pipeline, "load")` 42 and `patch.object(Pipeline,"__init__")` 2 (class attribute, survives a module move as long as `Pipeline` stays importable) | 22 | Pipeline 16, configure_components 10, load_component 5, stateful_cache_context 4, attach_audio_sample_rate 4, place_component 3, _diagnose_image_crf_error 3, load_loras 2 |
| dw.workflow | 19 in 5 files: `dw.workflow.empty_device_cache` 15, `dw.workflow.release_host_caches` 4; plus `patch.object(dw.workflow,"get_device_type")` 7, `patch.object(dw.workflow,"device_capacity_gb")` 7, `patch.object(workflow_module,"realize_args")` 1; `patch.object(Step,"run")` 19 | 54 | Workflow 70, workflow_from_file 24, workflow_from_definition 19, workflow_from_snapshot 4, release_unreferenced_results 2, workflow_output_subfolder 2 |
| dw.arguments | 8 in 1 file: `dw.arguments._fetch_remote_video` 4, `dw.arguments.safe_get` 2, `dw.arguments.load_video` 1, `dw.arguments.load_type_from_name` 1; plus `monkeypatch.setattr(arguments_module,"load_video")` 2 | 17 | fetch_image 10, fetch_video 10, realize_args 9, fetch_constant 9, NON_TYPE_KEYS 2, is_constant_reference 2, is_escaped 2, realize_constants 2 (also `_with_frame_rate`, `realize_object`, `OMITTED`) |
| dw.introspection | 0 | 13 | describe_task 9, list_tasks 4, workflow_argument_warnings 3, describe_class 3, unknown_task_arguments 2, task_signature_errors 2, `_type_reference_error` 2 (private) |
| dw.security | 0 | 27 | SecurityError 19, InvalidInputError 18, TRUST_WORKFLOWS_ENV_VAR 8, MAX_VARIABLE_VALUE_LENGTH 5, PathTraversalError 4, require_trusted_from_pretrained_arguments 4, contained 4, validate_subfolder 3 |
Other cycle modules: dw.locations 19 patch targets (all `dw.locations.socket.getaddrinfo`, 2 files), dw.tasks.video_utils 1 (`load_audio_video`), dw.media_frames 1 (`dw.media_frames.av.open`), dw.runs/shots/content_types/for_each/variables/vram_*/workspace/tasks.task/audio_utils 0. Importers: tasks.video_utils 4 files (frame_grid 13, load_audio_video 7, frames_as_array 6, VideoFileReference 5, `_fit_audio_to_frames` 1 - a private name), tasks.audio_utils 13 (slice_audio 26, normalize_audio 18), runs 19 (new_run_id 8, MANIFEST_FILE_NAME 7, activate/deactivate_output_root 5), shots 7 (shot_record 6), content_types 2, locations 3, task 23, workspace 12.
Implications:
- The patched-by-string names are all *module-global lookups inside the function under test*: patching `dw.result.export_to_video/encode_video/is_av_available` only works while the code that calls them still lives in `dw.result` (they are referenced at call time from `save_artifact`/`save_audio_video`). If `save_audio_video` and the video branch of `save_artifact` move to another module, those 39 patches (4 files) must retarget, or the writer must stay in result.py and receive the encoder functions via result's own namespace. The same goes for `dw.workflow.empty_device_cache` (15) and `dw.workflow.release_host_caches` (4): these are called from `Workflow.run`/`create_step_action`/`_finish_release`; if the release code moves to a `borrow.py`, the patch target must follow (or the moved code can keep calling them through `dw.workflow`'s namespace - avoid). `dw.workflow.get_device_type`/`device_capacity_gb` (7 each, `patch.object(dw.workflow, ...)`) belong to the validation/vram path (workflow.py `validation_context`/vram check) - moving validation into `dw/validation.py` retargets them.
- `warn_if_written_above_full_scale` is patched on `dw.result` (2) and imported from result (3 files); `warn_if_written_near_silent` imported from result in 4 files - re-export from result after extracting the audio-QC block (patching the re-export would NOT affect the moved caller; retarget those 2 patches to the new module).
- `dw.pipeline_processors.pipeline.load_component` (4), `.empty_device_cache` (2), `.load_loras` (1) are the only patched pipeline names; they are called by `Pipeline.load`/`load_optional_component`, so keep `Pipeline` and these three call sites together or retarget 7 patches.
- `dw.arguments._fetch_remote_video`/`safe_get`/`load_video`: all within fetch_image/fetch_video, i.e. the media half (lines 795-1235). Moving it to `media_arguments.py` retargets 8 patches (+2 module-object `setattr`s) in one file.
- `patch.object(Pipeline,"load")` (42) and `patch.object(Step,"run")` (19) are class-attribute patches: independent of module location, only require the class name stays importable at its old path (re-export).
- security/introspection have no string patches, so splitting them only needs re-exports (27 + 13 importer files, 271 `from dw.(arguments|security|introspection|workflow|pipeline_processors.pipeline) import` statements in total).
- Tests that assert on import structure/metrics: scripts/arch_metrics.py + docs/stabilization/baseline.json, tests touching `tests/test_plugin_skills.py` are unrelated; `docs/stabilization/hot-zone.txt` lists files under freeze (released per last commit, but check before touching).

---------------------------------------------------------------------------
## 7. Existing sibling modules that can absorb slices (instead of new files)

- `dw/references.py` (97 lines): reference prefix constants, `is_ref`, `ref_name`, `make_ref`, `author_index`, `render_path`, `MEMBER_SEPARATOR`. The natural home for `FROM_FILE_KEY`/`FROM_PREVIOUS_RESULT_KEY`/`FROM_ARGUMENTS_KEY`/`PREVIOUS_RESULT_PREFIX`/`CONSTANT_PREFIX` (kills for_each/variables/shots -> arguments edges and content_types -> for_each).
- `dw/content_types.py` (147 lines; `content_type_fault`, `content_type_errors`, `refuse_active_content_type`): a fitting home for `AUDIO_FORMATS`, `LOSSY_AUDIO_CONTENT_TYPES`, `MUXED_VIDEO_CONTENT_TYPE` and `guess_extension` (result.py 1808-1830); its docstring already says `AUDIO_FORMATS` in result.py is the source of truth for what it validates.
- `dw/media_info.py` (398; `probe_media` 19-202, `probe_metadata`, envelope helpers; PyAV-only, no torch): natural home for `_probe_written_media` (178-188) and the probe half of the QC warnings (the dBFS measuring: `_peak_dbfs`). Note result.py's warnings use `emit_warning`, media_info does not - keep warnings out of media_info, only move measurement.
- `dw/loudness.py` (82; `integrated_lufs`, `true_peak_dbfs`): measurement primitives; `_peak_dbfs` and sample-peak could sit here.
- `dw/media_audio.py` (266; `extract_audio`, `decode_soundtrack`, `audio_shape`): PyAV-only soundtrack extraction (server side); candidate for the audio-fit helper `_sample_axis`/`_fit_audio_to_frames` only if torch-free (they are numpy/torch on in-memory tracks, so probably not).
- `dw/media_frames.py` (367; `frames_at`, `contact_sheet`, `seam_tiles`, `_read_frames`): absorbs the 5 grid helpers from tasks/video_utils.py 281-335 (breaks media_frames <-> video_utils).
- `dw/shots.py` (410; `shot_record`, `measured_num_samples` 214, `shots_for_file`): result.py's shot re-measure block (~1038-1096) belongs as a function here (`remeasure_after_mux`), it already holds `measured_num_samples`. Needs `PREVIOUS_RESULT_PREFIX` via references instead of arguments.
- `dw/runs.py` (894), `dw/realize.py`, `dw/subfolders.py`: `_write_run_manifest` (workflow.py 1771-1837) joins runs.py; runs.py is itself large (894) so prefer a new `dw/run_manifest.py` if it would push runs.py over.
- `dw/validation.py` (879; `ERROR_CHECKS`/`WARNING_CHECKS` registries): receives workflow.py's validation block (740-964, ~225 lines) - it would grow to ~1,100, over the limit, so either split validation.py first or land it as `dw/workflow_validation.py`. Also `adapter_compatibility.py`, `elision.py`, `slice_preflight.py`, `shot_span_preflight.py`, `video_extensions.py`, `video_size_errors.py`, `dissolve_frame_errors.py`, `scalar_result_validation.py`, `variable_constraints.py`, `task_domains.py` are the existing per-check modules that `validation.py` calls; introspection's inert-argument warnings (1056-1242) and type-reference checks (779-1053) follow that precedent as new per-check modules.
- `dw/step.py`, `dw/step_cache.py` (621), `dw/previous_results.py` (408): `Step` is where the `missing_task_arguments` check could live (breaks introspection <-> task).
- `dw/locations.py` (675), `dw/assets.py`, `dw/prompts.py`: media location policy already split out; `arguments.fetch_*` depends on locations.
- `dw/events.py`, `dw/settings.py`, `dw/host_memory.py`, `dw/hub_cache.py`, `dw/download_watch.py`, `dw/teacache.py`, `dw/cache_blocks.py`: potential homes for pipeline.py's cache/transformer code (`get_cache_transformer`, `stateful_cache_context`, `enable_cache_on_transformer` -> teacache.py/cache_blocks.py), `_hub_auth_status`/`_diagnose_image_crf_error` (-> hub_cache.py / a diagnostics module).
- `dw/pipeline_processors/` has `chain.py` (run_chain, 140-line function), `config_objects.py` (quantization configs), `remote.py` - `apply_sdnq_optimizations` (2043-2090) fits config_objects.py (it already holds SDNQ conventions per CLAUDE.md).
- `dw/tasks/` already has `video_utils.py` (637) and `audio_utils.py` (2229 - itself an oversize file not in scope); `dw/tasks/tensor_image.py`, `image_utils.py`.
- `dw/workspace.py` (730) hosts RESERVED_WORKSPACE_NAMES/SUBDIRS; `dw/vram_estimate.py` (382) should take `pipeline_identity`.
