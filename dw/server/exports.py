"""Gathering one finished job into a directory that stands on its own.

The record of a run is split across four places on one machine: the job row,
the run's manifest, the workflow that ran, and the media on either side of it.
This module puts them in one tree - text at the top, media in folders, no
absolute paths inside the files - so the run can be committed, moved or handed
to someone else.

    exports/<job id>/
        README.md       what this is, how it was made, how to run it again
        workflow.json   the realized workflow, when the run wrote one
        manifest.json   the run's manifest, or a synthesized stand-in
        job.json        the job row: status, times, arguments, warnings, error
        assets/         every asset: the workflow names, under its own name
        inputs/         every file an output: reference named, by run
        outputs/        every file the manifest lists

Copies are copies, never hard links: exports/ is what a user moves or deletes,
and a hard link would make deleting it look like deleting the original.
"""

import json
import logging
import os
import shutil
from dataclasses import dataclass, field

from ..assets import ASSET_PREFIX
from ..realize import strings_with_prefix
from ..runs import (
    MANIFEST_FILE_NAME,
    OUTPUT_PREFIX,
    is_output_reference,
    resolve_output_reference,
)
from ..security import (
    SecurityError,
    validate_asset_reference,
    validate_output_path,
    validate_path,
)
from ..workspace import EXPORTS_SUBDIR
from .jobs import TERMINAL_STATES

logger = logging.getLogger("dw")

WORKFLOW_FILE_NAME = "workflow.json"
JOB_FILE_NAME = "job.json"
README_FILE_NAME = "README.md"

# What job.json holds, in this order, whatever the job was read from. A live
# job's detail and a historical one's differ in shape - history carries the
# submitted spec and a 'historical' flag, a live job a traceback and an event
# count - and an export is a record, so it takes one fixed key set: a
# traceback is a developer's artifact of one process, an event count
# describes a log this tree does not hold, and the spec is what
# workflow.json already is. 'realized' is added beside them
JOB_RECORD_KEYS = (
    "id",
    "workflow",
    "workflow_name",
    "status",
    "created_at",
    "started_at",
    "finished_at",
    "workspace",
    "arguments",
    "warnings",
    "manifest",
    "error",
    "run_id",
    "run_dir",
)

README_TEMPLATE = """# {workflow_name} - job {job_id}

{realized_sentence}

## The run

| | |
| --- | --- |
| Job | `{job_id}` |
| Workflow | `{workflow_name}` |
| Catalog entry | {catalog_name} |
| Status | {status} |
| Started | {started_at} |
| Finished | {finished_at} |
| Device | {device} |
| Engine | dw {dw_version} |
| Seed | `{seed}` |

Arguments as submitted:

```json
{arguments}
```

## Stored prompts

{prompts}

## Sub-workflows

{sub_workflows}

## Inputs

{inputs}

## Running it again

From a checkout of diffusers-workflow, with this directory as the working
directory:

```bash
python -m dw.run workflow.json
```

Over MCP, pass the contents of `workflow.json` to `run_workflow` as
`inline_workflow`. Either way the `asset:` and `output:` names in it resolve
against the server's libraries, not against this directory - the copies under
`assets/` and `inputs/` are the record of what those names meant, not a
substitute for them.

## Missing

{missing}

## A note on committing this

`assets/`, `inputs/` and `outputs/` hold generated and source media, which is
usually large and always binary. If this tree goes into Git, put those three
directories in Git LFS; the four files at the top are text and belong in Git
proper.
"""


@dataclass
class ExportSummary:
    """What an export produced, computed from what actually landed."""

    job_id: str
    directory: str
    files: list = field(default_factory=list)
    total_bytes: int = 0
    missing: list = field(default_factory=list)

    def as_dict(self):
        return {
            "job_id": self.job_id,
            "directory": self.directory,
            "files": self.files,
            "total_bytes": self.total_bytes,
            "missing": self.missing,
        }


def export_directory(workspace_root, job_id):
    """Where one job's export lives, confined to the workspace root.

    The job id is caller input - it arrives in a URL - so it is joined and
    then validated rather than trusted to be the hex string the manager
    generates.
    """
    if not workspace_root:
        raise ValueError("This server has no workspace root to export into")
    root = validate_output_path(os.path.join(workspace_root, EXPORTS_SUBDIR), None)
    return validate_path(os.path.join(root, job_id), root)


def export_job(manager, job_id, workspace_root, asset_roots, overwrite=False):
    """Gather one finished job into '<workspace_root>/exports/<job id>/'.

    Args:
        manager: The JobManager holding the job (live or historical).
        job_id: The job to export.
        workspace_root: The workspace the export lands in.
        asset_roots: The workspace's asset search path, in order - the same
            order 'asset:' resolves in, so what is copied is what the run
            loaded.
        overwrite: Replace an existing export rather than refusing.

    Returns:
        An ExportSummary.

    Raises:
        ValueError: Unknown job, or a job that has not finished.
        FileExistsError: The export exists and overwrite is False.
    """
    job = manager.get(job_id)
    if job is None:
        raise ValueError(f"Unknown job {job_id}")
    detail = job if isinstance(job, dict) else manager.describe(job)
    status = detail.get("status")
    if status not in TERMINAL_STATES:
        raise ValueError(
            f"Job {job_id} is {status} - only a finished job can be exported"
        )

    spec = job.get("spec", {}) if isinstance(job, dict) else job.spec
    run_dir = job.get("run_dir") if isinstance(job, dict) else job.run_dir
    output_root = validate_output_path(
        spec.get("output_dir") or manager.output_dir, None
    )

    target = export_directory(workspace_root, job_id)
    if os.path.exists(target):
        if not overwrite:
            raise FileExistsError(
                f"An export of job {job_id} already exists - pass overwrite to "
                f"replace it"
            )
        shutil.rmtree(target)
    os.makedirs(target)

    summary = ExportSummary(job_id=job_id, directory=target)

    realized = manager.realized(job_id)
    workflow = realized if realized is not None else (manager.definition(job_id) or {})
    _write_json(summary, target, WORKFLOW_FILE_NAME, workflow)

    manifest = _run_manifest(output_root, run_dir)
    if manifest is None:
        # No run directory, or it no longer holds a manifest: the job row's
        # own per-step file list is what is left, and it says so
        manifest = {
            "synthesized": True,
            "run_id": None,
            "status": status,
            "steps": detail.get("manifest") or [],
        }
    _write_json(summary, target, MANIFEST_FILE_NAME, manifest)

    record = {key: detail.get(key) for key in JOB_RECORD_KEYS}
    record["realized"] = realized is not None
    _write_json(summary, target, JOB_FILE_NAME, record)

    _copy_assets(summary, workflow, target, asset_roots)
    _copy_inputs(summary, workflow, target, output_root)
    _copy_outputs(summary, manifest, target, output_root, run_dir)

    readme = _readme(job_id, detail, manifest, workflow, summary, realized is not None)
    _write_text(summary, target, README_FILE_NAME, readme)
    return summary


# -------------------------------------------------------------- the pieces


def _run_manifest(output_root, run_dir):
    """The manifest the run left, or None when there is no reading it."""
    if not run_dir:
        return None
    try:
        path = validate_path(
            os.path.join(output_root, run_dir, MANIFEST_FILE_NAME), output_root
        )
        with open(path, "r") as file:
            return json.load(file)
    except (SecurityError, OSError, ValueError) as e:
        logger.debug(f"No run manifest to export from {run_dir}: {e}")
        return None


def _copy_assets(summary, workflow, target, asset_roots):
    """Every 'asset:' the workflow names, under its own name in assets/."""
    for reference in strings_with_prefix(workflow, ASSET_PREFIX):
        try:
            name = validate_asset_reference(
                reference.removeprefix(ASSET_PREFIX).strip()
            )
        except SecurityError:
            summary.missing.append(reference)
            continue
        source = None
        for root in asset_roots:
            try:
                candidate = validate_path(os.path.join(root, name), root)
            except SecurityError:
                continue
            if os.path.isfile(candidate):
                source = candidate
                break
        if source is None:
            summary.missing.append(reference)
            continue
        _copy(
            summary, source, target, os.path.join("assets", *name.split("/")), reference
        )


def _copy_inputs(summary, workflow, target, output_root):
    """Every 'output:' the workflow names, kept under the run it came from.

    The reference itself is not rewritten - the realized workflow is the
    immutable record of the run - so the directory name is the reference's
    own name, and the README says where each one came from.
    """
    for reference in strings_with_prefix(workflow, OUTPUT_PREFIX):
        if not is_output_reference(reference):
            continue
        name = reference.removeprefix(OUTPUT_PREFIX).strip()
        try:
            source = resolve_output_reference(reference, output_root)
        except (SecurityError, OSError, ValueError):
            summary.missing.append(reference)
            continue
        _copy(
            summary, source, target, os.path.join("inputs", *name.split("/")), reference
        )


def _copy_outputs(summary, manifest, target, output_root, run_dir):
    """Every file the manifest lists, under outputs/.

    A manifest entry names a file relative to the run directory when the run
    wrote it, and absolutely when a step-cache hit republished an earlier
    run's file. Both land here: the first under its own relative name, the
    second under its path relative to the output root, which keeps the
    identity and run id that say where it really came from.
    """
    run_root = os.path.join(output_root, run_dir) if run_dir else output_root
    # One file can be listed by two steps (a chain's output is the next
    # step's input); it is one file in the export, copied and counted once
    copied = set()
    for entry in manifest.get("steps") or []:
        if not isinstance(entry, dict):
            continue
        for recorded in entry.get("files") or []:
            source = (
                recorded
                if os.path.isabs(recorded)
                else os.path.join(run_root, recorded)
            )
            try:
                source = validate_path(source, output_root)
            except SecurityError:
                summary.missing.append(recorded)
                continue
            if not os.path.isfile(source):
                summary.missing.append(recorded)
                continue
            if source in copied:
                continue
            copied.add(source)
            try:
                relative = os.path.relpath(source, run_root)
            except ValueError:  # different drive on Windows
                relative = os.path.basename(source)
            if relative == os.pardir or relative.startswith(os.pardir + os.sep):
                relative = os.path.relpath(source, output_root)
            _copy(
                summary,
                source,
                target,
                os.path.join("outputs", *relative.split(os.sep)),
                recorded,
            )


def _readme(job_id, detail, manifest, workflow, summary, realized):
    workflow_block = manifest.get("workflow") or {}
    prompts = workflow_block.get("prompts") or []
    sub_workflows = workflow_block.get("sub_workflows") or {}
    inputs = [
        entry["path"] for entry in summary.files if entry["path"].startswith("inputs/")
    ]
    return README_TEMPLATE.format(
        job_id=job_id,
        workflow_name=detail.get("workflow") or "unknown",
        catalog_name=(
            f"`{detail['workflow_name']}`"
            if detail.get("workflow_name")
            else "none - an inline definition"
        ),
        realized_sentence=(
            "`workflow.json` is the *realized* workflow: every mutable input "
            "is pinned, so it reproduces this run whatever changes afterwards."
            if realized
            else "`workflow.json` is the definition as submitted - this job "
            "predates run tracking, so its arguments and prompts are not "
            "pinned into it."
        ),
        status=detail.get("status"),
        started_at=detail.get("started_at"),
        finished_at=detail.get("finished_at"),
        device=manifest.get("device", "unknown"),
        dw_version=manifest.get("dw_version", "unknown"),
        seed=manifest.get("seed", workflow.get("seed", "not recorded")),
        arguments=json.dumps(detail.get("arguments") or {}, indent=2),
        prompts=_bullets(f"`{name}` - inlined into `workflow.json`" for name in prompts)
        or "None: this workflow named no stored prompt.",
        sub_workflows=_bullets(
            f"`{path}` - sha256 `{digest}`" if digest else f"`{path}` - unreadable"
            for path, digest in sorted(sub_workflows.items())
        )
        or "None: this workflow composed no other workflow by path.",
        inputs=_bullets(
            f"`{path}` - copied from the run named in its own path" for path in inputs
        )
        or "None: this workflow named no file from an earlier run.",
        missing=_bullets(f"`{name}`" for name in summary.missing)
        or "Nothing: every file this run referenced was found and copied.",
    )


def _bullets(lines):
    return "\n".join(f"- {line}" for line in lines)


# ------------------------------------------------------------------- files


def _copy(summary, source, target, relative, name):
    """Copy one file into the export and record it.

    `name` is how a failure is reported - the reference or the manifest's own
    entry, never the absolute path, because `missing` is read out of the
    README on another machine where that path means nothing.
    """
    destination = os.path.join(target, relative)
    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copyfile(source, destination)
    except OSError as e:
        logger.warning(f"Could not copy {source} into the export: {e}")
        summary.missing.append(name)
        return
    _record(summary, destination, target)


def _write_json(summary, target, name, payload):
    _write_text(summary, target, name, json.dumps(payload, indent=2, default=str))


def _write_text(summary, target, name, text):
    path = os.path.join(target, name)
    try:
        with open(path, "w", encoding="utf-8") as file:
            file.write(text)
    except OSError as e:
        logger.warning(f"Could not write {path}: {e}")
        return
    _record(summary, path, target)


def _record(summary, path, target):
    try:
        size = os.path.getsize(path)
    except OSError:
        return
    summary.files.append(
        {"path": os.path.relpath(path, target).replace(os.sep, "/"), "bytes": size}
    )
    summary.total_bytes += size
