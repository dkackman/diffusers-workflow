"""Tool class for running, watching and diagnosing jobs."""

from typing import Literal

from dw_mcp import diagnose, exports


class JobTools:
    """Running, watching and diagnosing jobs."""

    def __init__(self, client):
        self.client = client

    def run_workflow(
        self,
        workflow_path: str | None = None,
        inline_workflow: dict | str | None = None,
        workflow: dict | str | None = None,
        name: str | None = None,
        arguments: dict | None = None,
        acknowledged_cost: bool | dict = False,
        workspace: str | None = None,
        wait_seconds: int = 0,
    ) -> dict:
        """Queue a workflow for generation. This costs GPU time: a run
        occupies a GPU for minutes and the engine runs one job per GPU.
        Tell the user what will run and get their go-ahead, then pass
        acknowledged_cost as below. Returns as soon as the job is queued;
        follow it with `wait_for_job`, then `get_job` for the manifest - or
        fold that first wait in with `wait_seconds` above 0, which waits on
        the job exactly as `wait_for_job(job_id,
        timeout_seconds=wait_seconds)` would ({cap}s cap per call) and adds
        its fields to the result (`still_running`, `waited_seconds`,
        `timeout_*`, the slim `job`). If the cap covers the job's
        runtime one call is enough; on `still_running: true` call
        `wait_for_job` again. Give exactly one of `workflow_path` - a
        catalog name from `list_workflows`, with or without .json, or a
        path on the server - or `inline_workflow`, a full definition
        nothing stored covers; `validate_workflow` calls these `name` and
        `workflow`, and both tools accept both spellings.
        `inline_workflow`/`workflow` may also be a JSON-encoded string; a
        parse failure is reported as invalid JSON, not a type mismatch.
        `arguments` overrides the workflow's variables by name. `workspace` pins this
        call to another workspace without switching the session (where its
        `output:`/`asset:` references live).

        Bind the acknowledgement to what you quoted: pass
        {"fingerprint": plan.fingerprint, "minutes": plan.estimate.minutes,
        "downloads": [...the non-null repos in plan.downloads_required]} from
        the validate plan; the server refuses with 409, naming the new
        plan, if the run's shape changed since. Bare true is for a plan
        that was null."""
        return diagnose.run_workflow(
            self.client,
            workflow_path=workflow_path,
            inline_workflow=inline_workflow,
            workflow=workflow,
            name=name,
            arguments=arguments,
            acknowledged_cost=acknowledged_cost,
            workspace=workspace,
            wait_seconds=wait_seconds,
        )

    # The cap is a number a caller paces against, so the description
    # states it (as wait_for_job's does, below). replace rather than
    # format: the docstring spells out a literal {fingerprint, ...} dict.
    # Done on the class attribute, before binding: a bound method's __doc__
    # is read-only, and wrapping the method would change its signature.
    if run_workflow.__doc__:  # absent under python -OO
        run_workflow.__doc__ = run_workflow.__doc__.replace(
            "{cap}", str(diagnose.MAX_WAIT_SECONDS)
        )

    def get_job(self, job_id: str) -> dict:
        """Get a job's status, argument warnings, output manifest, error and
        traceback. The manifest names each step's files the way
        `get_output_image`, `download_output` and `keep_output` take them,
        and each entry's `subfolder` says what kind of output the step
        declared - by convention `final` is the deliverable, `intermediate`
        the scratch work, and '' a step that said nothing. A step served
        from the step cache is marked `reused` and reports the earlier run's
        files. When a job failed, the error and traceback here are what to
        read before changing anything. `acknowledged` says which form of
        cost acknowledgement queued the job (`none`, `boolean`, `bound`) and
        `acknowledged_cost` is the bound `{fingerprint, minutes, downloads}`
        when there was one."""
        return diagnose.get_job(self.client, job_id)

    def get_job_workflow(self, job_id: str) -> dict:
        """Get the workflow a job actually ran. When `realized` is true every
        mutable input is pinned - the caller's arguments folded into the
        variables, the seed the run used, stored prompt text inlined, and any
        `output:.../latest/...` rewritten to the run it resolved to - so the
        definition reproduces that run however the library changes. When it is
        false the run's copy is gone (old job, or its workspace was deleted)
        and `note` says what was folded in from the recorded arguments. After a long inline run worth keeping, this then
        `save_workflow` is how it gets a name."""
        return diagnose.get_job_workflow(self.client, job_id)

    def get_job_events(
        self,
        job_id: str,
        after: int = -1,
        limit: int = 200,
        kinds: list[str] | None = None,
    ) -> dict:
        """Get a page of a job's progress events - phase transitions, denoise
        steps, memory readings and log lines. `after` is exclusive: pass back
        the previous call's `last_seq` to continue. Each event's `at` is
        seconds since the job started, so where a step's time went is the
        difference between two events. For 'is it still moving?' the
        `progress` block on get_job/wait_for_job is cheaper than a page of
        events. `kinds` (e.g. `["log", "warning"]`, or a warning's `kind`)
        filters the page - `memory` events otherwise dominate it.

        A `kind: "phase_stall"` entry is a watchdog notice, not progress - it
        fires every ~30s a phase goes quiet, not evidence of a hang by
        itself. It carries `seconds_since_last_progress` and
        `seconds_since_phase_start`; some models are silent for minutes
        normally - check the model's guide before treating one as a fault."""
        return diagnose.get_job_events(
            self.client, job_id, after=after, limit=limit, kinds=kinds
        )

    def wait_for_job(self, job_id: str, timeout_seconds: int = 20) -> dict:
        """Block until a job finishes, instead of polling get_job or
        get_job_events by hand: returns as soon as its status is succeeded,
        failed or cancelled, or with still_running: true when
        timeout_seconds elapses first, so you can call again. Queues
        nothing, so no acknowledged_cost.

        One call blocks for at most {cap} seconds, whatever timeout_seconds
        asks for - this deployment's cap, set for the tool-call budget the
        client holds open; a larger value is clamped, not honoured, so
        budget one call per {cap}s of the job, and one call is enough when
        {cap} covers its runtime. Every reply says which happened:
        waited_seconds, timeout_requested_seconds, timeout_applied_seconds
        and timeout_capped.

        Returns a slim job - status, warnings, error, the manifest once
        finished - without the arguments (get_job has those). A running job
        also carries `progress`: step, phase, and
        `denoise_step`/`denoise_total_steps`, null until the denoise loop
        starts. Tell a slow run from a stuck one by whether
        `denoise_step` has moved since a poll minutes ago, not by silence:
        a video reference's lead-in can run many minutes emitting nothing,
        and denoise gaps are uneven under a transformer block cache - both
        normal. If you're also reading get_job_events, a `phase_stall`
        entry there is the same silence being narrated, not a fault or a
        sign of progress - it repeats every ~30s the phase stays quiet, so
        neither seeing one nor watching its event_count climb tells you
        anything `denoise_step` doesn't already say better. Full diagnosis,
        and why `denoise_total_steps` can read one less than asked, in
        WORKFLOW_GUIDE's "The loop", step 5."""
        return diagnose.wait_for_job(
            self.client, job_id, timeout_seconds=timeout_seconds
        )

    # The cap is a number a caller paces against, so the description states
    # it rather than saying "well under a generation's runtime". On the
    # class attribute, before binding (see run_workflow).
    if wait_for_job.__doc__:  # absent under python -OO
        wait_for_job.__doc__ = wait_for_job.__doc__.format(
            cap=diagnose.MAX_WAIT_SECONDS
        )

    def cancel_job(self, job_id: str) -> dict:
        """Ask a queued or running job to stop. Cooperative: a running job
        stops at the next step or denoise-step boundary, not instantly.
        Deliberately not gated - it ends a cost rather than starting one."""
        return diagnose.cancel_job(self.client, job_id)

    def rerun_job(
        self,
        job_id: str,
        acknowledged_cost: bool | dict = False,
        new_seed: bool = False,
    ) -> dict:
        """Queue a fresh job from a previous job's stored specification. This
        costs GPU time: a rerun is a run - it occupies a GPU for
        minutes and the engine runs one job per GPU. Tell the user what
        will run and get their go-ahead, then pass acknowledged_cost.

        Pass new_seed=true for a different image: a workflow that pins its
        seed reruns to the same pixels, and the step cache serves that whole
        run from the earlier one's files (marked `reused`) in a fraction of a
        second rather than generating anything.

        `acknowledged_cost` takes the same bound form as run_workflow; a
        fresh seed never changes the fingerprint, so the original plan still
        binds a new_seed rerun."""
        return diagnose.rerun_job(
            self.client,
            job_id,
            acknowledged_cost=acknowledged_cost,
            new_seed=new_seed,
        )

    def move_job(
        self, job_id: str, direction: Literal["up", "down", "front", "back"]
    ) -> dict:
        """Reorder a queued job. Only a job still waiting can move; the one
        already running cannot."""
        return diagnose.move_job(self.client, job_id, direction)

    def export_job(self, job_id: str, overwrite: bool = False) -> dict:
        """Gather one finished job into a directory on the server: the
        realized workflow, the run's manifest, the job row, a README, and
        copies of every asset it used, every earlier run's file it read and
        every file it made. The export copies every output and input file
        rather than linking them, so a video job's export costs its size
        again on the server's disk; `total_bytes` in the result reports
        what was copied. Returns the directory, a zip URL, the file list
        with sizes and the total. The three JSON files are in the zip, not
        repeated here - get_job_workflow and get_job serve them individually.
        THE DIRECTORY IS ON THE MACHINE RUNNING THE SERVER, not on yours.

        open_url is the zip and needs no token: fetch it with any HTTP
        you have (prefix a relative one with the server's address) and
        unpack it into exports/ under the session's working directory -
        the user's deliverable, not a temp file; it already unpacks into
        one folder named after the job id, so do not create that folder first.
        Only without HTTP, give the user open_url to open. If `auth_required`
        is ever true, the zip is gated: hand it over (see `next`).
        Individual results stay reachable inline via
        get_output_image/get_output_audio/get_output_frames
        without the zip. Refuses a job that is still running; refuses an existing
        export unless overwrite=true."""
        return exports.export_job(self.client, job_id, overwrite=overwrite)
