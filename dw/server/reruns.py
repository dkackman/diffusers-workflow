"""Reruns: a fresh job from an earlier job's spec.

`Reruns` is a mixin of `JobManager` (`dw/server/jobs.py`), as `Dispatcher`
(`dw/server/dispatch.py`) is - its methods read the manager's live jobs and
history (`jobs`, `history`, `definition`, `realized`) and queue through its
`submit`. One job is rerun by `rerun_spec` then `rerun`, split so a route
admits the run in between (#85); the seed rule is `job_record.rerun_arguments`.
A dead server's whole batch of restart tombstones is rerun by
`rerun_restart_batch` (#1266), one member at a time through the same two;
`restart_batch_size` counts it for the Jobs banner (#1633).
"""

import logging
import os

from .. import ledger, references
from ..workspace import DEFAULT_WORKSPACE_NAME
from .job_history import DELETED_SUFFIX
from .job_record import ACK_BOOLEAN, ACK_NONE, RERUN_SPEC_KEYS, rerun_arguments

logger = logging.getLogger("dw")


class Reruns:
    """JobManager's rerun of one job, and of a restart batch."""

    def seed_variable(self, job_id):
        """The variable this job's workflow draws its seed from, or None.

        Read from the workflow as written, never from the realized copy the
        run wrote: realization pins the top-level seed to the integer the run
        used, so a realized workflow always looks like it names a literal.

        None means a new-seed rerun has nowhere to put one - either the seed
        is a literal (an argument cannot override it) or the workflow names
        no seed at all, in which case every run already draws a fresh one and
        the step cache is off.
        """
        definition = self.definition(job_id)
        seed = (definition or {}).get("seed")
        name = references.ref_name(references.VARIABLE, seed)
        if name is None:
            return None
        return name if name in (definition.get("variables") or {}) else None

    def rerun_spec(self, job_id, new_seed=False, overrides=None):
        """The spec and arguments a rerun of `job_id` would submit, as
        (spec, arguments), or None for an unknown job - split from rerun()
        so a route can admit the run before queuing it (#85).

        `new_seed` draws a fresh seed into the workflow's seed variable
        (see rerun). `overrides` are variables merged over the stored
        arguments first, keeping a seed the original drew (rerun_arguments).
        Raises ValueError when it cannot, or the job's workspace is gone."""
        job = self.jobs.get(job_id)
        if job is not None:
            spec = {key: job.spec[key] for key in RERUN_SPEC_KEYS if key in job.spec}
            arguments = job.spec.get("arguments", {})
        else:
            historical = self.history.get(job_id)
            if historical is None:
                return None
            spec = {
                key: historical["spec"][key]
                for key in RERUN_SPEC_KEYS
                if key in historical["spec"]
            }
            arguments = historical["arguments"]

        arguments, spec["rerun_warnings"] = rerun_arguments(
            arguments,
            overrides,
            new_seed,
            self.seed_variable(job_id),
            lambda: self.realized(job_id),
        )

        workspace = spec.get("workspace")
        if (
            workspace
            and workspace != DEFAULT_WORKSPACE_NAME
            and spec.get("output_dir")
            and not os.path.isdir(spec["output_dir"])
        ):
            raise ValueError(f"Workspace '{workspace}' the job ran in no longer exists")
        return spec, arguments

    def rerun(
        self,
        job_id,
        *,
        admitted,
        new_seed=False,
        acknowledged=ACK_NONE,
        acknowledged_cost=None,
        warnings=None,
        arguments=None,
        vram_need=None,
        host_ram_need=None,
        repos_named=None,
        overrides=None,
    ):
        """Queue a fresh job from a previous job's spec.

        Every root the original ran against (workflow_dir/output_dir/
        asset_dir/workspace) rides along, not just the workflow identity -
        otherwise a rerun of a job from a named workspace would fall back to
        the manager's process-wide default and silently run somewhere else.

        `new_seed` draws a fresh seed into the workflow's seed variable. A
        plain rerun of a seeded workflow repeats its arguments exactly, which
        makes every step a step-cache hit: it republishes the earlier run's
        files in a fraction of a second and generates nothing. That is the
        cache doing its job - the same seed and the same inputs would produce
        the same pixels - so the way to actually get another image is to
        change the seed, and this is that.

        Like submit, rerun does not re-check: callers admit first. The route
        admits the arguments `rerun_spec(job_id, new_seed)` answered and
        passes them back as `arguments`, with the admission's `warnings` and
        its Workflow as `admitted`, so what is queued is what was admitted -
        the seed is not drawn twice.

        `acknowledged` and `acknowledged_cost` are this request's own; the
        original's bound object rides along in the spec for the record when
        the request brought none.

        The rerun prefers the card the original ran on: its step cache is
        there, which is what makes a same-seed rerun finish at once. When
        that card is busy, the rerun takes any free card it fits.
        """
        prepared = self.rerun_spec(
            job_id, new_seed=new_seed and arguments is None, overrides=overrides
        )
        if prepared is None:
            return None
        spec, recorded = prepared
        link_warnings = self._link_warnings(spec)
        return self.submit(
            workflow_path=spec.get("workflow_path"),
            workflow=spec.get("workflow"),
            admitted=admitted,
            arguments=recorded if arguments is None else arguments,
            base_dir=spec.get("base_dir"),
            workflow_dir=spec.get("workflow_dir"),
            output_dir=spec.get("output_dir"),
            asset_dir=spec.get("asset_dir"),
            workspace=spec.get("workspace"),
            catalog_name=spec.get("catalog_name"),
            acknowledged=acknowledged,
            acknowledged_cost=(
                acknowledged_cost
                if acknowledged_cost is not None
                else spec.get("acknowledged_cost")
            ),
            # keep_drawn_seed's, unless a new_seed replaced the drawn seed
            warnings=[
                *(warnings or []),
                *([] if new_seed else spec["rerun_warnings"]),
                *link_warnings,
            ],
            vram_need=vram_need,
            host_ram_need=host_ram_need,
            repos_named=repos_named,
            preferred_device=self._original_device(job_id),
            note=spec.get("note"),
            rerun_of=job_id,
            overrides=overrides,
            # The original's requests-ledger link, kept whatever the ledger
            # says now (a warning, never a refusal)
            request=spec.get("request"),
            item=spec.get("item"),
        )

    def _link_warnings(self, spec):
        """The warning a rerun carries when the ledger link it inherits no
        longer admits a job: the request closed or gone, the item dropped
        or gone. The rerun runs and keeps the link regardless."""
        request, item = spec.get("request"), spec.get("item")
        if not (request and item) or self.ledger_root_for is None:
            return []
        try:
            root = self.ledger_root_for(spec.get("workspace") or DEFAULT_WORKSPACE_NAME)
            problem = ledger.link_problem(root, request, item)
        except Exception:
            return []
        if problem is None:
            return []
        return [f"rerun_ledger_link_{problem.code}: {problem}"]

    def _already_rerun(self, ids):
        """{member id: rerun id} for each of `ids` already rerun, in history
        or by a live job."""
        done = self.history.rerun_ids(ids)
        with self._lock:
            for job in self.jobs.values():
                if job.rerun_of in ids and job.rerun_of not in done:
                    done[job.rerun_of] = job.id
        return done

    def restart_batch_size(self, job_id):
        """How big the restart batch `job_id` belongs to is, whatever page
        a listing holds (#1633): `size`, every tombstone, and `pending`,
        those a rerun of the batch would still consider - not rerun, nor in
        a deleted workspace. Prices and admits nothing. None when `job_id`
        is not a restart tombstone."""
        batch = self.history.restart_batch(job_id)
        if batch is None:
            return None
        instance, members = batch
        done = self._already_rerun([member["id"] for member in members])
        pending = [
            member
            for member in members
            if member["id"] not in done
            and not (member["workspace"] or "").endswith(DELETED_SUFFIX)
        ]
        return {
            "server_instance": instance,
            "size": len(members),
            "pending": len(pending),
        }

    def rerun_restart_batch(self, job_id, admit, *, queue):
        """Rerun every tombstone the dead server that left `job_id` left
        (#1266), oldest first - or None when `job_id` is not a restart
        tombstone (`JobHistory.restart_batch`).

        `admit(member_id)` admits one member as the single rerun route does
        (rerun_spec -> admit_for -> check_fits) and answers (arguments,
        admission), or raises with why it refused. With `queue` false
        nothing is queued: the answer is the preview, each admitted member
        under `would_queue` with its admission. With it, each is queued
        through `rerun()`, so it carries `rerun_of`, its own VRAM and
        host-RAM needs and the original's card.

        A member already rerun (`rerun_of`, in history or live) or in a
        deleted workspace is skipped, and so is one admission refuses; one
        member failing never stops the rest. The batch lock makes the
        already-rerun check and the queuing one step, so two concurrent
        calls queue each member once."""
        with self._batch_lock:
            batch = self.history.restart_batch(job_id)
            if batch is None:
                return None
            instance, members = batch
            done = self._already_rerun([member["id"] for member in members])
            answer = {
                "server_instance": instance,
                "queued": [],
                "would_queue": [],
                "skipped": [],
            }
            for member in members:
                member_id = member["id"]
                workspace = member["workspace"] or DEFAULT_WORKSPACE_NAME

                def skip(reason):
                    answer["skipped"].append({"id": member_id, "reason": reason})

                if member_id in done:
                    skip(f"already rerun as {done[member_id]}")
                    continue
                if workspace.endswith(DELETED_SUFFIX):
                    skip(f"workspace '{workspace[: -len(DELETED_SUFFIX)]}' was deleted")
                    continue
                try:
                    arguments, admission = admit(member_id)
                except Exception as e:
                    skip(str(e) or type(e).__name__)
                    continue
                entry = {
                    "id": member_id,
                    "workflow": member["workflow"],
                    "workspace": workspace,
                }
                if not queue:
                    answer["would_queue"].append(
                        {
                            **entry,
                            "workflow_name": member.get("workflow_name"),
                            "admission": admission,
                        }
                    )
                    continue
                try:
                    job = self.rerun(
                        member_id,
                        admitted=admission.workflow,
                        acknowledged=ACK_BOOLEAN,
                        warnings=admission.warnings,
                        arguments=arguments,
                        vram_need=admission.vram_need,
                        host_ram_need=admission.host_ram_need,
                        repos_named=admission.repos_named,
                    )
                except Exception as e:
                    logger.warning(f"Batch rerun of {member_id} failed: {e}")
                    skip(f"rerun failed after admission: {e}")
                    continue
                if job is None:
                    skip("no longer in history")
                    continue
                done[member_id] = job.id
                answer["queued"].append({**entry, "rerun_of": member_id, "id": job.id})
            return answer

    def _original_device(self, job_id):
        """The ordinal of the card `job_id` ran on ('cuda:1'), or None."""
        job = self.jobs.get(job_id)
        if job is not None:
            return job.device_ordinal
        return self.history.device_ordinal(job_id)
