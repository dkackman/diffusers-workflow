"""Tool classes for workspaces, workflow authoring and the prompt library."""

from typing import Optional

from dw_mcp import authoring, loras, prompts, workspaces


class WorkspaceTools:
    """Workspaces."""

    def __init__(self, client):
        self.client = client

    def list_workspaces(self, detail: bool = False) -> dict:
        """List the server's workspaces and say which one this session is
        working in. Each has its own workflows, assets and outputs; the
        stored prompt library is shared by all of them, and so is the
        shared asset library that `upload_asset(shared=true)` and
        `keep_output(shared=true)` write into - which is how a recurring
        cast stays reachable from the workspace the next piece is made
        in. Entries carry name, default and usage (files/bytes) only; pass
        detail=true for each entry's full folder paths (workflows, assets,
        outputs, prompts, common_assets)."""
        return workspaces.list_workspaces(self.client, detail=detail)

    def use_workspace(self, name: str) -> dict:
        """Work in a different workspace for the rest of this session - every
        later call reads and writes there. Use this to keep your work out of
        another agent's namespace, rather than sharing the default one."""
        return workspaces.use_workspace(self.client, name)

    def create_workspace(self, name: str, use: bool = False) -> dict:
        """Create a workspace on the server. It gets its own workflows,
        assets and outputs and shares the one prompt library. The name is a
        single path segment and cannot be one of the reserved folder names
        (workflows, prompts, assets, outputs, exports, common, loras). Pass use=true to switch this
        session to it as well; otherwise the session stays where it was and
        the result says so."""
        return workspaces.create_workspace(self.client, name, use=use)

    def delete_workspace(self, name: str, acknowledged_cost: bool = False) -> dict:
        """Permanently delete a workspace and every workflow, asset and
        generated file in it. Refuses without acknowledged_cost=True, and
        reports what it would remove instead."""
        return workspaces.delete_workspace(
            self.client, name, acknowledged_cost=acknowledged_cost
        )


class AuthoringTools:
    """Validating, saving and deleting workflows."""

    def __init__(self, client):
        self.client = client

    def validate_workflow(
        self,
        workflow: dict | str | None = None,
        name: str | None = None,
        inline_workflow: dict | str | None = None,
        workflow_path: str | None = None,
        workspace: str | None = None,
        arguments: dict | None = None,
    ) -> dict:
        """Check a workflow against the schema and against real pipeline
        signatures. Free and instant - run it before run_workflow.
        Give exactly one of `workflow` or `name` (a stored workflow as
        `list_workflows` reports it); `run_workflow`'s `inline_workflow`
        and `workflow_path` spellings are accepted here too. `workflow` may
        be a JSON-encoded string. Every error comes back at once, each with
        its JSON path. `workspace` pins this one call to another workspace
        without switching the session.

        A valid answer carries `plan`: what will execute for these
        arguments - `estimate.minutes` and its `basis` (`observed`,
        `per_entry`, `catalog`, `derived`, `inherited`, `other_device` or `unknown` -
        how to quote each is WORKFLOW_GUIDE's "The loop", step 4), each
        `downloads_required` entry as its own cost line, and
        `steps`/`list_entries` for how many members the list produced.
        `plan` is null when it could not be built; the verdict stands.

        Pass the same `arguments` you will pass to `run_workflow` and they
        are checked too: an undeclared name, a value that will not coerce,
        and an `asset:`/`prompt:`/`output:` reference naming nothing this
        workspace can reach, each at `arguments.<name>`.
        `checked_arguments` says whether your values or only the stored
        defaults were checked. A value outside a bound the workflow
        declares is an error here rather than a failed run; one the workflow rounds up
        comes back as a warning naming what it becomes.

        Also checked: an unwritable `result.subfolder` or `file_base_name`,
        after `for_each` expansion; and each sub-workflow step - an
        unreachable `workflow.path`, the composed workflow in turn, a
        composition cycle, and (as a warning) an argument passed down that
        it declares no variable for."""
        return authoring.validate_workflow(
            self.client,
            workflow=workflow,
            name=name,
            inline_workflow=inline_workflow,
            workflow_path=workflow_path,
            workspace=workspace,
            arguments=arguments,
        )

    def save_workflow(
        self,
        name: str,
        workflow: dict | str | None = None,
        patch: dict | str | None = None,
        workspace: str | None = None,
    ) -> dict:
        """Save a workflow to the server's writable workflow directory,
        overwriting any existing workflow of that name there. Validate it
        first. A name resolving to a read-only source (an examples
        directory) is not overwritten - the copy lands in the writable
        directory and shadows it, adapting the example without damaging it.
        `name` may include folders.

        Give exactly one of `workflow` (the full document) or `patch`: a
        JSON Merge Patch (RFC 7396) merged onto the stored definition, so
        bumping one argument means sending just that argument -
        `{"variables": {"num_images_per_prompt": 4}}` rather than the whole
        workflow. A patch key set to `null` deletes that key; a list is
        replaced whole, never merged. Either may be a JSON-encoded string,
        parsed before saving; a parse failure is reported as invalid JSON.

        Mark each saving step's `result.subfolder` (`final`/`intermediate`)
        so a later consumer can tell the deliverable from scratch files.

        `workspace` scopes this one call without changing the session pin,
        so a save cannot be misdirected by another connection's pin change."""
        return authoring.save_workflow(
            self.client, name, workflow=workflow, patch=patch, workspace=workspace
        )

    def delete_workflow(self, name: str, workspace: str | None = None) -> dict:
        """Permanently delete a stored workflow from this workspace. A
        workflow from a read-only examples directory is refused rather than
        deleted - `list_workflows` reports which those are as
        `writable: false`.

        `workspace` scopes this one call without changing the session pin,
        movable by another connection on a mounted transport."""
        return authoring.delete_workflow(self.client, name, workspace=workspace)


class PromptTools:
    """The prompt library and the enhancers."""

    def __init__(self, client):
        self.client = client

    def list_prompts(
        self,
        tag: Optional[str] = None,
        intended_model: Optional[str] = None,
        include_text: bool = False,
    ) -> dict:
        """List the stored prompts - the worked examples a workflow reaches
        by writing "prompt:name" or "prompt:folder/name". Each entry carries
        its `description`, `intended_model`, `tags` and the size of its text;
        `get_prompt` returns the text itself. This is where the caption a
        model was trained on is already written out, so read the exemplar
        for the family you are about to run rather than inventing the
        format: `intended_model` narrows to one family, as the listing
        reports it, and `tag` to one label. `include_text=true` returns every body, which for the whole
        library is more than a client will accept - filter first."""
        return prompts.list_prompts(
            self.client,
            tag=tag,
            intended_model=intended_model,
            include_text=include_text,
        )

    def get_prompt(self, name: str) -> dict:
        """Get one stored prompt's full definition - its text, description,
        intended model and tags - by a name from `list_prompts`. The prompt
        library is shared by every workspace on the server."""
        return prompts.get_prompt(self.client, name)

    def get_prompt_schema(
        self,
    ) -> dict:
        """Get the JSON schema every stored prompt must satisfy. Check this
        before writing one, as you would get_schema before a workflow."""
        return prompts.get_prompt_schema(self.client)

    def save_prompt(self, name: str, prompt: dict | str) -> dict:
        """Save a prompt to the library, overwriting any prompt of that
        name. `prompt` is an object like {"text": "..."}; a plain string
        is saved as its `text`, one starting `{` is parsed as JSON. `text`
        may not begin with a reference prefix (variable:, previous_result:,
        constant:, asset:, output:, prompt:) - the server refuses it so a
        reference never resolves twice. The library is shared by every
        workspace on this server."""
        return prompts.save_prompt(self.client, name, prompt)

    def delete_prompt(self, name: str) -> dict:
        """Permanently delete a stored prompt. A workflow that still
        references it by "prompt:name" will fail to load."""
        return prompts.delete_prompt(self.client, name)

    def list_enhancers(
        self,
    ) -> dict:
        """List the enhancer presets `enhance_prompt` accepts - one per
        target model family. Call this before enhance_prompt rather than
        guessing a preset name."""
        return prompts.list_enhancers(self.client)

    def enhance_prompt(
        self,
        idea: str,
        preset: str = "h3",
        model_name: str | None = None,
        device: str | None = None,
        acknowledged_cost: bool = False,
    ) -> dict:
        """Expand a short idea into a full prompt with a language model.
        This costs time on the engine: it queues a real job, and the engine
        runs one job per GPU, so a generation waiting behind it is delayed.
        Tell the user what will be enhanced and get their go-ahead, then
        pass acknowledged_cost=true. Returns as soon as the job is queued;
        the enhanced text is the text file in its finished manifest."""
        return prompts.enhance_prompt(
            self.client,
            idea,
            preset=preset,
            model_name=model_name,
            device=device,
            acknowledged_cost=acknowledged_cost,
        )


class LoraTools:
    """The LoRA catalog and the opt-in Hub search."""

    def __init__(self, client):
        self.client = client

    def list_loras(
        self,
        model: Optional[str] = None,
        workflow: Optional[str] = None,
        status: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> dict:
        """LoRAs tried on a base model - proven, trial, or rejected with the
        reason. `model` is a workflow name or a Hub repo id; matching is
        exact on the base (and the H3 partition). `use_when` says when to
        reach for one; `trigger` and `scale` say how. Guide: loras."""
        return loras.list_loras(
            self.client, model=model, workflow=workflow, status=status, tag=tag
        )

    def save_lora(self, name: str, entry: dict | str) -> dict:
        """Save a catalog entry, e.g. promote a trial that worked to
        `proven` with its job in `evidence`. get_guide("loras") has the
        entry format."""
        return loras.save_lora(self.client, name, entry)

    def recommend_loras(self, model: str, query: str, limit: int = 8) -> dict:
        """Opt-in: catalog LoRAs for `model` ranked against a style request,
        then Hugging Face Hub adapters of that exact base (queries the Hub;
        no download, no GPU). Hub rows are candidates to trial, not
        recommendations - mind each one's `warnings`."""
        return loras.recommend_loras(self.client, model, query, limit=limit)
