"""Tool classes for generated outputs and the asset library.

The MCP content builders (image, audio, frames) live here with the tools
that return them, so the SDK stays out of the handler modules."""

from mcp.types import AudioContent, ImageContent, TextContent

from dw_mcp import assets, media


class MediaTools:
    """Look at, listen to, and manage generated outputs."""

    def __init__(self, client):
        self.client = client

    def get_output_image(
        self,
        name: str,
        max_dimension: int = 768,
        workspace: str | None = None,
        crop: list[int] | None = None,
    ) -> list[ImageContent | TextContent]:
        """Look at a generated image, named as `list_gallery` or a job's
        manifest reports it. Use this to judge output quality - a run that
        succeeded can still have made the wrong picture. Downscaled to
        `max_dimension` on its longest side; the second part reports the
        before/after size, so a downscale is never silent.
        `crop` is `[x, y, width, height]` in the original's pixels,
        cut before the downscale.

        `workspace` pins this call to another workspace."""
        result = media.get_output_image(
            self.client,
            name,
            max_dimension=max_dimension,
            workspace=workspace,
            crop=crop,
        )
        image = ImageContent(
            type="image", data=result["data"], mime_type=result["mime_type"]
        )
        telemetry = TextContent(
            type="text",
            text=(
                f"name: {result['name']}\n"
                f"original_size: {result['original_size']}\n"
                + (f"crop: {result['crop']}\n" if result["crop"] else "")
                + f"returned_size: {result['returned_size']}\n"
                f"bytes: {result['bytes']}"
            ),
        )
        return [image, telemetry]

    def get_output_audio(
        self,
        name: str,
        start: float | None = None,
        duration: float | None = None,
        workspace: str | None = None,
    ) -> list[AudioContent | TextContent]:
        """Listen to a generated soundtrack, named as `list_gallery` or a
        job's manifest reports it - an audio output, or a video's muxed
        track: own encoding when served whole, WAV when extracted or
        excerpted. No downscale exists for audio - a whole clip too
        large is refused; ask for a part with `start`/`duration` in
        seconds, per `get_gallery_metadata`'s envelope. The text part
        says what was cut. To *see* a video, `get_output_frames`. A
        text-only client confirms the *words* an output speaks by
        transcribing it instead: WORKFLOW_GUIDE's "The loop", step 6, in
        `get_guide("workflows", section="Authoring a workflow from an
        agent")`.

        `workspace` pins this call to another workspace."""
        result = media.get_output_audio(
            self.client, name, start=start, duration=duration, workspace=workspace
        )
        audio = AudioContent(
            type="audio", data=result["data"], mime_type=result["mime_type"]
        )
        lines = [f"name: {result['name']}", f"bytes: {result['bytes']}"]
        if result["duration_seconds"] is not None:
            lines.append(f"duration_seconds: {result['duration_seconds']}")
        if result["excerpt"]:
            e = result["excerpt"]
            lines.append(f"excerpt: {e['duration']}s from {e['start']}s of {e['of']}s")
        telemetry = TextContent(type="text", text="\n".join(lines))
        return [audio, telemetry]

    def get_output_frames(
        self,
        name: str,
        at: list[str | float] | None = None,
        seams: bool | list[int] | None = None,
        count: int | None = None,
        boundaries: list[int] | None = None,
        names: list[str] | None = None,
        max_dimension: int = 512,
        hear: float | None = None,
        workspace: str | None = None,
        crop: list[int] | None = None,
    ) -> list[ImageContent | AudioContent | TextContent]:
        """See a generated video as frames - no video content type exists
        over MCP. One selector: `count` (contact sheet), `at` (seconds or
        "frame:N"), or `seams` (true, or seam numbers from 1) for each
        join's frame pair, at a joined output's `media.shots`; else
        `boundaries` (each later shot's first frame) and `names`. Over budget, tiles shrink together.
        `hear=N` adds N seconds of soundtrack around each `at`.
        `crop` is `[x, y, width, height]` in the video's own source
        pixels, cut from every frame before any downscale, like
        `get_output_image`'s.

        `workspace` pins this call to another workspace."""
        result = media.get_output_frames(
            self.client,
            name,
            at=at,
            seams=seams,
            count=count,
            boundaries=boundaries,
            names=names,
            max_dimension=max_dimension,
            hear=hear,
            workspace=workspace,
            crop=crop,
        )
        parts = []
        for tile in result["tiles"]:
            parts.append(
                ImageContent(
                    type="image", data=tile["data"], mime_type=tile["mime_type"]
                )
            )
            if "audio" in tile:
                parts.append(
                    AudioContent(
                        type="audio",
                        data=tile["audio"]["data"],
                        mime_type=tile["audio"]["mime_type"],
                    )
                )
        lines = [
            f"name: {result['name']}",
            f"frame_count: {result['frame_count']}  fps: {result['fps']}",
        ]
        if result.get("crop"):
            lines.append(f"crop: {result['crop']}")
        fps = result["fps"]
        for tile in result["tiles"]:
            if tile.get("frames"):
                # a contact sheet: every cell, so each one can be located
                cells = ", ".join(
                    f"{frame} ({frame / fps:.2f}s)" if fps else str(frame)
                    for frame in tile["frames"]
                )
                where = f"frames: {cells}"
            else:
                where = f"frame {tile['frame']} @ {tile['seconds']:.2f}s"
            if tile.get("difference") is not None:
                where += f"  difference: {tile['difference']}"
            if tile.get("audio_error"):
                where += f"  hear: {tile['audio_error']}"
            lines.append(
                f"- {tile['label']}  {where}  [{tile['width']}x{tile['height']}]"
            )
        if result["downscaled_to"]:
            lines.append(
                f"downscaled_to: {result['downscaled_to']} (every tile, to fit the inline budget)"
            )
        if result.get("hear"):
            lines.append(f"hear: {result['hear']}s around each moment")
        if result.get("audio_truncated"):
            lines.append(
                "audio_truncated: some tiles' audio was skipped to stay within "
                "the response size budget"
            )
        parts.append(TextContent(type="text", text="\n".join(lines)))
        return parts

    def get_output_text(
        self, name: str, max_characters: int = 20000, workspace: str | None = None
    ) -> dict:
        """Read a text output - a prompt enhancement, or any step whose
        result is text/plain or JSON. Truncated to `max_characters`, and
        the reply says how long the file really was.

        `workspace` names the workspace for this one call without
        switching the session to it - the same pin `run_workflow`
        takes, so a job run into another workspace is reachable from
        here without leaving this one."""
        return media.get_output_text(
            self.client, name, max_characters=max_characters, workspace=workspace
        )

    def assess_output(
        self,
        name: str,
        probe: str | None = None,
        detail: bool = False,
        workspace: str | None = None,
    ) -> dict:
        """Measure a finished cut - seams, shot levels, sync - on the
        server, without queueing. Findings are places to look, not
        verdicts: check each with get_output_frames/get_output_audio.
        `probe` (analyze_shots, analyze_seams, analyze_sync_drift) returns
        one probe's full body; `detail` adds every probe's. Takes `asset:`."""
        return media.assess_output(
            self.client, name, probe=probe, detail=detail, workspace=workspace
        )

    def delete_output(
        self,
        name: str | None = None,
        workspace: str | None = None,
        job_id: str | None = None,
    ) -> dict:
        """Permanently remove one generated file from the output directory.
        Not recoverable (rerun the job to get it back), and any `output:`
        reference to it stops resolving; prefer `keep_output` if it is
        worth keeping. When it was the last media file of its run, the run
        directory goes with it, sidecars included. `name` may also be a run
        directory ("<workflow>/<run id>", the first two parts of a gallery
        name), which removes the whole run - the only handle on a run that
        failed before writing any media - or give `job_id` instead: the run
        that job wrote is removed whole, and the reply adds `job_id` and
        the resolved `run_dir`. Exactly one of the two; a job with no run
        directory, unknown, or still running is an error.

        `workspace` pins a `name` delete to another workspace without
        switching the session; a `job_id` delete goes to the workspace the
        job ran in."""
        return media.delete_output(
            self.client, name, workspace=workspace, job_id=job_id
        )

    def download_output(
        self,
        name: str,
        destination: str | None = None,
        overwrite: bool = False,
        workspace: str | None = None,
    ) -> dict:
        """Save one output file to disk on the
        machine running the MCP server - for the stdio `dw-mcp` that is
        your own machine; for a
        `dw.serve --mcp` endpoint it is the GPU box, and this tool is not
        the way to get a file to where you are (use get_output_image /
        get_output_text for inline content, or the `url` that
        `list_gallery` reports for each entry, which already carries the
        workspace selector - do not build an /outputs URL by hand). This is
        also not how a generated file becomes an input for a later
        workflow: use `keep_output`, which links it inside the workspace
        under an `asset:` name, rather than writing into the server's asset
        directory behind the API's back. Unlike the inline tools, this works
        for any file type, streams the body straight to disk rather than
        buffering it, and returns no content to the conversation - only
        where it was saved. `destination` may be a
        full path or a directory; a '..' path segment in it is refused. An
        existing file at the resolved path is left alone unless
        `overwrite=True`. On the stdio `dw-mcp`, omitting `destination`
        saves into the current working directory under the output's own
        name. On a `dw.serve --mcp` endpoint the save happens on the server, and destination is required there -
        an omitted one is refused rather than dropped loose in the
        workspace root, where nothing can find or delete it later; use
        the `url` list_gallery reports, get_output_image/get_output_audio/
        get_output_frames for inline content, or keep_output to make it a
        named asset instead.

        `workspace` names the workspace for this one call without
        switching the session to it - the same pin `run_workflow`
        takes, so a job run into another workspace is reachable from
        here without leaving this one."""
        return media.download_output(
            self.client,
            name,
            destination=destination,
            overwrite=overwrite,
            workspace=workspace,
        )


class AssetTools:
    """The asset library."""

    def __init__(self, client):
        self.client = client

    def list_assets(self, detail: bool = False, workspace: str | None = None) -> dict:
        """List the input media on the server, each with the `asset:`
        reference a workflow argument carries. Look here before asking for
        a file: what a workflow needs may already be there. Entries carry
        name, reference, kind, size and origin only - for duration, frame
        count, fps, sample rate or channels, pass the reference to
        `get_gallery_metadata`, which reads inputs as well as outputs. Pass
        detail=true for each entry's folder, mtime and url too, needed
        before naming a shared library's writable/read-only roots or
        opening the file's preview URL.

        `workspace` scopes this one call without changing the session pin."""
        return assets.list_assets(self.client, detail=detail, workspace=workspace)

    def upload_asset(
        self,
        file_path: str | None = None,
        content: str | None = None,
        asset_name: str | None = None,
        shared: bool = False,
        workspace: str | None = None,
    ) -> dict:
        """Put an image, video or audio file into the server's asset
        library and get back the `asset:` reference to use in a workflow.
        Pass exactly one of `file_path` or `content`.

        `file_path` is read from the machine this MCP server runs on and
        pushed to the engine - how an input reaches a dw.serve running
        elsewhere. When this MCP surface is served by dw.serve itself,
        "this machine" is the engine's own box, so `file_path` is confined
        to the directories it works in - a file that exists only on your
        own machine cannot be named this way.

        `content` is for that case: the file's bytes, base64-encoded, sent
        inline rather than read off disk. Use it for a file that lives
        only on the machine you're running on, against a remote
        `dw.serve --mcp` endpoint with no filesystem in common with you.
        Capped at 4MB, well under `file_path`'s 200MB. `asset_name` is
        required with `content`, since there is no file to name it from.

        Accepts the usual image, video and audio extensions; reference the
        result, not a path. Pass `asset_name` for a readable stored name
        ("cast/priya-voice.wav", folders allowed) - otherwise (with
        `file_path`) the name is random. Pass `shared=true` to put it in
        the library every workspace shares - where a recurring cast
        belongs, since a workspace's own assets are invisible from the next.

        `workspace` scopes this one call without changing the session pin."""
        return assets.upload_asset(
            self.client,
            file_path=file_path,
            content=content,
            asset_name=asset_name,
            shared=shared,
            workspace=workspace,
        )

    def keep_output(
        self,
        name: str,
        asset_name: str | None = None,
        overwrite: bool = False,
        shared: bool = False,
        workspace: str | None = None,
    ) -> dict:
        """Keep a generated file as an input asset under a stable `asset:`
        name, so later workflows can rely on it - a run's own name moves
        ("latest") or breaks when outputs are pruned. `name` is a gallery
        name; `asset_name` defaults to the file's own. The copy happens on
        the server, inside the workspace: nothing is downloaded or
        re-uploaded. Pass `shared=true` to keep it in the library every
        workspace shares instead.

        `workspace` scopes this one call without changing the session pin."""
        return assets.keep_output(
            self.client,
            name,
            asset_name=asset_name,
            overwrite=overwrite,
            shared=shared,
            workspace=workspace,
        )

    def delete_asset(self, name: str, workspace: str | None = None) -> dict:
        """Permanently remove one file from the asset library, by the name
        `list_assets` reports (without the `asset:` prefix). Not
        recoverable, and any workflow still carrying that reference stops
        loading. Deletes from whichever library holds it - this
        workspace's own before the shared one, the order an `asset:`
        reference resolves in; one from a read-only examples library is
        refused.

        `workspace` scopes this one call without changing the session pin,
        movable by another connection on a mounted transport."""
        return assets.delete_asset(self.client, name, workspace=workspace)
