"""The notes channel on the wire: `python -m dw_mcp --notes-channel` as a
real subprocess against a real dw.serve on a loopback socket.

Claude Code is not in the loop here - what is held is the contract it
documents (code.claude.com/docs/en/channels-reference): a stdio server whose
`initialize` result carries `experimental["claude/channel"] == {}` and its
instructions, that negotiates the handshake protocol (never 2026-07-28),
sends `notifications/claude/channel` with `content` and a flat string
`meta`, and writes nothing but JSON-RPC to stdout.
"""

import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

pytest.importorskip("mcp", reason="the mcp extra is not installed")

import uvicorn  # noqa: E402
from mcp_types.version import HANDSHAKE_PROTOCOL_VERSIONS  # noqa: E402

from dw.server.app import create_app  # noqa: E402
from dw.server.jobs import JobManager  # noqa: E402
from dw.workspace import Workspace  # noqa: E402
from dw_mcp import server as mcp_server  # noqa: E402
from dw_mcp.client import DwApiError  # noqa: E402
from dw_mcp.server import NOTES_INSTRUCTIONS  # noqa: E402

from .notes_helpers import save_note  # noqa: E402
from .test_server import ScriptedWorkerManager, success_script  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TOKEN = "notes-token"
WAIT = 20  # seconds: far above the 2 second poll, for a loaded machine


class DwServer:
    """A dw.serve on 127.0.0.1, and the user's side of its REST API."""

    def __init__(self, tmp_path):
        root = Workspace(tmp_path / "studio", "flag").ensure()
        manager = JobManager(
            root.outputs,
            worker_manager=ScriptedWorkerManager(success_script),
            history_path=str(tmp_path / "jobs.sqlite"),
            workflow_dir=root.workflows,
        )
        app = create_app(
            workflow_dir=root.workflows,
            output_dir=root.outputs,
            job_manager=manager,
            prompt_dir=root.prompts,
            asset_dir=root.assets,
            token=TOKEN,
            workspace=root.root,
        )
        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
        self.uvicorn = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.uvicorn.run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + WAIT
        while not self.uvicorn.started:
            assert time.monotonic() < deadline, "dw.serve did not start"
            time.sleep(0.02)
        port = self.uvicorn.servers[0].sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self.root = root.root
        self.api = httpx.Client(
            base_url=self.url, headers={"Authorization": f"Bearer {TOKEN}"}
        )

    def put(self, request_id, **fields):
        """One write of the whole plan, as the web UI makes it (`by` user
        unless given)."""
        path = f"/api/workspaces/default/requests/{request_id}"
        found = self.api.get(path)
        current = found.json() if found.status_code == 200 else None
        body = {
            "title": "Lighthouse short",
            "goal": "three shots",
            "items": current["items"] if current else [],
            "revision": current["revision"] if current else 0,
            "why": "test write",
            **fields,
        }
        response = self.api.put(path, json=body)
        assert response.status_code == 200, response.text
        return response.json()

    def save_note(self, request_id, text, item=None):
        """The user saves a note: the note is stored first, then a user write
        moves the revision, so a poll never sees the revision without it."""
        at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        saved = save_note(self.root, request_id, text, item, at)
        return saved, self.put(request_id, why="note")

    def stop(self):
        self.api.close()
        self.uvicorn.should_exit = True
        self.thread.join(WAIT)


@pytest.fixture
def dw_server(tmp_path):
    server = DwServer(tmp_path)
    yield server
    server.stop()


def channel_command(url, *extra):
    return [
        sys.executable,
        "-m",
        "dw_mcp",
        "--notes-channel",
        "--url",
        url,
        "--interval",
        "2",
        *extra,
    ]


def channel_env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("DW_")}
    env["PYTHONPATH"] = str(REPO)
    env.update(extra)
    return env


class Channel:
    """The channel process, driven the way Claude Code drives a stdio
    server: one JSON-RPC message per line each way."""

    def __init__(self, url, pump_stdout=True, **env):
        self.pumped = pump_stdout
        self.process = subprocess.Popen(
            channel_command(url),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=channel_env(**env),
            cwd=str(REPO),
        )
        self.stdout_lines = []
        self.stderr_lines = []
        self.messages = queue.Queue()
        if pump_stdout:
            self._pump(self.process.stdout, self.stdout_lines, self.messages)
        self._pump(self.process.stderr, self.stderr_lines, None)

    def _pump(self, stream, lines, messages):
        def read():
            try:
                for raw in stream:
                    line = raw.decode("utf-8")
                    lines.append(line)
                    if messages is not None:
                        messages.put(line)
            except ValueError:  # the test closed the pipe under the reader
                pass

        threading.Thread(target=read, daemon=True).start()

    def send(self, message):
        self.process.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
        self.process.stdin.flush()

    def receive(self):
        if not self.pumped:  # read here, so the test may close the pipe
            return json.loads(self.process.stdout.readline())
        return json.loads(self.messages.get(timeout=WAIT))

    def request(self, request_id, method, params=None):
        message = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        self.send(message)
        answer = self.receive()
        assert answer.get("id") == request_id, answer
        return answer

    def handshake(self):
        result = self.request(
            1,
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "0"},
            },
        )["result"]
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return result

    def finish(self):
        """Close stdin, as a client that is done does, and return the exit
        code."""
        self.process.stdin.close()
        return self.process.wait(WAIT)

    def stderr(self):
        return "".join(self.stderr_lines)

    def kill(self):
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(WAIT)


@pytest.fixture
def channel():
    started = []

    def start(url, **kwargs):
        started.append(Channel(url, **kwargs))
        return started[-1]

    yield start
    for process in started:
        process.kill()


def test_a_saved_note_reaches_the_client_as_a_channel_notification(dw_server, channel):
    dw_server.put(
        "lighthouse",
        by="agent",
        items=[
            {"key": "shot2", "title": "Shot 2: stairs", "depends_on": []},
            {"key": "shot3", "title": "Shot 3: lamp room", "depends_on": []},
        ],
    )
    dw_server.save_note("lighthouse", "colder overall")
    proc = channel(dw_server.url, DW_API_TOKEN=TOKEN)

    # A client on the 2026-07-28 revision probes first. The channel must not
    # take it up: Claude Code does not register such a server as a channel.
    probe = proc.request(
        0,
        "server/discover",
        {
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                "io.modelcontextprotocol/clientInfo": {"name": "t", "version": "0"},
                "io.modelcontextprotocol/clientCapabilities": {},
            }
        },
    )
    assert probe["error"]["code"] == -32601, probe

    result = proc.handshake()
    assert result["protocolVersion"] in HANDSHAKE_PROTOCOL_VERSIONS
    assert result["capabilities"] == {"experimental": {"claude/channel": {}}}
    assert result["serverInfo"]["name"] == "dw-notes"
    assert result["instructions"] == NOTES_INSTRUCTIONS

    # what was waiting before the session: one digest
    digest = proc.receive()
    assert digest["method"] == "notifications/claude/channel"
    assert "id" not in digest
    assert digest["params"]["meta"] == {
        "kind": "notes_waiting",
        "workspace": "default",
        "count": "1",
    }
    assert "colder overall" in digest["params"]["content"]

    # the user saves a note on an item in the web UI
    text = "this one has off lipsync - reroll it and recreate the final piece"
    saved, written = dw_server.save_note("lighthouse", text, "shot3")
    saved = saved["at"]

    event = proc.receive()
    assert set(event) == {"jsonrpc", "method", "params"}
    assert event["method"] == "notifications/claude/channel"
    assert set(event["params"]) == {"content", "meta"}
    assert event["params"]["meta"] == {
        "kind": "note",
        "workspace": "default",
        "request": "lighthouse",
        "item": "shot3",
        "revision": str(written["revision"]),
        "note": "n2",
        "created": saved,
    }
    minute = saved[:16].replace("T", " ")
    assert event["params"]["content"] == (
        'Note from the user on request "lighthouse", item "shot3" '
        f"(Shot 3: lamp room), saved {minute} UTC:\n\n"
        f"{text}\n\n"
        'Read the request with get_request("lighthouse", workspace="default") '
        'and act on the note; say what you did in the "why" of your next '
        "set_request, "
        'naming the note id you answer ("n3: rerolled shot3 as take 3").'
    )

    # no tools: a second copy of dw's tool surface is what this mode avoids
    assert proc.request(2, "tools/list")["error"]["code"] == -32601

    # the agent answers with a write, and nothing more is announced
    dw_server.put("lighthouse", by="agent", why="rerolled shot3")
    assert proc.request(3, "ping") == {"jsonrpc": "2.0", "id": 3, "result": {}}

    assert proc.finish() == 0
    for line in proc.stdout_lines:
        assert json.loads(line)["jsonrpc"] == "2.0", line
    assert len(proc.stdout_lines) == 6  # 4 answers and the 2 events
    assert "connected to" in proc.stderr()
    assert str(dw_server.url) in proc.stderr()


def test_a_server_that_is_down_is_retried_quietly_and_the_channel_stays_up(channel):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
    proc = channel(f"http://127.0.0.1:{closed_port}")

    result = proc.handshake()
    assert result["capabilities"] == {"experimental": {"claude/channel": {}}}
    # two polls have failed by the time the second ping is answered
    assert proc.request(2, "ping")["result"] == {}
    deadline = time.monotonic() + WAIT
    while "will keep trying" not in proc.stderr():
        assert time.monotonic() < deadline, proc.stderr()
        time.sleep(0.05)
    time.sleep(2.5)
    assert proc.request(3, "ping")["result"] == {}
    assert proc.process.poll() is None

    assert proc.finish() == 0
    assert proc.stderr().count("will keep trying") == 1
    assert len(proc.stdout_lines) == 3
    for line in proc.stdout_lines:
        assert json.loads(line)["jsonrpc"] == "2.0", line


def test_a_client_that_went_away_ends_the_process_cleanly(dw_server, channel):
    dw_server.put("lighthouse", by="agent")
    proc = channel(dw_server.url, pump_stdout=False, DW_API_TOKEN=TOKEN)
    proc.handshake()
    assert proc.request(2, "ping")["result"] == {}

    # the client's end of our stdout goes first, and an event is written
    # into it before our stdin reads end-of-file
    proc.process.stdout.close()
    dw_server.save_note("lighthouse", "anyone there?")
    time.sleep(4)  # two polls: the event has gone into the closed pipe

    assert proc.finish() == 0
    assert "the client went away" in proc.stderr()
    assert "Traceback" not in proc.stderr()


def test_the_sdk_client_negotiates_the_handshake_protocol_with_it(dw_server):
    """The MCP SDK's own client, in its default `auto` mode, probes
    `server/discover` at 2026-07-28 and falls back to `initialize` - the same
    policy Claude Code's v2 MCP runtime applies to a stdio server."""
    import anyio
    from mcp.client import Client
    from mcp.client.stdio import StdioServerParameters

    command = channel_command(dw_server.url)
    server = StdioServerParameters(
        command=command[0],
        args=command[1:],
        env=channel_env(DW_API_TOKEN=TOKEN),
        cwd=str(REPO),
    )

    async def connect():
        async with Client(server) as client:
            capabilities = client.session.server_capabilities
            return client.protocol_version, capabilities

    version, capabilities = anyio.run(connect)

    assert version in HANDSHAKE_PROTOCOL_VERSIONS
    assert capabilities.experimental == {"claude/channel": {}}
    assert capabilities.tools is None


# --- the same pieces in process ---------------------------------------------


def test_the_channel_declares_the_capability_and_nothing_else():
    server, options = mcp_server.build_notes_channel()
    capabilities = options.capabilities
    assert capabilities.experimental == {"claude/channel": {}}
    assert capabilities.tools is None
    assert capabilities.prompts is None and capabilities.resources is None
    assert options.server_name == "dw-notes"
    assert options.instructions == NOTES_INSTRUCTIONS
    assert server.get_request_handler("tools/list") is None


def test_the_channel_instructions_are_short_and_say_what_to_do():
    assert len(NOTES_INSTRUCTIONS) <= 900
    for needed in ("get_request", "set_request", "`why`", "notes_waiting", "one-way"):
        assert needed in NOTES_INSTRUCTIONS


class Stop(BaseException):
    """Ends the announce loop from its sleep once the script has run; not
    an Exception, which the loop survives."""


def announce(polls):
    """Run the announce loop over scripted polls (an event list, or an
    error to raise) until they run out. Returns what was sent, each delay
    slept and each line logged."""
    import anyio

    sent, delays, said = [], [], []
    script = iter(polls)

    class Watch:
        fault = None

        notice = None

        def poll(self, client):
            step = next(script)
            if isinstance(step, Exception):
                raise step
            self.fault = step.get("fault")
            self.notice = step.get("notice")
            return step["events"]

    class Connection:
        def __init__(self):
            self.initialized = anyio.Event()
            self.initialized.set()

        async def notify(self, method, params):
            sent.append((method, params))

    async def sleep(seconds):
        delays.append(seconds)
        if len(delays) == len(polls) + 1:  # the wait after the handshake
            raise Stop

    async def run():
        await mcp_server._announce(
            Connection(), None, Watch(), 10.0, sleep, said.append
        )

    with pytest.raises(Stop):
        anyio.run(run)
    return sent, delays, said


def test_events_are_sent_in_order_as_channel_notifications():
    first, second = {"content": "a", "meta": {}}, {"content": "b", "meta": {}}
    sent, delays, said = announce([{"events": [first, second]}, {"events": []}])
    assert sent == [
        ("notifications/claude/channel", first),
        ("notifications/claude/channel", second),
    ]
    assert delays == [5.0, 10.0, 10.0]  # the first is the wait to be listened to
    assert said == []


def test_a_failing_server_is_logged_once_backed_off_and_recovered_from():
    down = DwApiError("Cannot reach diffusers-workflow at http://x")
    event = {"content": "a", "meta": {}}
    sent, delays, said = announce(
        [
            {"events": []},
            down,
            down,
            DwApiError("bad token", status_code=401),
            httpx.ReadError("reset"),
            AttributeError("'list' object has no attribute 'get'"),
            {"events": [event]},
            {"events": [], "fault": "one request timed out"},
            {"events": []},
        ]
    )
    # a fault in part of the ledger (one request, one workspace) is said but
    # not backed off: the rest is still read every interval (#1529)
    assert delays[1:] == [10.0, 20.0, 40.0, 80.0, 160.0, 300.0, 10.0, 10.0, 10.0]
    assert said == [
        f"could not read the ledger, will keep trying: {down}",
        "could not read the ledger, will keep trying: bad token",
        "could not read the ledger, will keep trying: reset",
        "could not read the ledger, will keep trying: AttributeError: "
        "'list' object has no attribute 'get'",
        "reading the ledger again",
        "could not read the ledger, will keep trying: one request timed out",
        "reading the ledger again",
    ]
    assert sent == [("notifications/claude/channel", event)]


def test_a_notice_is_said_once_when_it_changes_and_is_not_a_fault():
    note = "the server has no GET /api/requests: watching default only"
    sent, delays, said = announce(
        [
            {"events": [], "notice": note},
            {"events": [], "notice": note},
            {"events": []},
        ]
    )
    assert said == [note]
    assert delays[1:] == [10.0, 10.0, 10.0]
