# Scope: Server-side media analysis helpers

**Status:** Draft / proposal — not implemented
**Date:** 2026-09-04
**Context:** Arose from the MarmotCroons full-song work; see the lip-sync and
MarmotCroons findings in the project memory notes.

## Goal

Make the **measure → plan → author → run → verify** loop available to any client
of `dw.serve`, including an MCP-only Claude Code session with no GPU, no torch,
and no local checkout.

## Motivation

The MarmotCroons work split into two halves with very different portability.
Authoring and running are already in the MCP surface. **Measuring is not** — the
mid/side vocal map, the drift check and the sharpness numbers were ad-hoc numpy
run over Bash.

Those measurements are what made the result good:

- Shot placement came from a vocal-presence table, not from taste.
- The framing-collapse fix was found by measuring pairwise distance
  (4.6–9.0 collapsed, vs 33–56 healthy).

An MCP-only client can currently generate but not perceive.

## Constraints that shape the design

1. **`dw_mcp` must not import `dw.*`** — a test guards this, because
   `dw/__init__.py` pulls in torch. All analysis runs server-side; the MCP half
   stays a thin HTTP client.
2. **Summaries cross the wire, never raw samples.** A 3.6M-sample track reduces
   to a 16-row table. This is the whole economic argument — DSP in numpy is cheap
   and correct; DSP in a context window is neither.
3. **CPU-only, seconds not minutes.** No GPU, so no `acknowledged_cost=True`
   gate. These are cheap tools an agent can call freely while iterating — which
   is precisely what makes interactive tuning viable.
4. **Analysis reads *inputs*, which the server currently may not.** See
   Security; this is the one genuine widening of the trust surface.

## Proposed modules

| Layer             | File                    | Pattern it follows                        |
| ----------------- | ----------------------- | ----------------------------------------- |
| Analysis core     | `dw/analysis.py`        | peer of `introspection.py` / `hub_cache.py` |
| Routes            | in `dw/server/app.py`   | existing endpoint conventions             |
| MCP handlers      | `dw_mcp/analysis.py`    | plain `(client, **kwargs)`, like `catalog.py` |
| Tool registration | `dw_mcp/server.py`      | only file importing the MCP SDK           |

Reuse `dw/tasks/audio_utils.py` (`load_audio`, `as_channels_samples`,
`frames_to_samples`) rather than reimplementing I/O.

## Endpoints

### 1. `POST /analyze/audio` — source-side, informs authoring

Request:

```json
{
  "source": "local_inputs/chia.mp3",
  "cell_ms": 250,
  "fps": 24,
  "shot_frames": 124
}
