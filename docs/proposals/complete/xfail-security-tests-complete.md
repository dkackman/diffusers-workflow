# Close the xfail security tests (#407)

Written by model `claude-opus-5-5` via provider `anthropic`, at close-out
2026-09-24, from plan v2 on #407 and the five stage threads (#409-#413).

## The ask

An audit left 23 strict markers (22 pytest xfails in 7 files, one Playwright
`test.fail`) describing security gaps a third-party workflow could reach.
#407 asked which were real and to make those pass. Verdict: **build
smaller** - close the real ones in five stages, cut three.

Don's answers (2026-09-24): Q1 HTML outputs - block for now (refuse at
validation *and* sandbox on serve); Q2 trust allowlist - keep whole-package
allowing, gate what a name may resolve to; Q3 pixel limit - 50 M; Q4 the
25,000-character default test - delete.

## What was built

| Stage | Issue | Merge on `develop` | What it closed |
|---|---|---|---|
| 1 trust gate + URL host | #409 | `9a3f629` | A `*_type` / `config_type` / `dtype` / `*_dtype` name must resolve to a class (or a `torch.dtype` for dtype keys) when untrusted, so `torch.hub.load` is refused. A `constant:` walk may not leave the trusted packages or pass a `_` name, so `constant:torch.os.environ` is refused. `validate_url` refuses `\`, which closed the `evil\@huggingface.co` HF-token leak. Validate reports all of it at the value's path. |
| 2 active result types + web headers | #410 | `779778b` | `content_type_fault` refuses `text/html` / `text/xml` results. Every response carries `nosniff` and `X-Frame-Options: DENY`. `/outputs` and `/inputs` add `Content-Security-Policy: sandbox` for the five active types, on the `StaticFiles` response so ETag/304 and Range/206 still work. A malformed `Origin` answers 403, not 500. |
| 3 SSRF redirects + CGNAT | #411 | `680937f` | `safe_get` follows at most 5 hops (`MAX_MEDIA_REDIRECTS`) and re-checks each `Location`. `_is_internal` refuses any non-global address, which covers 100.64/10 (Tailscale). |
| 4 symlink containment | #412 | `aede93f` | Gallery, asset, workflow and prompt listings and the export zip drop links that escape their root, compared against `realpath(root)`. |
| 5 pixel limit | #413 | `5d4bfe4` | `MAX_DECODE_PIXELS = 50_000_000`, checked after `Image.open`: `get_output_image` (crops too) refuses, the thumbnail route answers 413, embedded metadata is read from PNG header chunks without decoding. `dw_mcp` has its own copy, pinned equal. |

Two implementer fixes came out of verifying stage 2, outside the stage
chain: **#414** (`run_workflow` queued a job whose argument-resolved
`content_type` was `text/html`, which validate refused) and **#415**
(`JobManager.submit` and the worker's execute path refused a document default
the caller's argument overrode). Both now check `content_type` against the
caller's arguments, the way `validation_errors` does.

Docs: `docs/SECURITY.md` (trust scope, refused ranges, redirects, pixel
limit, refused result types) and `docs/SERVER.md` (response headers) were
updated by the stages.

## Bounces

- **#409: one.** Validate didn't check `constant:` values at all, and didn't
  check `*_dtype` keys, so both refusals landed only at run time. The rebuild
  made `component_type_errors` (`dw/introspection.py`) walk the whole step,
  resolving every literal `constant:` as the run does and gating every key
  the run loads as a type. It skips `{media_type, location}` dicts, whose
  `media_type` names a kind of media, not a type.
- **#410-#413: none.** (#414 and #415 were filed against #410's behavior and
  fixed as separate tickets.)

Cost is left out: the stage comments don't record `usage:` figures. The
plan's estimate was $16-21.

## Deferred, and why

- **E1, the variable default-length test** - deleted (Q4). The cap is
  20,000 (`MAX_VARIABLE_VALUE_LENGTH`); 25,000 was only the test's probe. The
  author owns the definition and the file is already capped at 50 MB.
  *Would come back if* a default ever reaches somewhere its length costs
  more than the file it sits in.
- **A Content-Security-Policy on the UI itself** - kept as an xfail, reason
  "deferred by #407" (`tests/test_security_web_headers.py`). Defence in depth
  after an XSS, and stage 2 closed the only known route. A working policy
  needs `data:`/`blob:`/remote media and Monaco's inline styles and blob
  workers, with no browser in CI to catch a break. *Would come back if* a
  second XSS route turns up or Playwright reaches CI.
- **Playwright in CI** - no. The fixture needs the torch backend. The pytest
  header suite covers the contract; `security.spec.ts` stays a local
  `npm run e2e` check.
- **Video and audio decode limits** - not in scope; no test and no measured
  threat.

## Design corrections found against the audit's notes

- There was no 25,000-character cap; the cap is 20,000.
- `/outputs` and `/inputs` are routes over a cached `StaticFiles`, not plain
  mounts, so headers go on the returned response.
- Only `text/html` and `text/xml` can be *written* by a workflow; `.svg` and
  `.xhtml` arrive only as planted files, which is why the sandbox header
  stays beside the validation refusal.
- `dw_mcp` cannot import `dw` (the torch boundary), hence the second
  `MAX_DECODE_PIXELS`.

## Behavior changes to name in the release note

Untrusted, a workflow naming a non-class under a `*_type` / dtype key, a
`constant:` outside the trusted packages or through a `_` name, a URL with
`\`, a media URL on a non-global address (including a tailnet host), or a
`text/html` / `text/xml` result is now refused. `--trust-workflows` lifts
the first two. Nothing in the catalog is affected.
