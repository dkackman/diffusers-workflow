# Remaining work: move model-specific narrative out of tool descriptions (#101, recommendation 4)

Split from `mcp-context-cost-complete.md` on 2026-09-20. Recommendations 1-3,
5, and 6 have shipped (`get_guide` without `section` returns a summary,
`get_schema` takes a `section` argument, `list_workflows()` with no shape
summarises, the skills were correctly left alone, and this harness confirms
it defers MCP tool schemas). What remains is recommendation 4, unimplemented:

## The remaining item

`dw_mcp/diagnose.py`'s `wait_for_job` docstring still carries the measured,
model-specific narrative inline — H3's 629 s video-reference lead-in and the
block-cache-gap arithmetic — rather than the rule alone. `get_memory`'s
pinned-host-cache explanation is the same shape of problem.

**Move measured, model-specific narrative out of tool descriptions into the
skills and guides.** Keep every *rule* in the description (what `live: false`
means, that silence is not a hang, that the cap is 55 s); move the *numbers
and worked examples* to the `minimax-h3` skill, which loads exactly when H3 is
in play, and to a guide section. Realistic trim on the top five: ~8,350 chars
→ ~4,000, saving ~1.1k tokens of connect cost.

## Remaining steps

1. Trim `wait_for_job`'s docstring in `dw_mcp/diagnose.py` (currently still
   contains "measured 629 s for one 5 s 960x544..." and similar) down to the
   rule only.
2. Do the same for `get_memory`'s pinned-host-cache explanation and any other
   tool description carrying worked numeric examples rather than rules.
3. Relocate the removed numbers/examples into `plugins/dw/skills/minimax-h3/SKILL.md`
   and/or a guide section (`docs/WORKFLOW_GUIDE.md` or similar).
4. Re-run the ticket cases that motivated the original wording, since this can
   reintroduce a closed ticket if done carelessly — the description-only
   change still wants a tester verifying nothing regresses.

## Note carried over from the original proposal

This item was explicitly recommended only "if item 6 comes back negative" —
item 6 (whether the harness defers MCP tool schemas on connect) was confirmed
**true** for this session's own harness. If the tester's actual driver
sessions also get that treatment, the ~10.6k connect-cost saving item 6
describes already applies without any `dw` change, which may make this
trimming work not worth doing. Confirm the driver's behavior before spending
effort here.
