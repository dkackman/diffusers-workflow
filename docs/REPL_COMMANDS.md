# REPL Command Reference

The REPL uses hierarchical commands grouped by function. Use `?` after any command group to see its subcommands.

## Starting the REPL

```bash
python -m dw.repl
python -m dw.repl -l DEBUG    # with debug logging
```

## Commands

### workflow — Manage and run workflows

```text
workflow list            List workflows under the workflow directory
workflow load <file>     Load a workflow from JSON file
workflow reload          Reload current workflow from disk
workflow status          Show current workflow information
workflow run             Execute the currently loaded workflow
workflow run <n>=<v> ... Set arguments and run in one line
workflow run ask <arg>   Prompt for one argument's value, then run
workflow restart         Restart the worker process (clears GPU cache)
```

`workflow load` resolves names against the workflow directory (`./examples`
by default), including subfolders - `workflow load flux/FluxDev` loads
`examples/flux/FluxDev.json`. `workflow list` shows the available names.

`workflow run ask <arg>` prompts you interactively for `<arg>`'s value (the
value is not saved to shell/readline history) before running — a shortcut
for `arg set` followed by `workflow run`.

### arg — Set workflow variables

```text
arg show                Show available variables and current values
arg set <name>=<value>  Set a variable value
arg clear [<name>]      Clear one variable, or all of them
```

### memory — Monitor GPU memory

```text
memory show             Show current GPU memory usage
memory clear            Clear GPU memory and cached models
```

### config — REPL settings

```text
config show                   Show all settings
config set output_dir=<path>  Change output directory
config set log_level=DEBUG    Change log level
config set workflow_dir=<path> Change default workflow directory
```

### General

```text
help / ?     Show all commands
exit / quit  Exit the REPL
```

`run`, `load`, and `set` also work at the top level as shortcuts for
`workflow run`, `workflow load`, and `arg set`.

Ctrl+C during a run cancels it cooperatively and keeps the worker's models
cached; a second Ctrl+C stops the worker process itself.

## Typical Session

```bash
dw> workflow load flux/FluxDev
dw> arg show
dw> arg set prompt="a majestic mountain landscape"
dw> workflow run
dw> arg set prompt="a serene beach at sunset"
dw> workflow run
dw> memory show
dw> exit
```

## Quick Reference

```text
workflow ── list | load <file> | reload | status | run [<n>=<v> ... | ask <arg>] | restart
arg      ── show | set <name>=<value> | clear [<name>]
memory   ── show | clear
config   ── show | set <name>=<value>
```
