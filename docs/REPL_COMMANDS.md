# REPL Command Reference

The REPL uses hierarchical commands grouped by function. Use `?` after any command group to see its subcommands.

## Starting the REPL

```bash
python -m dw.repl
python -m dw.repl -l DEBUG    # with debug logging
```

## Commands

### workflow — Manage workflow files

```text
workflow load <file>    Load a workflow from JSON file
workflow reload         Reload current workflow from disk
workflow status         Show current workflow information
```

`workflow load` searches `./examples` by default, so `workflow load FluxDev` works.

### arg — Set workflow variables

```text
arg show                Show available variables and current values
arg set <name>=<value>  Set a variable value
arg clear               Clear all variable values
```

### model — Execute workflows

```text
model run               Execute the loaded workflow
model restart           Restart worker process (clears GPU cache)
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

## Typical Session

```bash
dw> workflow load FluxDev
dw> arg show
dw> arg set prompt="a majestic mountain landscape"
dw> model run
dw> arg set prompt="a serene beach at sunset"
dw> model run
dw> memory show
dw> exit
```

## Quick Reference

```text
workflow ── load <file> | reload | status
arg      ── show | set <name>=<value> | clear
model    ── run | restart
memory   ── show | clear
config   ── show | set <name>=<value>
```
