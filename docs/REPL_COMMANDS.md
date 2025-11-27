# REPL Command Reference

The Diffusers Workflow REPL uses a hierarchical command structure for better organization. Commands are grouped by function, and each group supports `?` to show help.

## Command Groups

### Getting Help

```bash
help                    # Show all command groups and examples
?                       # Same as help
<command> ?             # Show subcommands for a specific command group
help <command>          # Traditional help for any command
```

## Workflow Commands

Manage workflow files and validation.

```bash
workflow ?              # Show workflow subcommands
workflow load <file>    # Load a workflow from JSON file
workflow reload         # Reload current workflow from disk
workflow status         # Show current workflow information
```

**Examples:**
```bash
workflow load FluxDev              # Load from default directory (./examples)
workflow load examples/FluxDev.json  # Load with path
workflow status                      # Check what's loaded
workflow reload                      # Reload after editing workflow file
```

---

## Arg Commands

Manage workflow arguments (variables).

```bash
arg ?                   # Show arg subcommands
arg show                # Show available variables and current values
arg set <name>=<value>  # Set an argument value
arg clear               # Clear all argument values
```

**Examples:**
```bash
arg show                              # See what variables are available
arg set prompt="a cute cat"           # Set the prompt variable
arg set num_inference_steps=50        # Set inference steps
arg clear                             # Clear all arguments
```

---

## Model Commands

Control model execution and worker process.

```bash
model ?                 # Show model subcommands
model run               # Execute the currently loaded workflow
model restart           # Restart worker process (clears cache)
```

**Examples:**
```bash
model run               # Run workflow with current arguments
model restart           # Restart if worker is stuck or to clear GPU cache
```

---

## Memory Commands

Monitor and manage GPU memory.

```bash
memory ?                # Show memory subcommands
memory show             # Show current GPU memory usage
memory clear            # Clear GPU memory and cached models
```

**Examples:**
```bash
memory show             # Check GPU memory usage
memory clear            # Clear cached models (keeps worker running)
```

**Backward Compatible Aliases:**
- `clear` → `memory clear`

---

## Config Commands

Configure global REPL settings.

```bash
config ?                      # Show config subcommands
config show                   # Show all configuration settings
config set <name>=<value>     # Set a configuration value
```

**Available Settings:**
- `output_dir` - Directory for output files (default: `./outputs`)
- `log_level` - Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
- `workflow_dir` - Default directory for workflows (default: `./examples`)

**Examples:**
```bash
config show                           # See all settings
config set output_dir=/path/to/output # Change output directory
config set log_level=DEBUG            # Enable debug logging
config set workflow_dir=./my-workflows # Change default workflow directory
```

---

## General Commands

```bash
help                    # Show all command groups and examples
?                       # Same as help
exit                    # Exit the REPL
quit                    # Exit the REPL
```

---

## Typical Workflow Example

```bash
dw> ?                    # Show available command groups
dw> workflow load FluxDev
dw> arg show
dw> arg set prompt="a majestic mountain landscape"
dw> model run
dw> arg set prompt="a serene beach at sunset"
dw> model run
dw> memory show
dw> workflow status
dw> exit
```

## Command Hierarchy Quick Reference

```
workflow
  ├─ load <file>
  ├─ reload
  └─ status

arg
  ├─ show
  ├─ set <name>=<value>
  └─ clear

model
  ├─ run
  └─ restart

memory
  ├─ show
  └─ clear

config
  ├─ show
  └─ set <name>=<value>
```

---

## Benefits of Hierarchical Commands

1. **Better Organization** - Related commands grouped together
2. **Discoverability** - Use `?` to explore available commands at any level
3. **Extensibility** - Easy to add new subcommands without cluttering the top level
4. **Clear Purpose** - Command groups make it obvious what each command does
5. **Consistent Interface** - All command groups follow the same pattern
