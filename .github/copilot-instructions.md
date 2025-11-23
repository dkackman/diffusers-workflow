# AI Coding Instructions for diffusers-workflow

This is a declarative workflow engine for the HuggingFace Diffusers library that executes AI model pipelines via JSON configuration files.

## Architecture Overview

**Core Components:**
- `dw/workflow.py`: Main orchestrator - loads JSON workflows, handles variable substitution, manages step execution
- `dw/step.py`: Individual workflow step executor - runs pipelines/tasks/sub-workflows 
- `dw/pipeline_processors/pipeline.py`: Manages HuggingFace pipeline loading, configuration, and shared components
- `dw/tasks/task.py`: Executes utility tasks (image processing, QR codes, data gathering)
- `dw/previous_results.py`: Handles cross-step data flow using cartesian products of previous results

**Key Data Flow:**
1. JSON workflow loaded → validated against `workflow_schema.json`
2. Variables processed (`variable:name` → actual values)
3. Steps executed sequentially, each can reference `previous_result:step_name`
4. Results saved to output directory with naming pattern `{workflow_id}-{step_name}.{index}`

## Critical Patterns

**Variable System:** Use `"variable:name"` in JSON to reference workflow variables. Variables can be set via command line (`prompt="a cat"`) or in workflow definition.

**Result References:** Use `"previous_result:step_name"` to pipe outputs between steps. System automatically generates all combinations when multiple results exist.

**Pipeline Configuration:**
```json
{
  "pipeline": {
    "configuration": {"component_type": "FluxPipeline", "offload": "sequential"},
    "from_pretrained_arguments": {"model_name": "black-forest-labs/FLUX.1-dev"},
    "arguments": {"prompt": "variable:prompt", "num_inference_steps": 25}
  }
}
```

**Task Execution:**
```json
{
  "task": {
    "command": "process_image",
    "arguments": {"image": "previous_result:step1", "operation": "resize"}
  }
}
```

## Development Workflows

**Testing:** Run `python -m dw.test` for basic validation or `pytest -v` for full test suite
**Validation:** Use `python -m dw.validate workflow.json` to check schema compliance
**Execution:** `python -m dw.run workflow.json variable1=value1`

**Adding New Tasks:** Extend `dw/tasks/task.py.run()` method with new command handlers
**Adding Pipeline Types:** Update `workflow_schema.json` and ensure proper component loading in `pipeline.py`

## Project-Specific Conventions

- All file paths in workflows are relative to the workflow file location
- Built-in workflows use `"builtin:filename.json"` and live in `dw/workflows/`
- Model offloading patterns: `"sequential"`, `"model"`, or component-specific configurations
- Quantization configs follow BitsAndBytesConfig pattern for 4-bit/8-bit loading
- Image results default to JPEG, videos to MP4 unless overridden in `result.content_type`

## Key Integration Points

**HuggingFace Diffusers:** Direct pipeline instantiation via `component_type` field
**Transformers:** Used for LLM-based prompt augmentation and text generation tasks  
**External Models:** Support for ControlNet, LoRA adapters, custom community pipelines
**Device Management:** Automatic CUDA detection with fallback, memory optimization via offloading

## Security

**Critical security module** (`dw/security.py`) provides comprehensive input validation and protection:
- **Path validation**: Prevents traversal attacks, validates file extensions, enforces directory restrictions
- **Input sanitization**: Validates variable names (alphanumeric + underscore/hyphen only), string lengths, control characters
- **Command safety**: Sanitizes subprocess arguments, blocks shell metacharacters, enforces `shell=False`
- **URL validation**: Restricts to http/https schemes only

All entry points (run.py, validate.py, repl.py) use security validation. When adding features:
- Always validate paths with `validate_path()` or `validate_workflow_path()`
- Use `validate_variable_name()` for user-provided variable names
- Sanitize URLs with `validate_url()` before remote loading
- Use `sanitize_command_args()` before subprocess calls
- Never use `eval()`, `exec()`, or `shell=True`

## Common Gotchas

- Schema validation happens before variable substitution - use exact JSON types in schema
- `previous_result` generates cartesian products - be aware of exponential combinations
- Pipeline component sharing requires exact key matching in `reused_components`
- Built-in workflows inherit parent variable scope but need explicit argument mapping
- Variable names must match `^[a-zA-Z_][a-zA-Z0-9_-]*$` pattern for security
- File paths with `../` are blocked - use absolute or safe relative paths only