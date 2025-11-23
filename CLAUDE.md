# CLAUDE.md - AI Assistant Guide for diffusers-workflow

**Last Updated:** 2025-11-23
**Version:** 0.37.0

## Project Overview

`diffusers-workflow` is a declarative workflow engine for the HuggingFace Diffusers library. It provides a JSON-based interface for executing complex AI image/video generation pipelines without writing custom Python code.

**Key Capabilities:**
- Text-to-image and text-to-video generation
- Image-to-image and image-to-video transformations
- Multi-step composable workflows
- Variable substitution and command-line execution
- Interactive REPL with GPU model persistence
- Built-in tasks: image processing, QR codes, background removal, upscaling
- LLM-based prompt augmentation and image description
- Support for LoRA adapters, ControlNet, quantization (4-bit/8-bit)

**Technology Stack:**
- Python 3.10+ with PyTorch 2.0+
- HuggingFace Diffusers, Transformers, Accelerate
- CUDA required for GPU acceleration
- Model quantization: bitsandbytes, torchao, optimum-quanto

---

## Repository Structure

```
diffusers-workflow/
├── dw/                          # Main package
│   ├── __init__.py             # Version, startup, torch config
│   ├── workflow.py             # Workflow orchestrator (core)
│   ├── step.py                 # Step execution engine
│   ├── run.py                  # CLI entry point
│   ├── validate.py             # Schema validation CLI
│   ├── repl.py                 # Interactive REPL (19KB)
│   ├── worker.py               # REPL worker process (13KB)
│   ├── arguments.py            # Argument processing & resource loading
│   ├── variables.py            # Variable substitution system
│   ├── result.py               # Result storage & file I/O
│   ├── previous_results.py     # Cross-step data flow & cartesian products
│   ├── schema.py               # JSON schema validation
│   ├── security.py             # Security validation (10KB, critical)
│   ├── settings.py             # User settings management
│   ├── type_helpers.py         # Dynamic type loading utilities
│   ├── log_setup.py            # Logging configuration
│   ├── test.py                 # Basic startup test
│   │
│   ├── pipeline_processors/    # Pipeline management
│   │   ├── pipeline.py         # Pipeline loading & configuration
│   │   └── quantization.py     # Quantization configs
│   │
│   ├── tasks/                  # Utility tasks
│   │   ├── task.py             # Task dispatcher
│   │   ├── image_utils.py      # Image processing (12KB)
│   │   ├── video_utils.py      # Video processing
│   │   ├── gather.py           # Resource gathering
│   │   ├── qr_code.py          # QR code generation
│   │   ├── background_remover.py
│   │   ├── depth_estimator.py
│   │   ├── borders.py          # Image border operations
│   │   ├── zoe_depth.py        # Depth estimation
│   │   └── format_messages.py  # Message formatting
│   │
│   ├── community_pipelines/    # Custom pipeline implementations
│   │   └── pipeline_flux_rf_inversion.py
│   │
│   ├── workflows/              # Built-in workflows
│   │   ├── augment_prompt.json
│   │   ├── describe_image.json
│   │   └── test.json
│   │
│   └── workflow_schema.json    # JSON schema definition (18KB)
│
├── examples/                    # Example workflows (50+ examples)
│   ├── FluxDev.json            # Basic Flux example
│   ├── FluxLora.json           # LoRA adapter example
│   ├── sd35.json               # Stable Diffusion 3.5
│   ├── controlnet.json         # ControlNet example
│   ├── img2vid.json            # Image-to-video
│   ├── txt2img2vid.json        # Multi-step example
│   ├── projects/               # Complex multi-file projects
│   │   ├── A Very Marmot Christmas/
│   │   └── looper/
│   └── ... (many more)
│
├── tests/                       # Comprehensive test suite
│   ├── test_workflow.py        # Workflow execution tests
│   ├── test_security.py        # Security validation tests
│   ├── test_variables.py       # Variable handling tests
│   ├── test_previous_results.py
│   ├── test_result.py
│   ├── test_arguments.py
│   ├── test_step.py
│   ├── test_gather.py
│   ├── test_type_helpers.py
│   ├── test_integration.py     # End-to-end tests
│   ├── test_task.py
│   ├── test_schema.py
│   ├── test_examples.py
│   ├── conftest.py             # Pytest fixtures
│   ├── run_tests.py            # Test runner
│   └── test_data/              # Test fixtures
│
├── docs/                        # Documentation
│   ├── SECURITY.md             # Security implementation guide
│   ├── TESTING.md              # Testing quick start
│   ├── REPL_WORKER_GUIDE.md    # REPL documentation
│   ├── DEPENDENCIES.md         # Dependency information
│   ├── AUDIT_REPORT.md         # Security audit report
│   └── ... (more docs)
│
├── .github/
│   └── copilot-instructions.md # AI coding instructions
│
├── README.md                    # User-facing documentation
├── setup.py                     # Package configuration
├── requirements.txt             # Dependencies
├── requirements-test.txt        # Test dependencies
├── install.sh / install.ps1    # Installation scripts
└── pytest.ini                   # Pytest configuration
```

---

## Core Architecture

### Component Hierarchy

```
Workflow (workflow.py)
  ├─ Variables (variables.py) → Variable substitution & validation
  ├─ Steps (step.py) → Sequential execution
  │   ├─ Pipeline (pipeline_processors/pipeline.py)
  │   │   ├─ Component Loading (transformers, VAE, etc.)
  │   │   ├─ Quantization (quantization.py)
  │   │   ├─ Schedulers & Adapters
  │   │   └─ Execution
  │   │
  │   ├─ Task (tasks/task.py)
  │   │   ├─ Image Processing (image_utils.py)
  │   │   ├─ Video Processing (video_utils.py)
  │   │   ├─ Gathering (gather.py)
  │   │   └─ Other utilities
  │   │
  │   └─ Sub-Workflow (recursive)
  │
  ├─ Previous Results (previous_results.py) → Cross-step data flow
  ├─ Arguments (arguments.py) → Resource loading & arg processing
  └─ Results (result.py) → File I/O & storage
```

### Data Flow

1. **Load Workflow**: `workflow_from_file()` loads and validates JSON against schema
2. **Variable Processing**: `replace_variables()` substitutes `variable:name` references
3. **Step Execution**: Each step runs sequentially
   - `get_iterations()` generates argument combinations from previous results
   - Creates cartesian product for multi-value references
4. **Action Execution**: Pipeline/Task/Sub-workflow runs with arguments
5. **Result Storage**: Outputs saved to `{output_dir}/{workflow_id}-{step_name}.{index}.{ext}`

### Key Modules

#### workflow.py (10KB, Core Orchestrator)
- `workflow_from_file(file_spec, output_dir)`: Loads workflow with security validation
- `Workflow.validate()`: Schema validation
- `Workflow.run(arguments, previous_pipelines)`: Main execution loop
- Manages variable substitution, step sequencing, result accumulation

#### step.py (3KB, Step Executor)
- `Step.run(previous_results, previous_pipelines, step_action)`: Executes single step
- Generates all argument iterations via `get_iterations()`
- Handles Pipeline/Task/Sub-workflow dispatch

#### pipeline_processors/pipeline.py (Pipeline Manager)
- `Pipeline.load(shared_components)`: Loads HuggingFace pipeline
- `Pipeline.run(arguments, previous_pipelines)`: Executes inference
- Component management: transformers, VAE, ControlNet, text encoders
- Adapter loading: LoRA, IP-Adapter
- Scheduler configuration
- Memory offloading strategies

#### tasks/task.py (Task Dispatcher)
- Command router for utility tasks
- Supported commands:
  - `qr_code`: QR code generation
  - `process_image`: Resize, crop, rotate, blur, etc.
  - `process_video`: Video operations
  - `gather_images/videos/inputs`: Collect resources
  - `batch_decode_post_process`: Text decoding
  - `get_dict_value`: Extract dictionary values

#### previous_results.py (Cross-Step Data Flow)
- `get_iterations(template, previous_results)`: Cartesian product generation
- Resolves `previous_result:step_name` references
- Handles multi-value expansion

#### security.py (10KB, Security Layer)
- **Path validation**: `validate_path()`, `validate_workflow_path()`, `validate_output_path()`
- **Input validation**: `validate_variable_name()`, `validate_string_input()`
- **Command sanitization**: `sanitize_command_args()`
- **URL validation**: `validate_url()`
- **File size limits**: `validate_json_size()` (50MB max)
- Exception hierarchy: `SecurityError`, `PathTraversalError`, `InvalidInputError`

---

## REPL Worker Architecture (NEW)

The REPL uses a **persistent worker subprocess** for GPU model caching:

**Key Features:**
- Models stay loaded in GPU memory between runs (2-4x faster)
- Automatic workflow file change detection (SHA256 hash)
- Aggressive memory cleanup: `gc.collect() + torch.cuda.empty_cache()`
- Full cleanup on workflow change or `clear` command
- 5-minute execution timeout with graceful shutdown
- Memory monitoring with growth warnings (>500MB increase)

**Architecture:**
- `dw/worker.py` (13KB): Worker process with command loop
- `dw/repl.py` (19KB): REPL interface with worker lifecycle
- Communication via `multiprocessing.Queue` (command_queue, result_queue)

**Worker Commands:**
- `execute`: Run workflow with arguments
- `shutdown`: Graceful termination
- `ping`: Health check
- `clear_memory`: Force memory cleanup
- `memory_status`: Report GPU memory usage

**REPL Commands:**
```bash
load <workflow>         # Load workflow file
arg <name>=<value>      # Set argument
run                     # Execute workflow
clear                   # Clear memory cache
status                  # Show worker status
restart                 # Restart worker process
history                 # Show command history
help                    # Show help
exit/quit               # Exit REPL
```

---

## Development Workflows

### Installation

```bash
# Clone repository
git clone https://github.com/dkackman/diffusers-workflow.git
cd diffusers-workflow

# Install (creates venv, installs dependencies)
bash ./install.sh
source ./activate

# Or on Windows
.\install.ps1
.\venv\scripts\activate

# Verify installation
python -m dw.test
```

### Running Workflows

```bash
# Basic execution
python -m dw.run examples/FluxDev.json

# With variables
python -m dw.run examples/FluxDev.json prompt="a cat" num_images_per_prompt=4

# Custom output directory
python -m dw.run examples/FluxDev.json -o ./my_outputs

# Validation only
python -m dw.validate examples/FluxDev.json
```

### Interactive REPL

```bash
python -m dw.repl

dw> load examples/FluxDev
dw> arg prompt="a beautiful sunset"
dw> run
# ... models load once ...

dw> arg prompt="a starry night"
dw> run
# ... runs 2-4x faster with cached models ...
```

### Testing

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests (134+ tests)
pytest -v

# Run specific module
pytest tests/test_security.py -v

# With coverage
pytest --cov=dw --cov-report=html

# Using test runner
python -m tests.run_tests
```

### Development Tools

```bash
# Code formatting
black dw/ tests/

# Schema validation
python -m dw.validate <workflow.json>

# Check logs
tail -f ~/.diffusers_helper/log/dw.log
```

---

## Key Conventions & Patterns

### Variable System

**Definition in JSON:**
```json
{
  "variables": {
    "prompt": "default value",
    "num_images_per_prompt": 1
  }
}
```

**Reference in workflow:**
```json
{
  "arguments": {
    "prompt": "variable:prompt",
    "num_images_per_prompt": "variable:num_images_per_prompt"
  }
}
```

**Override via CLI:**
```bash
python -m dw.run workflow.json prompt="a cat" num_images_per_prompt=4
```

**Variable Name Rules:**
- Pattern: `^[a-zA-Z_][a-zA-Z0-9_-]*$`
- Alphanumeric + underscore + hyphen only
- Must start with letter or underscore

### Result References

**Basic reference:**
```json
{
  "arguments": {
    "image": "previous_result:step_name"
  }
}
```

**Cartesian product expansion:**
If `step_name` produces 3 images, the next step runs 3 times automatically.

**Multiple references:**
```json
{
  "arguments": {
    "image": "previous_result:step1",
    "mask": "previous_result:step2"
  }
}
```
If step1 has 2 results and step2 has 3 results → 6 total iterations (2×3).

### Pipeline Configuration

**Basic structure:**
```json
{
  "pipeline": {
    "configuration": {
      "component_type": "FluxPipeline",
      "offload": "sequential"
    },
    "from_pretrained_arguments": {
      "model_name": "black-forest-labs/FLUX.1-dev",
      "torch_dtype": "torch.bfloat16"
    },
    "arguments": {
      "prompt": "variable:prompt",
      "num_inference_steps": 25,
      "guidance_scale": 3.5
    }
  }
}
```

**Memory offloading strategies:**
- `"sequential"`: Sequential CPU offloading
- `"model"`: Model CPU offloading
- Component-specific in configuration sections

**Quantization (4-bit example):**
```json
{
  "transformer": {
    "configuration": {
      "component_type": "SD3Transformer2DModel",
      "quantization_config": {
        "configuration": {
          "config_type": "BitsAndBytesConfig"
        },
        "arguments": {
          "load_in_4bit": true,
          "bnb_4bit_quant_type": "{nf4}",
          "bnb_4bit_compute_dtype": "torch.bfloat16"
        }
      }
    }
  }
}
```

**LoRA adapters:**
```json
{
  "adapters": [
    {
      "adapter_id": "model_path",
      "adapter_name": "name",
      "adapter_weight": 1.0
    }
  ]
}
```

**Scheduler configuration:**
```json
{
  "scheduler": {
    "configuration": {
      "scheduler_type": "DPMSolverMultistepScheduler"
    },
    "from_config_arguments": {
      "use_karras_sigmas": true,
      "algorithm_type": "dpmsolver++"
    }
  }
}
```

### Task Execution

**Image processing:**
```json
{
  "task": {
    "command": "process_image",
    "arguments": {
      "image": "previous_result:step1",
      "operation": "resize",
      "width": 512,
      "height": 512
    }
  }
}
```

**Gathering resources:**
```json
{
  "task": {
    "command": "gather_images",
    "inputs": [
      "path/to/image1.png",
      "path/to/image2.png",
      "previous_result:step1"
    ]
  }
}
```

### Sub-Workflows

**Built-in workflow:**
```json
{
  "workflow": {
    "path": "builtin:augment_prompt.json",
    "arguments": {
      "prompt": "variable:prompt"
    }
  }
}
```

**External workflow:**
```json
{
  "workflow": {
    "path": "relative/path/to/workflow.json",
    "arguments": {
      "input_image": "previous_result:step1"
    }
  }
}
```

Built-in workflows location: `dw/workflows/`

### File Paths

**Path resolution rules:**
- Workflow files: Relative to workflow file location
- Built-in workflows: `builtin:filename.json` → `dw/workflows/filename.json`
- Output files: `{output_dir}/{workflow_id}-{step_name}.{index}.{ext}`
- Security: Path traversal blocked (`../` not allowed)

**Image/video extensions:**
- Images: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.webp`
- Videos: `.mp4`, `.avi`, `.mkv`, `.mov`, `.webm`

### Result Content Types

**Image results:**
```json
{
  "result": {
    "content_type": "image/jpeg"  // or image/png
  }
}
```

**Video results:**
```json
{
  "result": {
    "content_type": "video/mp4",
    "file_base_name": "custom_name",
    "fps": 24
  }
}
```

**Text results:**
```json
{
  "result": {
    "content_type": "text/plain"
  }
}
```

---

## Security Considerations

**CRITICAL**: All entry points use security validation from `dw/security.py`.

### Security Rules

1. **Always validate paths:**
   - Use `validate_workflow_path()` for workflow files
   - Use `validate_output_path()` for output directories
   - Use `validate_path()` for general paths

2. **Always sanitize inputs:**
   - Use `validate_variable_name()` for variable names
   - Use `validate_string_input()` for text inputs
   - Use `validate_url()` for URLs

3. **Never use dangerous patterns:**
   - ❌ `eval()`, `exec()`
   - ❌ `shell=True` in subprocess
   - ❌ Path traversal (`../`)
   - ❌ Unsanitized command arguments

4. **File size limits:**
   - JSON files: 50MB max
   - Validated via `validate_json_size()`

5. **URL restrictions:**
   - Only `http://` and `https://` allowed
   - `file://` and other schemes blocked

### Security Integration Points

**workflow.py:**
```python
validated_path = validate_workflow_path(file_spec)
validate_json_size(validated_path)
validated_output = validate_output_path(output_dir, None)
```

**run.py:**
```python
validated_name = validate_variable_name(name.strip())
validated_value = validate_string_input(value.strip(), max_length=10000)
```

**repl.py:**
```python
safe_args = sanitize_command_args(args)
subprocess.Popen(safe_args, shell=False)  # Never shell=True
```

### Exception Handling

```python
try:
    validated_path = validate_workflow_path(file_spec)
except SecurityError as e:
    logger.error(f"Security validation failed: {e}")
    raise
except PathTraversalError as e:
    logger.error(f"Path traversal detected: {e}")
    raise
except InvalidInputError as e:
    logger.error(f"Invalid input: {e}")
    raise
```

---

## Testing Strategy

**134+ tests across 13 test files**

### Test Organization

| File | Purpose | Count |
|------|---------|-------|
| test_security.py | Security validation | 10 |
| test_variables.py | Variable handling | 12 |
| test_workflow.py | Workflow execution | 8 |
| test_previous_results.py | Result references | 15 |
| test_result.py | Result storage | 15 |
| test_arguments.py | Argument processing | 15 |
| test_step.py | Step execution | 7 |
| test_gather.py | Resource gathering | 15 |
| test_type_helpers.py | Type loading | 12 |
| test_integration.py | End-to-end tests | 12 |
| test_task.py | Task execution | 7 |
| test_schema.py | Schema validation | 5 |
| test_examples.py | Example validation | - |

### Key Test Patterns

**Mocking pipelines:**
```python
class MockPipeline:
    def __call__(self, **kwargs):
        return [PIL.Image.new('RGB', (512, 512))]
```

**Mocking resources:**
```python
@pytest.fixture
def mock_image():
    return PIL.Image.new('RGB', (512, 512))
```

**Security testing:**
```python
with pytest.raises(PathTraversalError):
    validate_path("../../etc/passwd")
```

### Running Tests

```bash
# All tests
pytest -v

# Specific module
pytest tests/test_security.py -v

# Pattern matching
pytest -k "security" -v

# Coverage
pytest --cov=dw --cov-report=html

# Parallel execution
pytest -n auto

# Stop on first failure
pytest -x

# Verbose with local variables
pytest -vv -l
```

---

## Common Tasks & Examples

### Adding a New Task

1. **Add command handler in `dw/tasks/task.py`:**

```python
def run(self, arguments, previous_pipelines={}):
    # ... existing commands ...

    elif self.command == "my_new_task":
        logger.debug("Running my new task")
        return my_task_function(**arguments)
```

2. **Implement task function:**

```python
def my_task_function(input_data, param1, param2):
    # Task implementation
    return result
```

3. **Update workflow schema if needed** (`dw/workflow_schema.json`)

4. **Add tests** (`tests/test_task.py`)

### Adding a New Pipeline Type

1. **Update schema** (`dw/workflow_schema.json`):

```json
{
  "enum": [
    "FluxPipeline",
    "MyNewPipeline"
  ]
}
```

2. **Ensure proper component loading** in `pipeline.py`:

```python
optional_component_names = [
    "controlnet",
    "my_new_component",  # Add if needed
    # ...
]
```

3. **Test with example workflow**

### Creating a Multi-Step Workflow

Example: Image generation → Upscale → Video

```json
{
  "id": "img_upscale_vid",
  "steps": [
    {
      "name": "generate",
      "pipeline": {
        "configuration": {"component_type": "FluxPipeline"},
        "from_pretrained_arguments": {"model_name": "..."},
        "arguments": {"prompt": "a cat"}
      },
      "result": {"content_type": "image/png"}
    },
    {
      "name": "upscale",
      "task": {
        "command": "process_image",
        "arguments": {
          "image": "previous_result:generate",
          "operation": "upscale",
          "scale": 2
        }
      },
      "result": {"content_type": "image/png"}
    },
    {
      "name": "animate",
      "pipeline": {
        "configuration": {"component_type": "CogVideoXImageToVideoPipeline"},
        "from_pretrained_arguments": {"model_name": "THUDM/CogVideoX-5b-I2V"},
        "arguments": {
          "image": "previous_result:upscale",
          "prompt": "The cat walks forward",
          "num_frames": 49
        }
      },
      "result": {"content_type": "video/mp4"}
    }
  ]
}
```

### Using Component Sharing

Share components between pipelines to save memory:

```json
{
  "steps": [
    {
      "pipeline": {
        "configuration": {
          "shared_components": ["vae", "text_encoder"]
        }
        // ... pipeline 1 config
      }
    },
    {
      "pipeline": {
        "configuration": {
          "component_type": "SamePipelineType"
        },
        "reused_components": ["vae", "text_encoder"]
        // ... pipeline 2 config
      }
    }
  ]
}
```

---

## Critical Gotchas & Important Notes

### Schema Validation vs Variable Substitution

**Problem**: Schema validation happens BEFORE variable substitution.

**Impact**: Variables must match expected JSON types in schema.

**Example:**
```json
// ❌ WRONG - schema expects number, gets string
{
  "variables": {"steps": "25"},
  "arguments": {"num_inference_steps": "variable:steps"}
}

// ✅ CORRECT - schema gets number
{
  "variables": {"steps": 25},
  "arguments": {"num_inference_steps": "variable:steps"}
}
```

### Cartesian Product Explosion

**Problem**: Multiple `previous_result` references create exponential combinations.

**Example:**
- Step1 produces 4 images
- Step2 produces 3 masks
- Step3 references both → 12 iterations (4×3)

**Mitigation**: Be intentional about multi-value results.

### Pipeline Component Sharing

**Requirement**: Exact key matching in `reused_components` and `shared_components`.

**Example:**
```json
// Must match exactly
"shared_components": ["vae", "text_encoder"]
"reused_components": ["vae", "text_encoder"]
```

### Built-in Workflow Variable Scope

**Behavior**: Built-in workflows inherit parent variables but need explicit argument mapping.

**Example:**
```json
{
  "variables": {"prompt": "a cat"},
  "workflow": {
    "path": "builtin:augment_prompt.json",
    "arguments": {
      "prompt": "variable:prompt"  // Explicit mapping required
    }
  }
}
```

### Variable Naming Restrictions

**Pattern**: `^[a-zA-Z_][a-zA-Z0-9_-]*$`

**Valid:**
- `prompt`
- `num_steps`
- `my_variable_123`
- `image-path`

**Invalid:**
- `123var` (starts with number)
- `my.var` (contains `.`)
- `var$name` (contains `$`)

### Path Security

**Blocked patterns:**
- `../` (path traversal)
- `~/` (home directory expansion in some contexts)
- `/dev/`, `/proc/`, `/sys/` (system directories)
- Absolute paths outside allowed directories

**Safe patterns:**
- `examples/FluxDev.json` (relative)
- `builtin:augment_prompt.json` (built-in)
- `./outputs` (explicit relative)

### Memory Management in REPL

**Cache clearing triggers:**
- Workflow file change (SHA256 hash comparison)
- `clear` command
- Worker restart

**Manual cleanup:**
```python
dw> clear  # Force memory cleanup
dw> restart  # Restart worker process
```

### Type Helpers

**Dynamic type loading** uses `type_helpers.py`:

```python
# String → Type conversion
"torch.bfloat16" → torch.bfloat16
"FluxPipeline" → FluxPipeline class
```

**In JSON:**
```json
{
  "torch_dtype": "torch.bfloat16",  // String that gets converted
  "component_type": "FluxPipeline"
}
```

### Logging Configuration

**Default**: `WARNING` level to `~/.diffusers_helper/log/dw.log`

**Override:**
```bash
python -m dw.run workflow.json -l DEBUG
```

**Settings file**: `~/.diffusers_helper/settings.json`
```json
{
  "log_level": "DEBUG",
  "log_filename": "log/dw.log",
  "log_to_console": true
}
```

### CUDA Requirements

**Startup check** in `dw/__init__.py`:
```python
if not torch.cuda.is_available():
    raise Exception("CUDA not present. Quitting.")
```

**Torch version check**:
```python
if version.parse(torch.__version__) < version.parse("2.0.0"):
    raise Exception("Pytorch must be 2.0 or greater")
```

---

## Integration Points

### HuggingFace Ecosystem

- **Diffusers**: Direct pipeline instantiation via `component_type`
- **Transformers**: LLM-based prompt augmentation, text generation
- **Accelerate**: Model offloading and optimization
- **PEFT**: LoRA and adapter support
- **Safetensors**: Safe model loading

### External Libraries

- **ControlNet Aux**: Preprocessing for ControlNet
- **OpenCV**: Image processing
- **MoviePy**: Video processing
- **QRCode**: QR code generation
- **Pillow**: Image I/O
- **MediaPipe**: Advanced image processing

### Model Formats

- **Standard PyTorch**: `.pt`, `.pth`
- **Safetensors**: `.safetensors`
- **GGUF**: Quantized models (`.gguf`)
- **4-bit/8-bit**: bitsandbytes quantization

---

## Quick Reference Commands

### CLI Commands

```bash
# Run workflow
python -m dw.run <workflow.json> [var=value ...]

# Validate workflow
python -m dw.validate <workflow.json>

# Start REPL
python -m dw.repl

# Run tests
pytest -v
python -m tests.run_tests

# Basic system test
python -m dw.test
```

### REPL Commands

```
load <workflow>         # Load workflow file
arg <name>=<value>      # Set workflow argument
args                    # Show current arguments
run                     # Execute workflow
clear                   # Clear memory cache
status                  # Show worker status
restart                 # Restart worker
history                 # Show command history
help                    # Show help
exit / quit             # Exit REPL
```

### Common File Patterns

```bash
# Example workflows
examples/*.json

# Built-in workflows
dw/workflows/*.json

# Test data
tests/test_data/workflows/*.json

# Output files
outputs/{workflow_id}-{step_name}.{index}.{ext}

# Logs
~/.diffusers_helper/log/dw.log

# Settings
~/.diffusers_helper/settings.json
```

---

## Resources & Documentation

### Internal Documentation

- `README.md`: User guide and examples
- `docs/SECURITY.md`: Security implementation
- `docs/TESTING.md`: Testing guide
- `docs/REPL_WORKER_GUIDE.md`: REPL documentation
- `docs/DEPENDENCIES.md`: Dependency information
- `.github/copilot-instructions.md`: AI coding instructions

### External Resources

- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Project Wiki](https://github.com/dkackman/diffusers-workflow/wiki)
- [GitHub Repository](https://github.com/dkackman/diffusers-workflow)

### Schema Reference

- JSON Schema: `dw/workflow_schema.json`
- [Schema Viewer](https://json-schema.app/view/%23?url=https%3A%2F%2Fraw.githubusercontent.com%2Fdkackman%2Fdiffusers-workflow%2Frefs%2Fheads%2Fmaster%2Fdw%2Fworkflow_schema.json)

---

## Development Principles

1. **Security First**: Always validate inputs, paths, and commands
2. **No Shell Execution**: Never use `shell=True` in subprocess calls
3. **Deep Copy Workflow Definitions**: Avoid mutation for multi-run support
4. **Comprehensive Testing**: Test security, functionality, integration
5. **Clear Logging**: Use structured logging with appropriate levels
6. **Schema Validation**: Validate all workflows against JSON schema
7. **Memory Management**: Clean up GPU memory aggressively in REPL
8. **Type Safety**: Use type helpers for dynamic type loading
9. **Error Handling**: Graceful degradation with informative errors
10. **Documentation**: Keep docs updated with code changes

---

## Version Information

- **Current Version**: 0.37.0
- **Python Requirements**: 3.10+
- **PyTorch Requirements**: 2.0+
- **CUDA**: Required
- **License**: Apache 2.0

---

**Last Updated**: 2025-11-23
**Maintainer**: dkackman
**Repository**: https://github.com/dkackman/diffusers-workflow
