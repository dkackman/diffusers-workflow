"""
Trust gate for workflow files: validation of classes, imports, and arguments
an untrusted (default) workflow may name without --trust-workflows.
"""

import os
import re
from .security import UntrustedWorkflowError


# The environment variable a parent process sets to hand its --trust-workflows
# choice down to a spawned worker subprocess - multiprocessing's 'spawn' start
# method launches a fresh interpreter that inherits os.environ, the same way
# DW_PROMPT_DIR reaches the worker (see dw/serve.py)
TRUST_WORKFLOWS_ENV_VAR = "DW_TRUST_WORKFLOWS"

# Dotted type/pre_load_modules references that resolve under one of these
# top-level packages are treated as part of the diffusers ecosystem the tool
# already assumes, and are allowed even for an untrusted workflow - everything
# else requires --trust-workflows. This is the dependency set pyproject.toml
# declares for exactly this purpose ("Quantization backends - config_objects.py
# loads them dynamically"), plus the framework packages workflows target
# directly, plus 'dw' itself - a workflow's own community_pipelines component
# lives in this package, not a third party one
TRUSTED_TOP_LEVEL_PACKAGES = (
    "diffusers",
    "torch",
    "torchvision",
    "transformers",
    "accelerate",
    "peft",
    "sdnq",
    "torchao",
    "optimum",  # optimum-quanto
    "gguf",
    "bitsandbytes",
    "dw",
)


# What an untrusted workflow may name as a class. The package allowlist above
# is necessary but not sufficient: every '*_type' value is a class the run
# constructs or calls from_pretrained on with the workflow's own arguments,
# and a class defined inside an allowed package can still do anything in its
# constructor - start a process, open a file for writing. So a class reached
# untrusted must also be one of the kinds below, each of which is a
# checkpoint-loading component or plain data. Each base is (module, name) and
# is matched by identity through sys.modules: a class can only subclass a
# base whose module is already imported, so nothing is imported to check,
# and a package that is not installed simply contributes no bases.
CONSTRUCTIBLE_BASE_CLASSES = (
    # diffusers: models, pipelines, schedulers, quantization configs
    ("diffusers.models.modeling_utils", "ModelMixin"),
    ("diffusers.pipelines.pipeline_utils", "DiffusionPipeline"),
    ("diffusers.modular_pipelines.modular_pipeline", "ModularPipeline"),
    ("diffusers.schedulers.scheduling_utils", "SchedulerMixin"),
    ("diffusers.quantizers.quantization_config", "QuantizationConfigMixin"),
    # transformers: models, tokenizers, processors, quantization configs
    ("transformers.modeling_utils", "PreTrainedModel"),
    ("transformers.tokenization_utils_base", "PreTrainedTokenizerBase"),
    ("transformers.processing_utils", "ProcessorMixin"),
    ("transformers.feature_extraction_utils", "FeatureExtractionMixin"),
    ("transformers.image_processing_base", "ImageProcessingMixin"),
    ("transformers.utils.quantization_config", "QuantizationConfigMixin"),
    # torchao quantization configs (quant_type); sdnq's SDNQConfig is a
    # diffusers QuantizationConfigMixin and is accepted through that
    ("torchao.core.config", "AOBaseConfig"),
)

# Modules whose classes are the libraries' auto-dispatch factories
# (AutoPipelineFor*, transformers' Auto*): they have no base in common with
# what they dispatch to, construct nothing themselves, and load only through
# from_pretrained - whose remote-code arguments are gated separately
# (require_trusted_from_pretrained_arguments)
CONSTRUCTIBLE_FACTORY_MODULES = (
    "diffusers.pipelines.auto_pipeline",
    "transformers.models.auto.",
)

# Reference and condition descriptions (MiniMaxH3ImageReference,
# LTX2VideoCondition, ...) are dataclasses with no base of their own. A
# dataclass's generated __init__ only assigns its fields, so one defined in
# these packages is plain data - unless it adds a __post_init__, which runs
# arbitrary code; those are accepted only once reviewed and listed below
CONSTRUCTIBLE_DATACLASS_PACKAGES = (
    "diffusers.pipelines.",
    "diffusers.modular_pipelines.",
)
REVIEWED_POST_INIT_DATACLASSES = (
    # __post_init__ only defaults fps to a module constant
    "diffusers.modular_pipelines.minimax_h3.references.MiniMaxH3VideoReference",
)

# attn_processor_type names an attention processor, constructed with no
# arguments. They share no base class; diffusers defines them in its models
# package, named '...Processor' (or '...Processor2_0'). Their constructors
# record configuration, except that one (LTX2VideoVaeNeighborhoodNattenProcessor)
# fetches a kernel from the Hub - from a repo diffusers hardcodes, never one
# its arguments name, and only when DIFFUSERS_DISABLE_REMOTE_CODE is unset
# (see dw/kernel_availability.py)
_ATTENTION_PROCESSOR_NAME = re.compile(r"Processor(\d+_\d+)?$")
_ATTENTION_PROCESSOR_PACKAGE = "diffusers.models."


def _constructible_bases():
    import sys

    bases = []
    for module_name, class_name in CONSTRUCTIBLE_BASE_CLASSES:
        module = sys.modules.get(module_name)
        base = getattr(module, class_name, None) if module is not None else None
        if isinstance(base, type):
            bases.append(base)
    return tuple(bases)


def is_constructible_class(cls) -> bool:
    """Whether an untrusted workflow may name `cls` as a type to construct.

    True for a subclass of CONSTRUCTIBLE_BASE_CLASSES, a class defined in
    CONSTRUCTIBLE_FACTORY_MODULES, a dataclass defined in
    CONSTRUCTIBLE_DATACLASS_PACKAGES with no unreviewed __post_init__, or a
    diffusers attention processor. See the comments on each for why.
    """
    import dataclasses

    if not isinstance(cls, type):
        return False
    # Real inheritance only: issubclass would also honour an ABC's register()
    # and __subclasshook__, and AOBaseConfig is an ABC
    bases = _constructible_bases()
    if any(base in cls.__mro__ for base in bases):
        return True
    module = getattr(cls, "__module__", None)
    if not isinstance(module, str):
        return False
    qualified = f"{module}.{cls.__qualname__}"
    if any(
        module == prefix.rstrip(".") or module.startswith(prefix)
        for prefix in CONSTRUCTIBLE_FACTORY_MODULES
    ):
        return True
    if dataclasses.is_dataclass(cls) and module.startswith(
        CONSTRUCTIBLE_DATACLASS_PACKAGES
    ):
        return (
            getattr(cls, "__post_init__", None) is None
            or qualified in REVIEWED_POST_INIT_DATACLASSES
        )
    return module.startswith(_ATTENTION_PROCESSOR_PACKAGE) and bool(
        _ATTENTION_PROCESSOR_NAME.search(cls.__name__)
    )


def require_constructible_class(name: str, cls, what: str) -> None:
    """Refuse a class an untrusted workflow may not construct.

    Raises:
        UntrustedWorkflowError: If untrusted and `cls` is not
            is_constructible_class
    """
    if workflows_are_trusted() or is_constructible_class(cls):
        return
    raise UntrustedWorkflowError(
        f"Refusing to load {what} '{name}': an untrusted workflow may only "
        f"name a model, pipeline, scheduler, tokenizer, processor, "
        f"quantization config, auto-pipeline/auto-model factory, attention "
        f"processor or reference/condition dataclass from the diffusers "
        f"ecosystem, and '{getattr(cls, '__module__', '?')}."
        f"{getattr(cls, '__qualname__', '?')}' is none of these - its "
        f"constructor could run anything with the workflow's arguments. Pass "
        f"--trust-workflows if you trust this workflow's source."
    )


def set_trust_workflows(trusted: bool) -> None:
    """Record the process-wide --trust-workflows choice.

    Called once at CLI/server startup, before any workflow loads. A spawned
    worker subprocess reads the same choice back from the environment
    variable this sets, rather than needing it passed as an argument.
    """
    os.environ[TRUST_WORKFLOWS_ENV_VAR] = "1" if trusted else "0"


def workflows_are_trusted() -> bool:
    """Whether the process has been told to trust workflow files fully.

    Defaults to untrusted (False) when nothing has set the flag, which is
    the secure default for any code path that loads a workflow without
    going through the CLI/server startup that calls set_trust_workflows().
    """
    return os.environ.get(TRUST_WORKFLOWS_ENV_VAR) == "1"


def _top_level_package(dotted_name: str) -> str:
    return dotted_name.split(".", 1)[0]


def require_trusted_dotted_name(dotted_name: str, what: str) -> None:
    """Refuse a dotted-name import outside the diffusers ecosystem unless
    the workflow is trusted.

    Args:
        dotted_name: The module.path.Name a workflow supplied
        what: Short phrase naming what kind of value this was ('a *_type
            value', 'a config_type value', ...), for the error message

    Raises:
        UntrustedWorkflowError: If untrusted and the name is outside
            TRUSTED_TOP_LEVEL_PACKAGES
    """
    if workflows_are_trusted():
        return

    top_level = _top_level_package(dotted_name)
    if top_level in TRUSTED_TOP_LEVEL_PACKAGES:
        return

    raise UntrustedWorkflowError(
        f"Refusing to load {what} '{dotted_name}': it imports the "
        f"'{top_level}' module, which is outside the ecosystem "
        f"({', '.join(TRUSTED_TOP_LEVEL_PACKAGES)}) this workflow is "
        f"allowed to reach untrusted. Loading a workflow JSON file can "
        f"execute arbitrary Python - see docs/SECURITY.md. Pass "
        f"--trust-workflows if you trust this workflow's source."
    )


def require_trusted_pre_load_modules(module_names) -> None:
    """Refuse a pre_load_modules entry outside the diffusers ecosystem
    unless the workflow is trusted.

    pre_load_modules exists to run a module's import-time registration
    side effects - sdnq registering its quantization method with diffusers
    is the pattern the bundled example workflows use - so an in-ecosystem
    module name is allowed the same way an in-ecosystem dotted type
    reference is; anything else requires trust.

    Raises:
        UntrustedWorkflowError: If untrusted and any name is outside
            TRUSTED_TOP_LEVEL_PACKAGES
    """
    for module_name in module_names or []:
        require_trusted_dotted_name(module_name, "pre_load_modules entry")


REMOTE_CODE_ARGUMENTS = ("trust_remote_code", "custom_pipeline")


def require_trusted_from_pretrained_arguments(arguments, what: str) -> None:
    """Refuse from_pretrained arguments that make diffusers/transformers
    download and execute Python from the Hub unless the workflow is trusted.

    `trust_remote_code: true` runs a repo's own modeling code; `custom_pipeline`
    fetches a pipeline module from the Hub (or a local path) and imports it.
    Both are arbitrary code chosen by the workflow file, reached without any
    importlib call of ours - so the importlib gate alone would leave them open.

    Args:
        arguments: The from_pretrained_arguments block
        what: The component being loaded, for the error message

    Raises:
        UntrustedWorkflowError: If untrusted and either argument is set
    """
    if workflows_are_trusted() or not arguments:
        return
    for key in REMOTE_CODE_ARGUMENTS:
        if arguments.get(key):
            raise UntrustedWorkflowError(
                f"Refusing to load {what}: its from_pretrained_arguments set "
                f"'{key}', which downloads and executes Python from the model "
                f"repository. Loading a workflow JSON file can execute arbitrary "
                f"Python - see docs/SECURITY.md. Pass --trust-workflows if you "
                f"trust this workflow's source."
            )
