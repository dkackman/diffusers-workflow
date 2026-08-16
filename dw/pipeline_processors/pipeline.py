import torch
import contextlib
import copy
import importlib
import logging
from .config_objects import (
    get_quantization_configuration,
    get_group_offload_configuration,
    get_cache_configuration,
    get_load_components_arguments,
)
from .remote import remote_text_encoder
from ..cache_blocks import register_cache_blocks
from ..teacache import teacache_context
from ..type_helpers import has_method
from .. import get_device_type
from diffusers import attention_backend

# dw.prompt_weighting (transformers) and diffusers.hooks (peft, bitsandbytes) are
# imported where they are used - at module scope they add seconds to every startup

logger = logging.getLogger("dw")

# Names the cache state a run accumulates. Any stable string works - it only has to
# match itself across the steps of one call
_CACHE_CONTEXT_NAME = "dw"

optional_component_names = [
    "controlnet",
    "transformer",
    "transformer_2",
    "vae",
    "unet",
    "text_encoder",
    "text_encoder_2",
    "text_encoder_3",
    "tokenizer",
    "tokenizer_2",
    "tokenizer_3",
    "image_encoder",
    "feature_extractor",
    "prompt_enhancer_head",
    "model",
]

# Pipeline-definition keys that can never name a component
_NON_COMPONENT_KEYS = {
    "configuration",
    "from_pretrained_arguments",
    "arguments",
    "scheduler",
    "loras",
    "ip_adapter",
    "seed",
    "remote_text_encoder",
}


def declared_component_names(pipeline_definition):
    """The component names a pipeline definition can load or configure.

    The known names plus any other key shaped like a component - a dict carrying
    'from_pretrained_arguments' (a scheduler carries 'from_config_args' instead).
    Diffusers grows new component names faster than the list above; a workflow
    naming one gets it loaded rather than silently dropped.
    """
    names = list(optional_component_names)
    for key, value in pipeline_definition.items():
        if (
            key not in names
            and key not in _NON_COMPONENT_KEYS
            and isinstance(value, dict)
            and "from_pretrained_arguments" in value
        ):
            logger.info(f"Treating '{key}' as a component definition")
            names.append(key)
    return names


class Pipeline:
    """
    Manages pipeline initialization, configuration, and execution.
    Handles loading of models, schedulers, and adapters.
    """

    def __init__(self, pipeline_definition, default_seed, device, pipeline=None):
        """
        Initialize pipeline with configuration and device settings.

        Args:
            pipeline_definition: Dictionary containing pipeline configuration
            default_seed: Seed value for reproducibility
            device: Device to run pipeline on (e.g., 'cuda', 'mps', 'cpu') - the
                configuration's own 'device' takes precedence over it
            pipeline: Optional existing pipeline to use
        """
        self.pipeline_definition = pipeline_definition
        self.default_seed = default_seed
        # A step can pin itself to a device, overriding the one dw is running on. It
        # becomes the default for this pipeline's components as well
        self.device = self.configuration.get("device", device)
        self.pipeline = pipeline
        logger.debug(f"Initialized pipeline with device: {self.device}")

    @property
    def configuration(self):
        return self.pipeline_definition.get("configuration", {})

    @property
    def name(self):
        return self.from_pretrained_arguments.get("model_name", "")

    @property
    def from_pretrained_arguments(self):
        return self.pipeline_definition.get("from_pretrained_arguments", {})

    @property
    def argument_template(self):
        return self.pipeline_definition["arguments"]

    def populate_from_pretrained_arguments(self, device, shared_components):
        """
        Prepare arguments for pipeline creation, including shared components.

        Args:
            device: Device to run pipeline on
            shared_components: Dictionary of components shared between pipelines
        """
        logger.debug("Populating from_pretrained arguments")
        from_pretrained_arguments = self.from_pretrained_arguments

        # Add shared components to arguments
        for reused_component_name in self.pipeline_definition.get(
            "reused_components", []
        ):
            logger.debug(f"Adding reused component: {reused_component_name}")
            from_pretrained_arguments[reused_component_name] = shared_components[
                reused_component_name
            ]

        # Load optional components (controlnet, vae, unet, etc.), including any
        # component-shaped key outside the known names
        for component_name in declared_component_names(self.pipeline_definition):
            self.load_optional_component(
                component_name, from_pretrained_arguments, device
            )

        # Handle remote text encoder configuration by setting local text_encoder to None
        if self.pipeline_definition.get("remote_text_encoder", None):
            logger.info("Configuring remote text encoder")
            from_pretrained_arguments["text_encoder"] = None

        return from_pretrained_arguments

    def load(self, shared_components):
        """
        Load and configure the pipeline with all components.

        Args:
            shared_components: Dictionary of components shared between pipelines
        """
        logger.debug(f"Loading pipeline: {self.name}")

        # Import modules that need to register with diffusers/transformers before loading
        # (e.g., sdnq registers its quantization method on import)
        for module_name in self.configuration.get("pre_load_modules", []):
            logger.info(f"Pre-loading module: {module_name}")
            importlib.import_module(module_name)

        # Prepare arguments and load pipeline
        from_pretrained_arguments = self.populate_from_pretrained_arguments(
            self.device, shared_components
        )

        # Load and configure the main pipeline
        self.pipeline = load_component(
            "pipeline",
            self.configuration,
            from_pretrained_arguments,
            self.device,
        )

        # Enable attention slicing if explicitly requested or automatically on MPS
        # MPS benefits from slicing since Metal shares system RAM with the GPU
        if self.configuration.get("enable_attention_slicing", False) or (
            get_device_type(self.device) == "mps"
            and not self.configuration.get("disable_attention_slicing", False)
        ):
            # Modular pipelines have no attention slicing - on MPS this is applied
            # automatically, so skip rather than fail when the pipeline lacks it
            if has_method(self.pipeline, "enable_attention_slicing"):
                logger.debug("Enabling attention slicing for pipeline")
                self.pipeline.enable_attention_slicing()
            else:
                logger.debug(
                    f"{type(self.pipeline).__name__} does not support attention slicing, skipping"
                )

        # configure components that are not shared
        self.configure_loaded_components()

        # Apply SDNQ quantized matmul optimization to specified components
        sdnq_optimize = self.configuration.get("sdnq_optimize", [])
        if sdnq_optimize:
            apply_sdnq_optimizations(self.pipeline, sdnq_optimize)

        # Enable diffusers built-in cache acceleration on transformer
        cache_config = get_cache_configuration(self.configuration)
        if cache_config is not None:
            enable_cache_on_transformer(self.pipeline, cache_config)

        # Configure scheduler if specified
        load_and_configure_scheduler(
            self.pipeline_definition.get("scheduler", None), self.pipeline
        )

        # Store components that will be shared with other pipelines
        for shared_component_name in self.pipeline_definition.get(
            "shared_components", []
        ):
            logger.debug(f"Storing shared component: {shared_component_name}")
            shared_components[shared_component_name] = getattr(
                self.pipeline, shared_component_name
            )

        # Load and configure LoRA models
        load_loras(self.pipeline_definition.get("loras", []), self.pipeline)

        # Load and configure IP-Adapter
        load_ip_adapter(self.pipeline_definition.get("ip_adapter", None), self.pipeline)

        # Place the components the pipeline loaded itself, once everything that alters
        # them - dtypes, adapters, quantized matmuls - has been applied. Offloading hooks
        # installed before those would be fighting them
        configure_components(self.pipeline, self.configuration, self.device)

        # Set up random generator if needed - no_generator is a boolean, so an
        # explicit false still gets a generator
        if not self.configuration.get("no_generator", False):
            logger.debug("Setting up random generator")
            self.argument_template["generator"] = torch.Generator(
                self.device
            ).manual_seed(self.pipeline_definition.get("seed", self.default_seed))

        logger.debug("Pipeline loaded successfully")

    @torch.inference_mode()
    def run(self, arguments, previous_pipelines={}):
        """
        Execute the pipeline with given arguments.

        Args:
            arguments: Dictionary of arguments for pipeline execution
            previous_pipelines: Dictionary of previously created pipelines

        Returns:
            Pipeline output or dictionary containing special outputs
        """
        if self.pipeline is None:
            logger.error("Pipeline not initialized")
            raise ValueError(
                "Pipeline has not been initialized. Call load(device_identifier, shared_components) first."
            )

        logger.debug(f"Running pipeline with arguments: {arguments}")

        try:
            # Handle inversion pipeline
            if self.configuration.get("inversion", False):
                logger.debug("Running inversion pipeline")
                invert_arguments = copy.deepcopy(arguments)
                invert_arguments.pop("generator", None)
                inverted_latents, image_latents, latent_image_ids = (
                    self.pipeline.invert(**invert_arguments)
                )
                return {
                    "inverted_latents": inverted_latents,
                    "image_latents": image_latents,
                    "latent_image_ids": latent_image_ids,
                }

            # Handle generation pipeline
            if self.configuration.get("generate", False):
                logger.debug("Running generation pipeline")
                return {"generated_ids": self.pipeline.generate(**arguments)}

            if self.pipeline_definition.get("remote_text_encoder", None) is not None:
                logger.info("Invoking remote text encoder")
                remote_config = self.pipeline_definition["remote_text_encoder"]
                prompt_embeds = remote_text_encoder(
                    arguments.pop("prompt"),
                    remote_config.get("url"),
                    device=self.device,
                )
                arguments["prompt_embeds"] = prompt_embeds
            elif self.configuration.get("prompt_weighting", False):
                from ..prompt_weighting import apply_prompt_weighting

                # The step's device override travels with the call - embeddings
                # must land where the transformer runs
                apply_prompt_weighting(self.pipeline, arguments, self.device)

            # Run standard pipeline
            logger.debug("Running standard pipeline")
            output = self._execute_pipeline(arguments)

            # A raw tensor result - latents, embeddings - is held for the rest of the
            # workflow, so it rests in system memory instead of occupying the
            # accelerator that the next step needs. Pipelines consuming it place it back
            # on their own device.
            if hasattr(output, "to"):
                logger.debug("Moving tensor output to system memory")
                output = output.to("cpu")

            attach_audio_sample_rate(self.pipeline, output)

            return output

        except Exception as e:
            # One log line with the full traceback - every error class was
            # logged and re-raised identically
            logger.error(f"{type(e).__name__} running pipeline: {e}", exc_info=True)
            raise

    def _execute_pipeline(self, arguments):
        """Execute the pipeline with optional TeaCache and attention backend contexts."""
        teacache_config = self.configuration.get("teacache", None)
        attn_backend = self.configuration.get("attention_backend", None)

        # Determine the execution context
        if teacache_config is not None:
            num_steps = arguments.get("num_inference_steps", None)
            if num_steps is None:
                logger.warning(
                    "TeaCache requires num_inference_steps in arguments, running without TeaCache"
                )
                return self._call_pipeline(arguments, attn_backend)

            rel_l1_thresh = teacache_config.get("rel_l1_thresh", None)
            coefficients = teacache_config.get("coefficients", None)
            variant = teacache_config.get("variant", None)
            with teacache_context(
                self.pipeline, num_steps, rel_l1_thresh, coefficients, variant
            ):
                return self._call_pipeline(arguments, attn_backend)
        else:
            return self._call_pipeline(arguments, attn_backend)

    def _call_pipeline(self, arguments, attn_backend):
        """Call the pipeline with optional attention backend and cache contexts."""
        with contextlib.ExitStack() as stack:
            if attn_backend is not None:
                logger.debug(f"Using attention backend: {attn_backend}")
                stack.enter_context(attention_backend(attn_backend))

            stack.enter_context(stateful_cache_context(self.pipeline))

            return self.pipeline(**arguments)

    def load_optional_component(
        self, component_name, from_pretrained_arguments, default_device
    ):
        """Load an optional component if specified in pipeline definition."""
        component_definition = self.pipeline_definition.get(component_name, None)

        if component_definition is not None:
            logger.info(f"Loading component: {component_name}")
            component_configuration = component_definition.get("configuration", None)
            if component_configuration is not None:
                component_from_pretrained_arguments = component_definition[
                    "from_pretrained_arguments"
                ]

                # Handle quantization configuration
                quantization_configuration = get_quantization_configuration(
                    component_definition
                )
                if quantization_configuration is not None:
                    logger.debug(f"Adding quantization config for {component_name}")
                    component_from_pretrained_arguments["quantization_config"] = (
                        quantization_configuration
                    )

                device = component_configuration.get("device", default_device)

                component = load_component(
                    component_name,
                    component_configuration,
                    component_from_pretrained_arguments,
                    device,
                )

                logger.debug(f"Loaded optional component: {component_name}")
                from_pretrained_arguments[component_name] = component

    def configure_loaded_components(self):
        # Configure VAE settings
        vae = self.configuration.get("vae", {})
        if vae.get("enable_slicing", False):
            logger.debug("Enabling VAE slicing")
            self.pipeline.vae.enable_slicing()
        if vae.get("enable_tiling", False):
            logger.debug("Enabling VAE tiling")
            self.pipeline.vae.enable_tiling()
        if vae.get("channels_last", False):
            logger.debug("Setting VAE memory format")
            self.pipeline.vae.to(memory_format=torch.channels_last)

        # Configure UNet settings
        unet = self.configuration.get("unet", {})
        if unet.get("enable_forward_chunking", False):
            logger.debug("Enabling UNet forward chunking")
            self.pipeline.unet.enable_forward_chunking()
        if unet.get("channels_last", False):
            logger.debug("Setting UNet memory format")
            self.pipeline.unet.to(memory_format=torch.channels_last)

        # Configure UNet attention processor
        if unet.get("attn_processor_type", None) is not None:
            logger.debug("Enabling UNet custom attention processor")
            attn_processor = unet["attn_processor_type"]()
            self.pipeline.unet.set_attn_processor(attn_processor)

        # Configure transformer settings
        transformer = self.configuration.get("transformer", {})
        if transformer.get("attn_processor_type", None) is not None:
            logger.debug("Enabling transformer custom attention processor")
            attn_processor = transformer["attn_processor_type"]()
            self.pipeline.transformer.set_attn_processor(attn_processor)

        # configure optional components
        for component_name in declared_component_names(self.pipeline_definition):
            component_configuration = self.configuration.get(component_name, None)
            if component_configuration is None:
                continue

            # get_component() raises on a genuinely missing attribute (a typo) and
            # returns None for one that is registered but unloaded - both cases are
            # unconfigurable, so both are skipped here exactly as a plain missing
            # component always was
            try:
                component = get_component(self.pipeline, component_name)
            except ValueError:
                component = None

            if component is not None:
                logger.debug(f"Configuring optional component: {component_name}")
                torch_dtype = component_configuration.get("torch_dtype", None)
                if torch_dtype is not None:
                    logger.debug(f"Setting {component_name} torch dtype: {torch_dtype}")
                    component.to(torch_dtype)


def configure_components(pipeline, configuration, default_device):
    """Place the components a pipeline loaded for itself.

    A modular pipeline pulls its own component weights, so they are only reachable once
    the pipeline is loaded - too late for the offloading load_component sets up. Group
    offloading a component here streams it between system memory and the accelerator a
    piece at a time, which is what fits a pipeline whose components are each larger than
    the device.

    Args:
        pipeline: The loaded pipeline
        configuration: Pipeline configuration dictionary
        default_device: Device the pipeline runs on
    """
    for component_name, component_configuration in configuration.get(
        "components", {}
    ).items():
        component = get_component(pipeline, component_name)
        if component is None:
            # Registered but unloaded (e.g. a components map reused across workflow
            # selections, or a component diffusers warned-and-skipped past at load) -
            # skip just this entry rather than aborting the whole run
            logger.warning(
                f"Component '{component_name}' is not loaded (workflow selection "
                "may not use it) - skipping its configuration"
            )
            continue

        group_offload_configuration = get_group_offload_configuration(
            component_configuration, default_device
        )
        if group_offload_configuration is not None:
            # apply_group_offloading rather than the component's own
            # enable_group_offload - a component may be a transformers model, or a
            # module inside one, and only diffusers models have the method
            from diffusers.hooks import apply_group_offloading

            logger.info(f"Group offloading {component_name}")
            apply_group_offloading(component, **group_offload_configuration)

        device = component_configuration.get("device", None)
        if device is not None:
            logger.info(f"Moving {component_name} to device: {device}")
            component.to(device)

        # A compiled component should pin its attention backend - the per-call
        # attention_backend context manager would switch implementations under a
        # compiled graph and force a recompile on every run
        component_attention_backend = component_configuration.get(
            "attention_backend", None
        )
        if component_attention_backend is not None:
            logger.info(
                f"Setting {component_name} attention backend: {component_attention_backend}"
            )
            component.set_attention_backend(component_attention_backend)

        # Compile last - the graph must capture final dtypes, adapters,
        # quantization, and offload hooks
        compile_configuration = component_configuration.get("compile", None)
        if compile_configuration is not None:
            apply_compile(
                component,
                component_name,
                compile_configuration,
                device if device is not None else default_device,
            )


def apply_compile(component, component_name, compile_configuration, device):
    """Compile a component with torch.compile.

    Compilation happens in place (nn.Module.compile) so the module stays registered
    on its pipeline. With 'repeated_blocks' true, only the model's repeated block
    classes are compiled (diffusers' regional compilation) - near the same speedup
    as full compilation with a fraction of the cold-start cost.

    Args:
        component: The component to compile
        component_name: Name of the component, for logging
        compile_configuration: Dict of options - 'repeated_blocks' selects regional
            compilation, everything else ('mode', 'fullgraph', 'dynamic', ...) is
            passed to torch.compile
        device: Device the component runs on
    """
    # Inductor support on MPS is too immature to be worth the compile time
    if get_device_type(device) == "mps":
        logger.warning(
            f"torch.compile is not supported on MPS, skipping {component_name}"
        )
        return

    options = dict(compile_configuration)
    repeated_blocks = options.pop("repeated_blocks", False)

    if repeated_blocks:
        if not has_method(component, "compile_repeated_blocks"):
            raise ValueError(
                f"repeated_blocks compilation requires a diffusers model with "
                f"repeated block support, {type(component).__name__} does not have it"
            )
        logger.info(f"Compiling repeated blocks of {component_name}")
        component.compile_repeated_blocks(**options)
    else:
        logger.info(f"Compiling {component_name}")
        component.compile(**options)


_MISSING = object()


def get_component(pipeline, component_name):
    """Look a component up on a pipeline, by name or by a dotted path into it.

    A dotted path reaches a module inside a component, which is how a component that
    holds the model rather than being one - a transformers model wrapping its own - is
    offloaded.

    A modular pipeline registers a component it has not loaded (a workflow selection
    that does not use it, or one diffusers warned-and-skipped past) as a None-valued
    attribute rather than omitting it entirely - that is a real attribute, not a typo,
    so it is returned as None rather than raising. A missing attribute is still a hard
    error: it means the name itself is wrong. Callers decide what "unloaded" should
    mean for them (skip with a warning, skip silently, ...); this just tells them apart.

    Args:
        pipeline: The loaded pipeline
        component_name: Name of the component, e.g. 'vae' or 'text_encoder.model'

    Returns:
        The named component, or None if it (or a step along a dotted path) is
        registered but not loaded

    Raises:
        ValueError: If the pipeline has no attribute by that name (or dotted path)
    """
    component = pipeline
    for attribute_name in component_name.split("."):
        component = getattr(component, attribute_name, _MISSING)
        if component is _MISSING:
            raise ValueError(
                f"{type(pipeline).__name__} has no component '{component_name}'"
            )
        if component is None:
            return None

    return component


def attach_audio_sample_rate(pipeline, output):
    """Record the vocoder's sample rate on an output that carries generated audio.

    Pipelines that generate audio with their video (LTX-2) return the waveform without
    its sample rate - only the vocoder that produced it knows that. Saving the video
    needs the rate to mux the audio, so it travels with the output.

    Args:
        pipeline: The pipeline that produced the output
        output: The pipeline output
    """
    if getattr(output, "audio", None) is None:
        return

    vocoder_config = getattr(getattr(pipeline, "vocoder", None), "config", None)
    sample_rate = getattr(vocoder_config, "output_sampling_rate", None)
    if sample_rate is None:
        logger.warning(
            "Pipeline generated audio but has no vocoder sample rate - "
            "set 'audio_sample_rate' in the step result to save it with the video"
        )
        return

    logger.debug(f"Generated audio has a sample rate of {sample_rate}Hz")
    output.audio_sample_rate = sample_rate


def load_loras(loras, pipeline):
    """Load and configure LoRA models."""
    adapter_names = []
    adapter_weights = []

    for i, lora in enumerate(loras):
        model_name = lora.pop("model_name", None)
        logger.info(f"Loading LoRA: {model_name}")

        # Use provided adapter_name or generate from index
        adapter_name = lora.pop("adapter_name", str(i))
        adapter_names.append(adapter_name)

        # Extract scale for adapter weights
        scale = lora.pop("scale", 1.0)
        adapter_weights.append(scale)

        # Load the LoRA with the adapter name
        pipeline.load_lora_weights(model_name, adapter_name=adapter_name, **lora)

    # Set adapter weights for all loaded LoRAs
    if adapter_names:
        logger.info(
            f"Setting adapter weights: {list(zip(adapter_names, adapter_weights))}"
        )
        pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)


def load_ip_adapter(ip_adapter_definition, pipeline):
    """Load and configure IP-Adapter if specified."""
    if ip_adapter_definition is not None:
        model_name = ip_adapter_definition.pop("model_name")
        logger.info(f"Loading IP-Adapter: {model_name}")
        scale = ip_adapter_definition.pop("scale", None)
        pipeline.load_ip_adapter(model_name, **ip_adapter_definition)
        if scale is not None:
            pipeline.set_ip_adapter_scale(scale)


def load_and_configure_scheduler(scheduler_definition, pipeline):
    """Load and configure custom scheduler if specified."""
    if scheduler_definition is not None:
        scheduler_configuration = scheduler_definition.get("configuration", None)
        from_config_args = scheduler_definition.get("from_config_args", {})
        scheduler_type = scheduler_configuration.get("scheduler_type", None)
        logger.info(f"Loading scheduler: {scheduler_type}")

        pipeline.scheduler = scheduler_type.from_config(
            pipeline.scheduler.config, **from_config_args
        )


def auto_cpu_offload_enabled(configuration):
    """Whether the configuration asks its components manager to offload to the CPU."""
    return configuration.get("components_manager", {}).get(
        "enable_auto_cpu_offload", False
    )


def create_components_manager(configuration, device):
    """Create the components manager for a modular pipeline, when one is configured.

    A ComponentsManager tracks the components of a modular pipeline and can keep only
    the ones currently running on the device, moving the rest to system memory.

    Args:
        configuration: Pipeline configuration dictionary
        device: Device the pipeline runs on

    Returns:
        A configured ComponentsManager, or None when the pipeline does not use one
    """
    manager_configuration = configuration.get("components_manager", None)
    if manager_configuration is None:
        return None

    # Imported here because importing modular diffusers warns that it is experimental
    from diffusers import ComponentsManager

    logger.info("Creating components manager")
    components_manager = ComponentsManager()

    if auto_cpu_offload_enabled(configuration):
        offload_arguments = {}
        memory_reserve_margin = manager_configuration.get("memory_reserve_margin", None)
        if memory_reserve_margin is not None:
            offload_arguments["memory_reserve_margin"] = memory_reserve_margin

        # Enabled before the components load so each one is hooked as it is added
        logger.info(f"Enabling components manager auto CPU offload on {device}")
        components_manager.enable_auto_cpu_offload(device=device, **offload_arguments)

    return components_manager


def has_component_group_offload(configuration):
    """Whether a per-component entry configures group offloading.

    The 'components' block is applied by configure_components() after the pipeline is
    loaded, but load_component() has to decide where to materialize weights and whether
    to move the pipeline to the device before that block is ever read. A workflow whose
    only offload configuration lives under components.*.group_offload still needs both
    of those earlier decisions to treat it as offloading.

    Args:
        configuration: Configuration of the component being loaded

    Returns:
        True when any per-component entry configures group offloading
    """
    components = configuration.get("components") or {}
    return any(
        isinstance(settings, dict) and settings.get("group_offload") is not None
        for settings in components.values()
    )


def loading_device(configuration):
    """The device a component's weights are materialized on while it loads.

    Offloading brings each part of a model onto the device only while it runs, so the
    weights have to land in system memory first. A default torch device pointing at the
    GPU would build every module directly in VRAM instead, running a large pipeline out
    of memory before its offload hooks are ever installed.

    Args:
        configuration: Configuration of the component being loaded

    Returns:
        A context manager active for the duration of the load
    """
    offloads = (
        configuration.get("offload", None) is not None
        or configuration.get("group_offload", None) is not None
        or has_component_group_offload(configuration)
    )

    if offloads:
        logger.debug("Loading into system memory - the component will be offloaded")
        return torch.device("cpu")

    return contextlib.nullcontext()


def load_component(component_name, configuration, from_pretrained_arguments, device):
    """Load and configure a pipeline or component."""
    component_type = configuration["component_type"]
    component = None

    # A modular pipeline can hand its components to a ComponentsManager, which then
    # owns their device placement
    components_manager = create_components_manager(configuration, device)
    if components_manager is not None:
        from_pretrained_arguments["components_manager"] = components_manager

    # MPS (Apple Silicon) has numerical instability with float16 matmul operations,
    # producing NaN values that result in black images. The dtype is left as asked for -
    # silently loading a model in a dtype the workflow did not request would be worse -
    # so this only warns.
    if (
        get_device_type(device) == "mps"
        and from_pretrained_arguments.get("torch_dtype") == torch.float16
    ):
        logger.warning(
            f"On MPS devices float16 produces NaN values on Apple Silicon"
            f"Consider changing torch_dtype from float16 to float32 for {component_name} "
        )

    try:
        with loading_device(configuration):
            # Load from model name
            if "model_name" in from_pretrained_arguments:
                model_name = from_pretrained_arguments.pop("model_name")
                logger.info(f"Loading {component_name} from model: {model_name}")
                component = component_type.from_pretrained(
                    model_name, **from_pretrained_arguments
                )

            # Load from single file
            elif "from_single_file" in from_pretrained_arguments:
                from_single_file = from_pretrained_arguments.pop("from_single_file")
                logger.info(
                    f"Loading {component_name} from single file: {from_single_file}"
                )
                component = component_type.from_single_file(
                    from_single_file, **from_pretrained_arguments
                )

            # Create new component
            else:
                logger.info(f"Creating new {component_name}")
                component = component_type(**from_pretrained_arguments)

            # Modular pipelines load only their config in from_pretrained - the component
            # weights are pulled separately by load_components()
            load_components_arguments = get_load_components_arguments(configuration)
            if load_components_arguments is not None:
                if not has_method(component, "load_components"):
                    raise ValueError(
                        f"load_components is only supported on modular pipelines, "
                        f"{component_type.__name__} does not have it"
                    )
                logger.info(f"Loading components for {component_name}")
                component.load_components(**load_components_arguments)

        # Handle group_offload configuration
        group_offload_configuration = get_group_offload_configuration(
            configuration, device
        )
        if group_offload_configuration is not None:
            component.enable_group_offload(**group_offload_configuration)

        # Handle enable_layerwise_casting configuration
        enable_layerwise_casting_configuration = configuration.get(
            "enable_layerwise_casting", None
        )
        if enable_layerwise_casting_configuration is not None:
            component.enable_layerwise_casting(**enable_layerwise_casting_configuration)

        # Configure component device settings
        do_not_send_to_device = configuration.get("do_not_send_to_device", False)
        offload = configuration.get("offload", None)

        # Offloading streams a model between system memory and an accelerator - there is
        # nothing to stream to when the run is on the CPU
        if offload is not None and get_device_type(device) == "cpu":
            logger.warning(
                f"Ignoring '{offload}' offload - {device} is not an accelerator"
            )
            offload = None

        if offload == "model":
            logger.debug(f"Enabling model CPU offload onto {device}")
            component.enable_model_cpu_offload(device=device)
        elif offload == "sequential":
            logger.debug(f"Enabling sequential CPU offload onto {device}")
            for component_name in configuration.get("exclude_from_cpu_offload", []):
                logger.debug(f"Excluding {component_name} from CPU offload")
                component._exclude_from_cpu_offload.append(component_name)
            component.enable_sequential_cpu_offload(device=device)
        elif components_manager is not None and auto_cpu_offload_enabled(configuration):
            # Moving everything to the device here would defeat the offloading - the
            # manager's hooks bring each component on device as the pipeline needs it
            logger.debug("Device placement is owned by the components manager")
        elif has_component_group_offload(configuration):
            # configure_components() installs group-offload hooks per-component after
            # this returns - moving the whole pipeline to the device now would load it
            # in full before those hooks exist, defeating the offloading
            logger.info(
                f"components configure group offloading - not moving pipeline to {device}"
            )
        elif hasattr(component, "to") and not do_not_send_to_device:
            logger.debug(f"Moving {component_name} to device: {device}")
            component = component.to(device)

        return component

    except Exception as e:
        # One log line with the full traceback - every error class was logged
        # and re-raised identically
        logger.error(f"{type(e).__name__} loading {component_name}: {e}", exc_info=True)
        raise


def apply_sdnq_optimizations(pipeline, component_names):
    """Apply SDNQ quantized matmul optimization to pipeline components.

    Uses sdnq's apply_sdnq_options_to_model to enable INT8 matmul
    on supported hardware (CUDA, XPU).

    Args:
        pipeline: The loaded diffusers pipeline
        component_names: List of component names to optimize (e.g., ["transformer", "text_encoder"])
    """
    try:
        from sdnq.loader import apply_sdnq_options_to_model
        from sdnq.common import use_torch_compile as triton_is_available
    except ImportError:
        logger.warning("sdnq not installed, skipping SDNQ optimizations")
        return

    if not triton_is_available:
        logger.info("Triton not available, skipping SDNQ quantized matmul optimization")
        return

    if not (
        torch.cuda.is_available() or hasattr(torch, "xpu") and torch.xpu.is_available()
    ):
        logger.info(
            "SDNQ quantized matmul requires CUDA or XPU, skipping on this device"
        )
        return

    for name in component_names:
        # A missing name (typo) and a registered-but-unloaded one both mean "nothing
        # to optimize here" for this call - same warn-and-skip either way
        try:
            component = get_component(pipeline, name)
        except ValueError:
            component = None

        if component is not None:
            logger.info(f"Applying SDNQ quantized matmul to {name}")
            setattr(
                pipeline,
                name,
                apply_sdnq_options_to_model(component, use_quantized_matmul=True),
            )
        else:
            logger.warning(
                f"Component '{name}' not found on pipeline, skipping SDNQ optimization"
            )


@contextlib.contextmanager
def stateful_cache_context(pipeline):
    """Provide the context a stateful cache hook reads its state through.

    first_block, mag and layer_skip keep per-context state, and their hooks go
    through diffusers' StateManager, which raises "No context is set" unless a
    context is active. A DiffusionPipeline sets one around each denoising step and
    clears the state afterwards in maybe_free_model_hooks; ModularPipeline is not a
    DiffusionPipeline and does neither, so caching a modular pipeline dies on the
    first step - and would otherwise carry the previous run's residuals into the
    next run of a pipeline this process keeps loaded.

    One context spans the whole call rather than each step. The state is keyed by
    context name, so re-entering per step only re-reads the same entry. Pipelines
    that run separate conditional and unconditional passes name a context per pass
    to keep their caches apart, which a shared context would defeat - but a modular
    pipeline that needed that would be setting its own contexts already, and this
    is a no-op for pipelines whose cache is not enabled.
    """
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None or not getattr(transformer, "is_cache_enabled", False):
        yield
        return

    logger.debug(f"Entering cache context for {transformer.__class__.__name__}")
    try:
        with transformer.cache_context(_CACHE_CONTEXT_NAME):
            yield
    finally:
        # Private, but it is what diffusers' own pipelines call and there is no
        # public equivalent. Also clears the context an errored call left set
        transformer._reset_stateful_cache()


def enable_cache_on_transformer(pipeline, cache_config):
    """Enable cache configuration on the pipeline's transformer.

    Args:
        pipeline: The loaded diffusers pipeline
        cache_config: Cache configuration object from get_cache_configuration()
    """
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        logger.warning("Pipeline has no transformer, skipping cache configuration")
        return

    if not hasattr(transformer, "enable_cache"):
        logger.warning(
            f"{transformer.__class__.__name__} does not support enable_cache(), skipping"
        )
        return

    # FasterCache decides skipping from the pipeline's current timestep. The
    # callback is a callable, which workflow JSON cannot express, and diffusers
    # calls it unconditionally on every denoiser forward - left None, the first
    # inference step dies. Wire it to the pipeline here, where both exist
    if (
        cache_config.__class__.__name__ == "FasterCacheConfig"
        and getattr(cache_config, "current_timestep_callback", None) is None
    ):
        logger.debug("Wiring FasterCache current_timestep_callback to the pipeline")
        cache_config.current_timestep_callback = lambda: pipeline._current_timestep

    # first_block, mag and layer_skip resolve the transformer's block class
    # through diffusers' registry and raise when it is absent - fill in the
    # blocks diffusers has not registered before handing the config over
    register_cache_blocks()

    logger.info(
        f"Enabling {cache_config.__class__.__name__} on {transformer.__class__.__name__}"
    )
    transformer.enable_cache(cache_config)
