import torch
import contextlib
import copy
import functools
import gc
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
from .. import empty_device_cache, get_device_type
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

    def __init__(
        self,
        pipeline_definition,
        default_seed,
        device,
        pipeline=None,
        output_dir=None,
        file_prefix=None,
    ):
        """
        Initialize pipeline with configuration and device settings.

        Args:
            pipeline_definition: Dictionary containing pipeline configuration
            default_seed: Seed value for reproducibility
            device: Device to run pipeline on (e.g., 'cuda', 'mps', 'cpu') - the
                configuration's own 'device' takes precedence over it
            pipeline: Optional existing pipeline to use
            output_dir: The workflow's output directory - where a chained run
                with save_segments writes its segment files
            file_prefix: Naming prefix for those files, matching the step's
                result naming (workflow id + step name)
        """
        self.pipeline_definition = pipeline_definition
        self.default_seed = default_seed
        # A step can pin itself to a device, overriding the one dw is running on. It
        # becomes the default for this pipeline's components as well
        self.device = self.configuration.get("device", device)
        self.pipeline = pipeline
        self.output_dir = output_dir
        self.file_prefix = file_prefix
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

    def component_names(self, key):
        """The component names one of the sharing lists holds.

        The lists were only ever read off the pipeline itself, while the schema and
        the guide put them in its configuration - a workflow written to the docs
        shared nothing and said nothing about it. Both places are read now.

        Args:
            key: 'shared_components' or 'reused_components'

        Returns:
            List of component names
        """
        return list(self.pipeline_definition.get(key, [])) + list(
            self.configuration.get(key, [])
        )

    def resolve_reused_components(self, shared_components):
        """The components an earlier step shared that this one asks to reuse.

        Args:
            shared_components: Dictionary of components shared between pipelines

        Returns:
            Dict of component name to the component itself

        Raises:
            ValueError: If a name was never shared by an earlier step
        """
        reused = {}
        for name in self.component_names("reused_components"):
            if name not in shared_components:
                raise ValueError(
                    f"Cannot reuse component '{name}' - no earlier step shared it. "
                    f"Shared so far: {sorted(shared_components) or 'nothing'}"
                )
            logger.debug(f"Reusing component: {name}")
            reused[name] = shared_components[name]
        return reused

    def populate_from_pretrained_arguments(self, device, shared_components):
        """
        Prepare arguments for pipeline creation, including shared components.

        The loaded components go into a copy, not into the definition they were read
        from. The definition belongs to the workflow and outlives every step, so a
        component stored there is a component the run holds until it ends: releasing
        the pipeline frees nothing, and a workflow that loads a second large model
        after releasing the first runs out of memory holding both. Copying also
        leaves the definition intact for a second load - load_component consumes
        'model_name' out of the arguments it is handed.

        Args:
            device: Device to run pipeline on
            shared_components: Dictionary of components shared between pipelines
        """
        logger.debug("Populating from_pretrained arguments")
        from_pretrained_arguments = dict(self.from_pretrained_arguments)

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
        reused_components = self.resolve_reused_components(shared_components)

        # Load and configure the main pipeline
        self.pipeline = load_component(
            "pipeline",
            self.configuration,
            from_pretrained_arguments,
            self.device,
            reused_components,
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

        # Configure the schedulers if specified - a pipeline that denoises two
        # modalities against two schedules configures each of them separately
        load_and_configure_scheduler(
            self.pipeline_definition.get("scheduler", None), self.pipeline
        )
        load_and_configure_scheduler(
            self.pipeline_definition.get("audio_scheduler", None),
            self.pipeline,
            "audio_scheduler",
        )

        # Store components that will be shared with other pipelines. get_component
        # rather than getattr - a modular pipeline registers a component it did not
        # load as None rather than omitting it, and sharing that None silently would
        # surface as a missing-component error inside the step that reused it
        for shared_component_name in self.component_names("shared_components"):
            component = get_component(self.pipeline, shared_component_name)
            if component is None:
                raise ValueError(
                    f"Cannot share component '{shared_component_name}' - "
                    f"{type(self.pipeline).__name__} registers it but has not "
                    f"loaded it"
                )
            logger.debug(f"Storing shared component: {shared_component_name}")
            shared_components[shared_component_name] = component

        # Load and configure LoRA models
        load_loras(self.pipeline_definition.get("loras", []), self.pipeline)

        # Load and configure IP-Adapter
        load_ip_adapter(self.pipeline_definition.get("ip_adapter", None), self.pipeline)

        # Place the components the pipeline loaded itself, once everything that alters
        # them - dtypes, adapters, quantized matmuls - has been applied. Offloading hooks
        # installed before those would be fighting them
        configure_components(
            self.pipeline, self.configuration, self.device, reused_components
        )

        # Set up random generator if needed - no_generator is a boolean, so an
        # explicit false still gets a generator
        if not self.configuration.get("no_generator", False):
            logger.debug("Setting up random generator")
            self.argument_template["generator"] = torch.Generator(
                self.device
            ).manual_seed(self.pipeline_definition.get("seed", self.default_seed))

        # Hand the first run a clean allocator. Loading churns the device even
        # when little of the pipeline stays there - a quantization pass with
        # 'quantization_device' set works on the accelerator and returns the
        # weights to the host, and group offloading moves components off it
        # again - and the cached blocks left behind are the wrong shape for
        # inference. workflow.py does this between steps; a one-step workflow
        # would otherwise run its only step on top of the loading debris
        gc.collect()
        empty_device_cache()

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

            chain_definition = self.pipeline_definition.get("chain", None)
            if chain_definition is not None:
                from .chain import run_chain

                logger.debug("Running chained pipeline")
                return run_chain(self, chain_definition, arguments)

            return self._run_once(arguments)

        except Exception as e:
            # One log line with the full traceback - every error class was
            # logged and re-raised identically
            logger.error(f"{type(e).__name__} running pipeline: {e}", exc_info=True)
            raise

    def _run_once(self, arguments):
        """Run one standard pipeline invocation with fully resolved arguments.

        This is the whole per-call execution path - prompt encoding, the
        pipeline call itself, and output normalization - shared by the single
        run and every segment of a chained run.
        """
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
                logger.info(f"Using attention backend: {attn_backend}")
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
                # A copy for the same reason the pipeline's own arguments are copied:
                # what goes in here is consumed by load_component, and the definition
                # is the workflow's, not this load's
                component_from_pretrained_arguments = dict(
                    component_definition["from_pretrained_arguments"]
                )

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


def configure_components(pipeline, configuration, default_device, reused_components=()):
    """Place the components a pipeline loaded for itself.

    A modular pipeline pulls its own component weights, so they are only reachable once
    the pipeline is loaded - too late for the offloading load_component sets up. Group
    offloading a component here streams it between system memory and the accelerator a
    piece at a time, which is what fits a pipeline whose components are each larger than
    the device.

    A component this step reused is not one it loaded: it already carries the placement
    the step that shared it gave it, and offloading hooks do not survive being applied
    twice. Those are skipped, so a workflow can reuse a component into a step whose
    configuration was written for loading it.

    Args:
        pipeline: The loaded pipeline
        configuration: Pipeline configuration dictionary
        default_device: Device the pipeline runs on
        reused_components: Names of the components an earlier step shared into this one
    """
    for component_name, component_configuration in configuration.get(
        "components", {}
    ).items():
        # A dotted path reaches inside a component, and it is the component itself
        # that was shared - 'text_encoder.model' belongs to a reused 'text_encoder'
        if component_name.split(".")[0] in reused_components:
            logger.info(
                f"Component '{component_name}' was shared by an earlier step - "
                "keeping the placement that step gave it"
            )
            continue

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

        # Tiled decoding, for a component that decodes but is not the one called
        # 'vae' - LTX-2.5's diffusion decoder, which decodes the whole video volume
        # in one allocation unless it is told to tile
        enable_tiling(component, component_name, component_configuration)

        device = component_configuration.get("device", None)
        residency = component_configuration.get("residency", "resident")
        if residency == "on_demand":
            apply_on_demand_placement(
                component,
                component_name,
                device if device is not None else default_device,
                group_offload_configuration is not None,
            )
        elif device is not None:
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


def enable_tiling(component, component_name, component_configuration):
    """Turn on tiled decoding for a component whose configuration asks for it.

    The pipeline-level `vae` block covers the component actually named 'vae'. This
    covers any other component that decodes - LTX-2.5's `diffusion_decoder`, which
    otherwise decodes the whole video volume in a single allocation and asks for
    tens of GiB at 2x resolutions. `true` takes the model's own default tile size;
    a dict passes the tile and stride sizes through, which is what a card smaller
    than those defaults needs.

    Args:
        component: The loaded component
        component_name: Its name, for logging and errors
        component_configuration: That component's configuration block

    Raises:
        ValueError: If the component has no enable_tiling() to call
    """
    tiling = component_configuration.get("enable_tiling", False)
    if not tiling:
        return

    if not has_method(component, "enable_tiling"):
        raise ValueError(
            f"'{component_name}' does not support tiling - "
            f"{type(component).__name__} has no enable_tiling()"
        )

    arguments = tiling if isinstance(tiling, dict) else {}
    logger.info(
        f"Enabling tiling on {component_name}"
        + (f" with {', '.join(arguments)}" if arguments else "")
    )
    component.enable_tiling(**arguments)


# The calls that mean "this component is working now". A component is moved to
# the accelerator around whichever of these it actually defines
_ON_DEMAND_ENTRY_POINTS = ("forward", "encode", "decode")


def apply_on_demand_placement(
    component, component_name, device, group_offloaded, offload_device="cpu"
):
    """Keep a component in system memory and move it to the device only while it runs.

    Sits between the two placements dw already has. A 'device' component is resident
    for the whole run, which wastes the accelerator on something used twice; group
    offloading streams per submodule forward, which restreams the whole model once
    per call of every leaf - ruinous for a VAE, whose tiled decode calls its blocks
    once per tile. This moves the model as a whole around each entry point, so a
    tiling loop sits inside a single pair of transfers.

    That trade only pays for components called a handful of times per run. A
    denoising transformer is called once per step, so per-call transfers would cost
    far more than they save - group offloading is the tool for those.

    Args:
        component: The component to place
        component_name: Name of the component, for logging
        device: Device to run the component on
        group_offloaded: Whether group offloading was applied to this component
        offload_device: Where the component rests between calls

    Raises:
        ValueError: If the component is also group offloaded
    """
    if group_offloaded:
        raise ValueError(
            f"Component '{component_name}' sets both 'group_offload' and "
            "'residency: on_demand'. A group offloaded module holds one group at a "
            "time and ignores the whole-model moves on-demand placement makes, so "
            "the two cannot both own its placement - pick one"
        )

    if get_device_type(device) == "cpu":
        # Nothing to move it off of, so the wrappers would be pure overhead
        logger.debug(
            f"Ignoring 'residency: on_demand' for {component_name} - {device} is the "
            "device it would rest on anyway"
        )
        return

    component.to(offload_device)

    # One depth counter for the whole component, not one per entry point: decode()
    # calls forward() internally, and an inner return must not offload the model
    # out from under the call that is still running
    state = {"depth": 0}

    def wrap(entry_point):
        original = getattr(component, entry_point, None)
        if not callable(original):
            return False

        @functools.wraps(original)
        def on_demand(*args, **kwargs):
            if state["depth"] == 0:
                component.to(device)
            state["depth"] += 1
            try:
                return original(*args, **kwargs)
            finally:
                state["depth"] -= 1
                if state["depth"] == 0:
                    component.to(offload_device)
                    # Hand the freed space back to the driver rather than leaving it
                    # reserved - the headroom is the entire point of doing this
                    empty_device_cache()

        # functools.wraps carries __wrapped__, so inspect.signature() still reports
        # the real parameters. Callers introspect them: MiniMax H3's denoiser picks
        # which arguments to pass by reading signature(transformer.forward)
        setattr(component, entry_point, on_demand)
        return True

    wrapped = [name for name in _ON_DEMAND_ENTRY_POINTS if wrap(name)]
    if not wrapped:
        raise ValueError(
            f"Component '{component_name}' sets 'residency: on_demand' but defines "
            f"none of {', '.join(_ON_DEMAND_ENTRY_POINTS)}, so there is no call to "
            "move it around"
        )
    logger.info(
        f"Placing {component_name} on demand: resting on {offload_device}, "
        f"running on {device} around {', '.join(wrapped)}"
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

        # Extract scale for adapter weights - float() because the schema takes a
        # 'variable:' reference here, and a variable declared as a string default
        # substitutes as one
        scale = float(lora.pop("scale", 1.0))
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


def load_and_configure_scheduler(
    scheduler_definition, pipeline, component_name="scheduler"
):
    """Load and configure a pipeline's scheduler if specified.

    A definition does either or both of two things, in that order: replace the
    scheduler with one built from another type's config, and set the sigma
    shift on whatever scheduler the pipeline then holds.

    The component is named rather than assumed because a pipeline can carry
    more than one. MiniMax-H3 steps video and audio latents down two schedules
    inside a single transformer call - 'scheduler' and 'audio_scheduler', whose
    shifts (12.0 and 3.0 in the released checkpoint) are set independently, and
    the video one is what a few-step schedule has to lower: at the checkpoint's
    12.0 a five-point sigma grid spends every step above 0.8 and then drops to
    zero in one, which denoises to noise.

    Args:
        scheduler_definition: The step's scheduler block, or None
        pipeline: The loaded pipeline
        component_name: Which scheduler the definition configures
    """
    if scheduler_definition is None:
        return

    scheduler_configuration = scheduler_definition.get("configuration", None) or {}
    scheduler_type = scheduler_configuration.get("scheduler_type", None)
    if scheduler_type is not None:
        from_config_args = scheduler_definition.get("from_config_args", {})
        logger.info(f"Loading {component_name}: {scheduler_type}")
        setattr(
            pipeline,
            component_name,
            scheduler_type.from_config(
                get_component(pipeline, component_name).config, **from_config_args
            ),
        )

    shift = scheduler_definition.get("shift", None)
    if shift is None:
        return

    scheduler = get_component(pipeline, component_name)
    if scheduler is None:
        raise ValueError(
            f"Cannot set a shift on '{component_name}' - the pipeline registers "
            "it but has not loaded it"
        )
    if not has_method(scheduler, "set_shift"):
        raise ValueError(
            f"{type(scheduler).__name__} does not take a sigma shift - "
            f"'{component_name}' has no set_shift()"
        )

    # Instance state the scheduler keeps until its next set_timesteps, which is
    # the run itself - so this survives loading and every later run of the step
    logger.info(f"Setting {component_name} shift: {shift}")
    scheduler.set_shift(float(shift))


def auto_cpu_offload_enabled(configuration):
    """Whether the configuration asks its components manager to offload to the CPU."""
    return configuration.get("components_manager", {}).get(
        "enable_auto_cpu_offload", False
    )


def auto_cpu_offload_active(configuration, device):
    """Whether the components manager actually owns device placement.

    Mirrors the MPS skip in create_components_manager() - on MPS the manager
    never installs its offload hooks, so callers must not assume it owns
    device placement there.
    """
    return auto_cpu_offload_enabled(configuration) and get_device_type(device) != "mps"


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
        # ComponentsManager.enable_auto_cpu_offload() calls device.mem_get_info(),
        # which torch does not implement for MPS. Unified memory also makes the
        # feature far less useful there than on CUDA, so skip it rather than fail.
        if get_device_type(device) == "mps":
            logger.warning(
                "components_manager auto CPU offload is not supported on MPS, skipping"
            )
        else:
            offload_arguments = {}
            memory_reserve_margin = manager_configuration.get(
                "memory_reserve_margin", None
            )
            if memory_reserve_margin is not None:
                offload_arguments["memory_reserve_margin"] = memory_reserve_margin

            # Enabled before the components load so each one is hooked as it is added
            logger.info(f"Enabling components manager auto CPU offload on {device}")
            components_manager.enable_auto_cpu_offload(
                device=device, **offload_arguments
            )

    return components_manager


def has_component_group_offload(configuration):
    """Whether a per-component entry keeps its component off the device.

    The 'components' block is applied by configure_components() after the pipeline is
    loaded, but load_component() has to decide where to materialize weights and whether
    to move the pipeline to the device before that block is ever read. A workflow whose
    only offload configuration lives under components.* still needs both of those
    earlier decisions to treat it as offloading.

    Group offloading and on-demand residency both qualify: each leaves its component in
    system memory between uses, so materializing the pipeline on the device first would
    load in full exactly what these were configured to avoid holding.

    Args:
        configuration: Configuration of the component being loaded

    Returns:
        True when any per-component entry keeps its component off the device
    """
    components = configuration.get("components") or {}
    return any(
        isinstance(settings, dict)
        and (
            settings.get("group_offload") is not None
            or settings.get("residency") == "on_demand"
        )
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


def get_block_configs(configuration, component):
    """The block configs a workflow sets on a modular pipeline, checked against it.

    A modular pipeline's blocks declare configs of their own - values they read while
    they run rather than components or call arguments. MiniMax-H3 declares three, and
    they are how the canvas the request generates on and the resolution its references
    are encoded at are set:

        "configs": { "canvas_short_edge": 768, "reference_image_short_edge": 1024 }

    This is deliberately not a knob per config per model. Every modular pipeline
    declares its own set, `update_components()` sets any of them, and what a workflow
    may say here is whatever the pipeline it named declares. The names are checked
    because update_components ignores the ones it does not know with a warning, and a
    silently dropped config reads as a setting that did nothing.

    Args:
        configuration: Pipeline configuration dictionary
        component: The loaded pipeline the configs are for

    Returns:
        Dict of config name to value, empty when the workflow sets none

    Raises:
        ValueError: If the pipeline takes no configs, or does not declare one by name
    """
    configs = configuration.get("configs", None)
    if not configs:
        return {}

    if not has_method(component, "update_components"):
        raise ValueError(
            f"'configs' is only supported on modular pipelines, "
            f"{type(component).__name__} does not have update_components"
        )

    # The specs a pipeline builds from its blocks. Guarded rather than indexed - a
    # pipeline that stops keeping them under this name should lose the check, not
    # the feature
    declared = getattr(component, "_config_specs", None)
    if declared is not None:
        unknown = [name for name in configs if name not in declared]
        if unknown:
            raise ValueError(
                f"{type(component).__name__} declares no config named "
                f"{', '.join(sorted(unknown))} - the ones it declares are "
                f"{', '.join(sorted(declared)) or 'none'}"
            )

    logger.info(f"Setting block configs: {', '.join(configs)}")
    return dict(configs)


def load_component(
    component_name,
    configuration,
    from_pretrained_arguments,
    device,
    reused_components=None,
):
    """Load and configure a pipeline or component.

    Args:
        component_name: What is being loaded, for the log
        configuration: The component's configuration block
        from_pretrained_arguments: Arguments for the constructor
        device: Device the component is loaded for
        reused_components: Components an earlier step shared into this one, by name
    """
    component_type = configuration["component_type"]
    component = None

    # A standard pipeline takes a component as a constructor argument. A modular one
    # cannot: it is built from the component specs in its own index and given the
    # objects afterwards, which is also what keeps load_components() from pulling a
    # second copy of the weights - it skips the components already registered
    reused_components = reused_components or {}
    takes_components_after_load = has_method(component_type, "update_components")
    if reused_components and not takes_components_after_load:
        from_pretrained_arguments.update(reused_components)

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

            # Register the shared components before anything is pulled, so the
            # weights an earlier step already loaded and quantized are the ones
            # this step runs on rather than a second copy of them. The block
            # configs go in the same call - update_components takes both
            update_arguments = get_block_configs(configuration, component)
            if reused_components and takes_components_after_load:
                logger.info(
                    f"Reusing {', '.join(reused_components)} from an earlier step"
                )
                update_arguments.update(reused_components)
            if update_arguments:
                component.update_components(**update_arguments)

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
        preserve_device_placement = configuration.get(
            "preserve_device_placement", False
        )
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
        elif components_manager is not None and auto_cpu_offload_active(
            configuration, device
        ):
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
        elif hasattr(component, "to") and not preserve_device_placement:
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


def get_cache_transformer(pipeline):
    """Find the denoiser a cache hook attaches to.

    Most pipelines register theirs as 'transformer', but a modular pipeline names
    it after the workflow it serves - MiniMax-H3's ref2va denoises through
    'transformer_ref'. Looking only for 'transformer' silently skips caching on
    those, so try the alternates diffusers' modular pipelines actually use.

    Args:
        pipeline: The loaded diffusers pipeline

    Returns:
        The transformer component, or None when the pipeline has none
    """
    for name in ("transformer", "transformer_ref"):
        transformer = getattr(pipeline, name, None)
        if transformer is not None:
            return transformer
    return None


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
    transformer = get_cache_transformer(pipeline)
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
    transformer = get_cache_transformer(pipeline)
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
