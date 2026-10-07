import torch
import contextlib
import copy
import gc
import importlib
import inspect
import logging
from .config_objects import (
    get_quantization_configuration,
    get_cache_configuration,
)
from .adapters import active_loras, load_ip_adapter, load_loras
from .components import (
    apply_sdnq_optimizations,
    configure_components,
    enable_cache_on_transformer,
    get_component,
    load_and_configure_scheduler,
    load_component,
    stateful_cache_context,
)
from .placement import (
    apply_mps_rope_precision,
    attention_slicing_requested,
    place_component,
)
from .h3_blocks import (
    GUIDE_LIMIT,
    GUIDES_INPUT,
    HELD_AUDIO_OUTPUT,
    HELD_AUDIO_RATE_OUTPUT,
    HOLD_AUDIO_INPUT,
    REFINE_STRENGTH_INPUT,
    guide_audio_waveform,
    guide_frame_problem,
    guide_frames_array,
    guides_refusal,
    hold_audio_reference,
    holds_audio,
    refine_problems,
    refines,
    snap_guide_length,
)
from .progress import reported_blocks, reported_progress_bars
from .remote import remote_text_encoder
from ..type_helpers import has_method
from ..trust import (
    require_trusted_from_pretrained_arguments,
    require_trusted_pre_load_modules,
)
from .. import empty_device_cache, resolve_device
from diffusers import attention_backend

# dw.prompt_weighting (transformers) and diffusers.hooks (peft, bitsandbytes) are
# imported where they are used - at module scope they add seconds to every startup

from ..argument_media import fetch_video
from ..tasks.video_utils import load_audio_video
from ..events import WorkflowCancelled, emit_phase, emit_warning, get_context
from ..media_types import AudioVideo
from ..step_cache import component_names, copy_containers

logger = logging.getLogger("dw")

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


def _loading_copy(pipeline_definition):
    """The copy of a step's pipeline definition a Pipeline works on.

    Loading edits what it reads: group offload replaces device names with
    torch.device objects and drops the stream flags off CUDA, load_loras and
    load_ip_adapter pop their keys, and load sets 'generator' on the argument
    template. The definition belongs to the workflow - the step cache
    snapshots it, embedded metadata records it, a deferred step loads from it
    later - so none of that may reach it.

    Every edit load makes is to a container - a key assigned or popped - and
    none to a leaf in place, so the containers are copied and the leaves are
    shared: a realized type or dtype, or media, is the object the workflow
    holds, not a duplicate of it. 'arguments' is copied one level only; its
    values may be realized media, and the generator is set at its top.
    """
    copied = {}
    for key, value in pipeline_definition.items():
        if key == "arguments":
            copied[key] = dict(value) if isinstance(value, dict) else value
        else:
            copied[key] = copy_containers(value)
    return copied


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
        base_dir=None,
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
            base_dir: The workflow file's directory - what a relative media path
                in a call argument resolves against at run time
        """
        self.pipeline_definition = _loading_copy(pipeline_definition)
        self.default_seed = default_seed
        # A step can pin itself to a device, overriding the one dw is running on. It
        # becomes the default for this pipeline's components as well, and is
        # translated to a backend this machine has so the workflow travels
        self.device = resolve_device(self.configuration.get("device", device))
        self.pipeline = pipeline
        self.output_dir = output_dir
        self.file_prefix = file_prefix
        self.base_dir = base_dir
        # What a chained run calls the segment it is on, so progress can say
        # which one the denoise counter belongs to - it restarts per segment
        self.segment_label = None
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

        Args:
            key: 'shared_components' or 'reused_components'

        Returns:
            List of component names
        """
        return component_names(self.pipeline_definition, key)

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

    def check_trusted(self):
        """Refuse an untrusted definition before anything says a load began.

        The gates themselves live inside load() and load_component(), which is
        where they have to be - that is the last point before the bytes are
        fetched. But `load()` is entered under a 'loading' phase event, so a
        run refused by them emitted the same marker as one that loaded a model
        and then failed, and job events could no longer tell the two apart
        (#137). This runs the same checks over the definition first, so the
        caller emits 'loading' only once a load can actually begin. It is a
        pre-flight, not the boundary: the in-load checks stay.

        Raises:
            UntrustedWorkflowError: If untrusted and the definition reaches
                for remote code
        """
        require_trusted_pre_load_modules(self.configuration.get("pre_load_modules", []))
        # Walk the whole definition rather than the component names this class
        # knows: a gate that only covers what it remembers to enumerate stops
        # covering a block added later
        self._check_trusted_block(self.pipeline_definition, "pipeline")

    @staticmethod
    def _check_trusted_block(block, what):
        if not isinstance(block, dict):
            return
        require_trusted_from_pretrained_arguments(
            block.get("from_pretrained_arguments"), what
        )
        for key, value in block.items():
            if key == "from_pretrained_arguments":
                continue
            if isinstance(value, dict):
                Pipeline._check_trusted_block(value, key)
            elif isinstance(value, list):
                for entry in value:
                    Pipeline._check_trusted_block(entry, key)

    def load(self, shared_components):
        """
        Load and configure the pipeline with all components.

        Args:
            shared_components: Dictionary of components shared between pipelines
        """
        logger.debug(f"Loading pipeline: {self.name}")

        # Import modules that need to register with diffusers/transformers before loading
        # (e.g., sdnq registers its quantization method on import). This runs
        # arbitrary python at import time, so an untrusted workflow is refused
        # here unless --trust-workflows was passed - see docs/SECURITY.md
        pre_load_modules = self.configuration.get("pre_load_modules", [])
        require_trusted_pre_load_modules(pre_load_modules)
        for module_name in pre_load_modules:
            logger.info(f"Pre-loading module: {module_name}")
            importlib.import_module(module_name)

        # A load that raises partway has already built some of what it was
        # asked for - the pipeline itself, its quantized weights, a placement
        # half applied - and none of that is reachable from the caller, which
        # never received a pipeline. Left alone it stays resident until the
        # exception is handled and something else happens to collect, so a
        # retry loads its own copy on top of the last attempt's: three failures
        # is three pipelines' worth of dead weight. Tear the attempt down here
        # instead, then let the failure carry on
        try:
            # Prepare arguments and load pipeline
            from_pretrained_arguments = self.populate_from_pretrained_arguments(
                self.device, shared_components
            )
            reused_components = self.resolve_reused_components(shared_components)

            # Adapters add weights to the components they attach to, and an offloading
            # hook only streams the weights that existed when it was installed - so a
            # pipeline that loads any is placed after they are on it, not at load
            adapters_to_load = bool(
                active_loras(self.pipeline_definition.get("loras", []))
            ) or (self.pipeline_definition.get("ip_adapter", None) is not None)

            # Load and configure the main pipeline
            self.pipeline = load_component(
                "pipeline",
                self.configuration,
                from_pretrained_arguments,
                self.device,
                reused_components,
                defer_placement=adapters_to_load,
            )

            # Attention slicing trades speed for memory - automatic on MPS, where
            # it is often the faster path too (see attention_slicing_requested)
            if attention_slicing_requested(self.configuration, self.device):
                # Modular pipelines have no attention slicing - skip rather than fail
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

            self.publish_shared_components(shared_components)

            # Load and configure LoRA models
            load_loras(self.pipeline_definition.get("loras", []), self.pipeline)

            # Load and configure IP-Adapter
            load_ip_adapter(
                self.pipeline_definition.get("ip_adapter", None), self.pipeline
            )

            # The adapters are on the pipeline now, so its offloading hooks can be
            # installed over the weights they added
            if adapters_to_load:
                self.pipeline = place_component(
                    self.pipeline,
                    "pipeline",
                    self.configuration,
                    self.device,
                    # A modular pipeline's manager owns its placement, and the manager
                    # load_component gave it is the one it holds
                    getattr(self.pipeline, "_components_manager", None),
                )

            # Place the components the pipeline loaded itself, once everything that alters
            # them - dtypes, adapters, quantized matmuls - has been applied. Offloading hooks
            # installed before those would be fighting them
            configure_components(
                self.pipeline, self.configuration, self.device, reused_components
            )
            apply_mps_rope_precision(self.pipeline, self.device)

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
        except BaseException:
            self._discard_failed_load(shared_components)
            raise

        logger.debug("Pipeline loaded successfully")

    def _discard_failed_load(self, shared_components):
        """Drop everything a load that raised had built, and reclaim it.

        Unpublishes as well as releases: a component shared before the failure
        would otherwise be handed to a later step as a component of a pipeline
        that does not exist. A name this pipeline reused rather than loaded
        stays published - that entry belongs to the earlier pipeline that put
        it there, which is still alive.
        """
        logger.info(f"Load of pipeline '{self.name}' failed - releasing what it built")
        reused = set(self.component_names("reused_components"))
        for shared_component_name in self.component_names("shared_components"):
            if shared_component_name not in reused:
                shared_components.pop(shared_component_name, None)
        self.pipeline = None
        self.argument_template.pop("generator", None)
        gc.collect()
        empty_device_cache()

    def publish_shared_components(self, shared_components):
        """Store components that will be shared with other pipelines.

        Called from load, and again by the workflow when a cached pipeline is
        reused - a cache hit skips load entirely, and the shared_components
        dict is fresh every run, so a warm sharing step must republish or a
        later reusing step finds nothing. get_component rather than getattr -
        a modular pipeline registers a component it did not load as None, and
        sharing that None silently would surface as a missing-component error
        inside the step that reused it.
        """
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

        except WorkflowCancelled:
            logger.info("Pipeline run cancelled")
            raise
        except Exception as e:
            diagnosed = _diagnose_image_crf_error(e, arguments)
            if diagnosed is not None:
                logger.error(f"{type(e).__name__} running pipeline: {e}", exc_info=True)
                raise diagnosed from e
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
        warn_if_safety_checker_blanked(output)

        return output

    def _execute_pipeline(self, arguments):
        """Execute the pipeline inside the optional attention backend context."""
        attn_backend = self.configuration.get("attention_backend", None)
        return self._call_pipeline(arguments, attn_backend)

    def _call_pipeline(self, arguments, attn_backend):
        """Call the pipeline with optional attention backend and cache contexts."""
        arguments = self._with_step_callback(arguments)
        self._check_refine(arguments)
        arguments = self._with_held_audio(arguments)
        arguments = self._with_guides(arguments)
        # The load is over and the denoise loop is starting. Pipelines whose
        # signature has no step callback report nothing else at all, so this
        # is the only thing that distinguishes running from still loading
        emit_phase("generating", detail=self.segment_label or self.name)
        with contextlib.ExitStack() as stack:
            if attn_backend is not None:
                logger.info(f"Using attention backend: {attn_backend}")
                stack.enter_context(attention_backend(attn_backend))

            stack.enter_context(stateful_cache_context(self.pipeline))
            if not self._takes_step_callback():
                # A modular pipeline takes no step callback at all, so this
                # is the only per-step signal it has: the denoise blocks
                # drive a tqdm bar, and a bar that reports each advance is
                # the difference between a slow run and a hung one
                stack.enter_context(reported_progress_bars(self.pipeline))
                # The bar only covers the denoise loop; the blocks around it
                # are where a reference encode's minutes go (#95)
                stack.enter_context(
                    reported_blocks(self.pipeline, self.segment_label or self.name)
                )

            return self.pipeline(**arguments)

    def _with_held_audio(self, arguments):
        """`hold_audio` as the reference the H3 hold block takes, and the
        held track asked for alongside `audio` (dw/pipeline_processors/h3_blocks.py).

        Raises:
            ValueError: If this pipeline cannot hold audio, or the value is not audio
        """
        held = arguments.get(HOLD_AUDIO_INPUT)
        if held is None:
            return arguments
        if not holds_audio(self.pipeline):
            raise ValueError(
                f"Step '{self.name}': hold_audio is a MiniMax-H3 argument "
                f"(t2va, fl2va or ref2va), and {type(self.pipeline).__name__} "
                f"cannot hold a soundtrack"
            )
        arguments = dict(arguments)
        arguments[HOLD_AUDIO_INPUT] = hold_audio_reference(held, self.base_dir)
        output = arguments.get("output")
        if isinstance(output, (list, tuple)) and "audio" in output:
            arguments["output"] = list(output) + [
                HELD_AUDIO_OUTPUT,
                HELD_AUDIO_RATE_OUTPUT,
            ]
        return arguments

    def _with_guides(self, arguments):
        """`guides` as the H3 guide layout takes them - each clip as uint8
        frames cut to a whole-latent length (dw/pipeline_processors/h3_blocks.py).
        An empty list is no guides.

        A guide with `"audio": true` also carries its video's soundtrack, as
        `audio` and `sample_rate`; with `audio` false or absent it carries none.

        Raises:
            ValueError: If this pipeline cannot take guides, the step also passes
                `references`, a guide is not `{video, frame, audio?}` with a video,
                or `audio` is true on a video with no audio
        """
        guides = arguments.get(GUIDES_INPUT)
        if guides is None:
            return arguments
        arguments = dict(arguments)
        if isinstance(guides, (list, tuple)) and not guides:
            del arguments[GUIDES_INPUT]
            return arguments
        where = f"Step '{self.name}': guides"
        refusal = guides_refusal(self.pipeline)
        if refusal:
            raise ValueError(f"{where}: {refusal}")
        if arguments.get("references") is not None:
            raise ValueError(
                f"{where} cannot be combined with references - ref2va lays out "
                f"its own conditioning; use guides on t2va or fl2va"
            )
        if not isinstance(guides, (list, tuple)):
            raise ValueError(
                f"{where} must be a list of {{video, frame}}, got {type(guides).__name__}"
            )
        if len(guides) > GUIDE_LIMIT:
            raise ValueError(
                f"{where} takes at most {GUIDE_LIMIT} clips, got {len(guides)}"
            )
        prepared = []
        for index, guide in enumerate(guides):
            if not isinstance(guide, dict):
                raise ValueError(
                    f"{where}[{index}] must be {{video, frame}}, got {type(guide).__name__}"
                )
            unknown = sorted(set(guide) - {"video", "frame", "audio"})
            if unknown:
                raise ValueError(
                    f"{where}[{index}] has unknown key(s) {unknown} - a guide is "
                    f"{{video, frame}}, with an optional 'audio'"
                )
            if "video" not in guide or "frame" not in guide:
                raise ValueError(f"{where}[{index}] needs both 'video' and 'frame'")
            problem = guide_frame_problem(guide["frame"])
            if problem:
                raise ValueError(f"{where}[{index}]: {problem}")
            with_audio = guide.get("audio", False)
            if not isinstance(with_audio, bool):
                raise ValueError(
                    f"{where}[{index}]: 'audio' must be true or false, got "
                    f"{type(with_audio).__name__}"
                )
            video = guide["video"]
            if isinstance(video, str) or (isinstance(video, dict) and with_audio):
                # Read with its soundtrack only when the guide holds it
                video = (
                    load_audio_video(video, self.base_dir)
                    if with_audio
                    else fetch_video(video, self.base_dir)
                )
            try:
                frames = guide_frames_array(video)
                audio, sample_rate = (
                    guide_audio_waveform(video) if with_audio else (None, None)
                )
            except ValueError as error:
                raise ValueError(f"{where}[{index}]: {error}") from error
            length = snap_guide_length(frames.shape[0])
            if length != frames.shape[0]:
                emit_warning(
                    f"{where}[{index}]: a guide clip encodes to whole latents only at "
                    f"1, 5 or 17m + 5 frames, so its {frames.shape[0]} frames are cut "
                    f"to the first {length}"
                )
                frames = frames[:length]
            prepared.append(
                {
                    "video": frames,
                    "frame": guide["frame"],
                    "audio": audio,
                    "sample_rate": sample_rate,
                }
            )
        arguments[GUIDES_INPUT] = prepared
        return arguments

    def _check_refine(self, arguments):
        """Refuse a `refine_strength` the H3 refine block cannot run, before the
        call (dw/pipeline_processors/h3_blocks.py).

        Raises:
            ValueError: If this pipeline cannot refine, the strength or step count
                is out of range, or the step passes no `latents` or `hold_audio`
        """
        strength = arguments.get(REFINE_STRENGTH_INPUT)
        if strength is None:
            return
        where = f"Step '{self.name}': refine_strength"
        if not refines(self.pipeline):
            raise ValueError(
                f"{where} is a MiniMax-H3 argument (t2va, fl2va or ref2va), and "
                f"{type(self.pipeline).__name__} cannot refine"
            )
        problems = refine_problems(arguments)
        if problems:
            raise ValueError(f"Step '{self.name}': {problems[0]}")

    def _takes_step_callback(self):
        """Whether this pipeline names `callback_on_step_end` in its own
        signature. Only a pipeline that names the parameter explicitly gets
        one - a **kwargs signature is no promise the pipeline honors it, and
        a ModularPipeline (H3, LTX-2, Qwen-Image) has no such parameter at
        all, which is why it needs the progress-bar route instead."""
        try:
            parameters = inspect.signature(self.pipeline.__call__).parameters
        except (TypeError, ValueError):
            return False
        return "callback_on_step_end" in parameters

    def _with_step_callback(self, arguments):
        """Inject a callback_on_step_end that reports per-step progress to the
        active run context and raises when the run has been cancelled.

        Workflow JSON cannot express a callable, so this is the only way a
        diffusion-step callback ever reaches a pipeline call.
        """
        if not self._takes_step_callback():
            return arguments

        run_context = get_context()
        num_steps = arguments.get("num_inference_steps", None)

        def on_step_end(pipe, step_index, timestep, callback_kwargs):
            total = getattr(pipe, "_num_timesteps", None) or num_steps
            run_context.emit("pipeline_step", step=step_index + 1, total_steps=total)
            # Past the last step the pipeline still has to decode the latents,
            # which on video is minutes with the bar sitting at 100%
            if total is not None and step_index + 1 >= total:
                emit_phase("decoding")
            run_context.check_cancelled()
            return callback_kwargs

        return {**arguments, "callback_on_step_end": on_step_end}

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

                device = resolve_device(
                    component_configuration.get("device", default_device)
                )

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


def warn_if_safety_checker_blanked(output):
    """Say so when a safety checker replaced a generated image with a black one.

    Stable Diffusion 1.5's checker false-positives readily, and it returns a
    solid black image rather than an error. Run to run that reads as the seed
    having no effect - the same result every time - so the reason belongs
    where the identical images do: the run's warnings, which is the only
    place a consumer of a `succeeded` job would ever see it.

    Args:
        output: The pipeline output, which may carry nsfw_content_detected
    """
    flags = getattr(output, "nsfw_content_detected", None)
    if not flags:
        return

    blanked = sum(1 for flag in flags if flag)
    if blanked:
        # emit_warning rather than logger.warning: a blanked image is a
        # succeeded job whose file is solid black, and a consumer over the
        # API or MCP sees the job's warnings list and nothing else - the log
        # line never reaches the one party that cannot tell the picture apart
        # from a rendered one (#133)
        emit_warning(
            f"The safety checker blanked {blanked} of {len(flags)} generated "
            "images - they are solid black, and no seed will change that. "
            "Pass 'safety_checker': null in from_pretrained_arguments to "
            "load the pipeline without it.",
            kind="safety_checker_blanked",
            blanked=blanked,
            images=len(flags),
        )


# Where an audio pipeline's components record the rate they generate at, in
# the order they are tried. LTX-2's vocoder names it output_sampling_rate,
# AudioLDM2's vocoder and StableAudio's VAE name it sampling_rate, and
# Kandinsky 6's MMAudioVAE names it sample_rate (its vocoder carries none).
# A vocoder's rate comes first: LTX-2's audio VAE works below it
_SAMPLE_RATE_SOURCES = (
    ("vocoder", "output_sampling_rate"),
    ("vocoder", "sampling_rate"),
    ("audio_vae", "sample_rate"),
    ("vae", "sampling_rate"),
)


def _component_sample_rate(pipeline, has_audios):
    for component_name, attribute in _SAMPLE_RATE_SOURCES:
        if component_name == "vae" and not has_audios:
            # A video VAE's config is not where an audio rate lives - only an
            # audio-only pipeline's VAE (StableAudio) reports one there
            continue
        config = getattr(getattr(pipeline, component_name, None), "config", None)
        sample_rate = getattr(config, attribute, None)
        if sample_rate is not None:
            return sample_rate
    return None


def attach_audio_sample_rate(pipeline, output):
    """Record the generating component's sample rate on an output that carries audio.

    Pipelines that generate audio - with their video (LTX-2, on `.audio`) or by
    itself (AudioLDM2, StableAudio, on `.audios`) - return the waveform without
    its sample rate; only the vocoder or VAE that produced it knows that. Saving
    needs the rate, so it travels with the output, and get_artifact_list wraps
    `.audios` items in an AudioTrack that carries it.

    Args:
        pipeline: The pipeline that produced the output
        output: The pipeline output
    """
    has_audio = getattr(output, "audio", None) is not None
    has_audios = getattr(output, "audios", None) is not None
    if not (has_audio or has_audios):
        return

    sample_rate = _component_sample_rate(pipeline, has_audios)
    if sample_rate is None:
        logger.warning(
            "Pipeline generated audio but no component reports its sample rate - "
            "set 'sample_rate' (audio results) or 'audio_sample_rate' (muxed video) "
            "in the step result to save it correctly"
        )
        return

    logger.debug(f"Generated audio has a sample rate of {sample_rate}Hz")
    output.audio_sample_rate = sample_rate


def _diagnose_image_crf_error(error, arguments):
    """Rewrite diffusers' `image_crf` re-compression error into one that
    names the actual mismatch, when it is one: an argument the pipeline
    expects as a still image (`PIL.Image.Image`) instead holds an
    `AudioVideo` - a video where an image was required. Diffusers' own
    message suggests `image_crf=0`, which does not fix anything, since the
    argument still receives a video; the caller needs to derive a still
    first. Returns None when the error is something else, so the caller
    re-raises unchanged.
    """
    if not isinstance(error, (ValueError, TypeError)):
        return None
    if "image_crf" not in str(error) or "re-compression requires" not in str(error):
        return None

    offending = [
        name for name, value in arguments.items() if isinstance(value, AudioVideo)
    ]
    if not offending:
        return None

    names = ", ".join(sorted(offending))
    return ValueError(
        f"Argument(s) {names} expected an image but received a video. Derive a "
        f"still from it first - `get_last_frame` or `video_frames` - and pass "
        f"that instead."
    )
