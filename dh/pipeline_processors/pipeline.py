import torch
import copy
import logging
from .quantization import get_quantization_configuration

logger = logging.getLogger("dh")


class Pipeline:
    """
    Manages pipeline initialization, configuration, and execution.
    Handles loading of models, schedulers, and adapters.
    """

    def __init__(
        self, pipeline_definition, default_seed, device_identifier, pipeline=None
    ):
        """
        Initialize pipeline with configuration and device settings.

        Args:
            pipeline_definition: Dictionary containing pipeline configuration
            default_seed: Seed value for reproducibility
            device_identifier: Device to run pipeline on (e.g., 'cuda')
            pipeline: Optional existing pipeline to use
        """
        self.pipeline_definition = pipeline_definition
        self.default_seed = default_seed
        self.device_identifier = device_identifier
        self.pipeline = pipeline
        logger.debug(f"Initialized pipeline with device: {device_identifier}")

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

    def populate_from_pretrained_arguments(self, device_identifier, shared_components):
        """
        Prepare arguments for pipeline creation, including shared components.

        Args:
            device_identifier: Device to run pipeline on
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

        # Load optional components (controlnet, vae, unet, etc.)
        optional_components = [
            "controlnet",
            "transformer",
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
            "model",
        ]

        for component_name in optional_components:
            self.load_optional_component(
                component_name, from_pretrained_arguments, device_identifier
            )

        return from_pretrained_arguments

    def load(self, shared_components):
        """
        Load and configure the pipeline with all components.

        Args:
            shared_components: Dictionary of components shared between pipelines
        """
        logger.debug(f"Loading pipeline: {self.name}")

        # Prepare arguments and load pipeline
        from_pretrained_arguments = self.populate_from_pretrained_arguments(
            self.device_identifier, shared_components
        )

        # Load and configure the main pipeline
        self.pipeline = load_and_configure_pipeline(
            self.configuration,
            from_pretrained_arguments,
            self.device_identifier,
        )

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

        # Set up random generator if needed
        if not "no_generator" in self.configuration:
            logger.debug("Setting up random generator")
            self.argument_template["generator"] = torch.Generator(
                self.device_identifier
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

            # Run standard pipeline
            logger.debug("Running standard pipeline")
            output = self.pipeline(**arguments)

            # Ensure output is on correct device
            if hasattr(output, "to"):
                output = output.to(self.device_identifier)

            return output

        except Exception as e:
            logger.error(f"Error running pipeline: {str(e)}", exc_info=True)
            raise

    def load_optional_component(
        self, component_name, from_pretrained_arguments, device_identifier
    ):
        """Load an optional component if specified in pipeline definition."""
        component = load_and_configure_component(
            self.pipeline_definition.get(component_name, None),
            component_name,
            device_identifier,
        )
        if component is not None:
            logger.debug(f"Loaded optional component: {component_name}")
            from_pretrained_arguments[component_name] = component


def load_loras(loras, pipeline):
    """Load and configure LoRA models."""
    for lora in loras:
        model_name = lora.pop("model_name", None)
        logger.info(f"Loading LoRA: {model_name}")
        scale = lora.pop("scale", None)
        pipeline.load_lora_weights(model_name, **lora)
        if scale is not None:
            pipeline.fuse_lora(lora_scale=scale)


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


def load_and_configure_component(
    component_definition, component_name, device_identifier
):
    """Load and configure a pipeline component."""
    if component_definition is not None:
        logger.info(f"Loading component: {component_name}")
        component_configuration = component_definition["configuration"]
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

        return load_and_configure_pipeline(
            component_configuration,
            component_from_pretrained_arguments,
            device_identifier,
        )

    return None


def load_and_configure_pipeline(
    configuration, from_pretrained_arguments, device_identifier
):
    """Load and configure a pipeline or component."""
    pipeline_type = configuration["pipeline_type"]
    pipeline = None

    try:
        # Load from model name
        if "model_name" in from_pretrained_arguments:
            model_name = from_pretrained_arguments.pop("model_name")
            logger.info(f"Loading pipeline from model: {model_name}")
            pipeline = pipeline_type.from_pretrained(
                model_name, **from_pretrained_arguments
            )

        # Load from single file
        elif "from_single_file" in from_pretrained_arguments:
            from_single_file = from_pretrained_arguments.pop("from_single_file")
            logger.info(f"Loading pipeline from single file: {from_single_file}")
            pipeline = pipeline_type.from_single_file(
                from_single_file, **from_pretrained_arguments
            )

        # Create new pipeline
        else:
            logger.info("Creating new pipeline")
            pipeline = pipeline_type(**from_pretrained_arguments)

        # Configure pipeline device settings
        do_not_send_to_device = configuration.get("do_not_send_to_device", False)
        offload = configuration.get("offload", None)

        if offload == "model":
            logger.debug("Enabling model CPU offload")
            pipeline.enable_model_cpu_offload()
        elif offload == "sequential":
            logger.debug("Enabling sequential CPU offload")
            for component_name in configuration.get("exclude_from_cpu_offload", []):
                logger.debug(f"Excluding {component_name} from CPU offload")
                pipeline._exclude_from_cpu_offload.append(component_name)
            pipeline.enable_sequential_cpu_offload()
        elif hasattr(pipeline, "to") and not do_not_send_to_device:
            logger.debug(f"Moving pipeline to device: {device_identifier}")
            pipeline = pipeline.to(device_identifier)

        # Configure VAE settings
        vae = configuration.get("vae", {})
        if vae.get("enable_slicing", False):
            logger.debug("Enabling VAE slicing")
            pipeline.vae.enable_slicing()
        if vae.get("enable_tiling", False):
            logger.debug("Enabling VAE tiling")
            pipeline.vae.enable_tiling()
        if vae.get("set_memory_format", False):
            logger.debug("Setting VAE memory format")
            pipeline.vae.to(memory_format=torch.channels_last)

        # Configure UNet settings
        unet = configuration.get("unet", {})
        if unet.get("enable_forward_chunking", False):
            logger.debug("Enabling UNet forward chunking")
            pipeline.unet.enable_forward_chunking()
        if unet.get("set_memory_format", False):
            logger.debug("Setting UNet memory format")
            pipeline.unet.to(memory_format=torch.channels_last)

        return pipeline

    except Exception as e:
        logger.error(f"Error loading pipeline: {str(e)}", exc_info=True)
        raise
