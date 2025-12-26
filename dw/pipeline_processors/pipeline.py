import torch
import copy
import logging
from .quantization import get_quantization_configuration
from .remote import remote_text_encoder
from diffusers import attention_backend

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
    "model",
]


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
            device: Device to run pipeline on (e.g., 'cuda', 'mps', 'cpu')
            pipeline: Optional existing pipeline to use
        """
        self.pipeline_definition = pipeline_definition
        self.default_seed = default_seed
        self.device = device
        self.pipeline = pipeline
        logger.debug(f"Initialized pipeline with device: {device}")

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

        # Load optional components (controlnet, vae, unet, etc.)
        for component_name in optional_component_names:
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

        if self.configuration.get("enable_attention_slicing", False):
            logger.debug("Enabling attention slicing for pipeline")
            self.pipeline.enable_attention_slicing()

        if self.configuration.get("xformers_memory_efficient_attention", False):
            logger.debug(
                "Enabling attention xformers memory efficient attention for pipeline"
            )
            self.pipeline.enable_xformers_memory_efficient_attention()

        # configure components that are not shared
        self.configure_loaded_components()

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

            # Run standard pipeline
            logger.debug("Running standard pipeline")
            attn_backend = self.configuration.get("attention_backend", None)
            if attn_backend is None:
                output = self.pipeline(**arguments)
            else:
                logger.debug(f"Using attention backend: {attn_backend}")
                with attention_backend(attn_backend):
                    output = self.pipeline(**arguments)

            # Ensure output is on correct device
            if hasattr(output, "to"):
                logger.debug(f"Moving output to {self.device}")
                output = output.to(self.device)

            return output

        except (KeyError, ValueError, TypeError) as e:
            # Missing arguments, invalid configuration, type mismatches
            logger.error(f"Configuration error running pipeline: {e}", exc_info=True)
            raise
        except (OSError, IOError) as e:
            # File operations, resource loading errors
            logger.error(f"I/O error running pipeline: {e}", exc_info=True)
            raise
        except RuntimeError as e:
            # CUDA OOM, model inference failures, torch errors
            logger.error(f"Runtime error running pipeline: {e}", exc_info=True)
            raise
        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(
                f"Unexpected error ({type(e).__name__}) running pipeline: {e}",
                exc_info=True,
            )
            raise

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
        if vae.get("set_memory_format", False):
            logger.debug("Setting VAE memory format")
            self.pipeline.vae.to(memory_format=torch.channels_last)

        # Configure UNet settings
        unet = self.configuration.get("unet", {})
        if unet.get("enable_forward_chunking", False):
            logger.debug("Enabling UNet forward chunking")
            self.pipeline.unet.enable_forward_chunking()
        if unet.get("set_memory_format", False):
            logger.debug("Setting UNet memory format")
            self.pipeline.unet.to(memory_format=torch.channels_last)

        # Configure UNet attention processor (mutually exclusive options)
        if unet.get("attn_processor_type", None) is not None:
            logger.debug("Enabling UNet custom attention processor")
            attn_processor = unet["attn_processor_type"]()
            self.pipeline.unet.set_attn_processor(attn_processor)
        elif unet.get("enable_xformers_memory_efficient_attention", False):
            logger.info("Enabling xFormers memory efficient attention for UNet")
            if hasattr(
                self.pipeline.unet, "enable_xformers_memory_efficient_attention"
            ):
                self.pipeline.unet.enable_xformers_memory_efficient_attention()
            else:
                logger.warning(
                    "UNet does not support xFormers memory efficient attention"
                )

        # Configure transformer settings
        transformer = self.configuration.get("transformer", {})
        if transformer.get("attn_processor_type", None) is not None:
            logger.debug("Enabling transformer custom attention processor")
            attn_processor = transformer["attn_processor_type"]()
            self.pipeline.transformer.set_attn_processor(attn_processor)
        elif transformer.get("enable_xformers", False):
            logger.info("Enabling xFormers memory efficient attention for transformer")
            if hasattr(
                self.pipeline.transformer, "enable_xformers_memory_efficient_attention"
            ):
                self.pipeline.transformer.enable_xformers_memory_efficient_attention()
            else:
                logger.warning(
                    "Transformer does not support xFormers memory efficient attention"
                )

        # configure optional components
        for component_name in optional_component_names:
            component_configuration = self.configuration.get(component_name, None)
            component = getattr(self.pipeline, component_name, None)
            if component_configuration is not None and component is not None:
                logger.debug(f"Configuring optional component: {component_name}")
                torch_dtype = component_configuration.get("torch_dtype", None)
                if torch_dtype is not None:
                    logger.debug(f"Setting {component_name} torch dtype: {torch_dtype}")
                    component.to(torch_dtype)


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


def load_component(component_name, configuration, from_pretrained_arguments, device):
    """Load and configure a pipeline or component."""
    component_type = configuration["component_type"]
    component = None

    try:
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

        # Configure component device settings
        do_not_send_to_device = configuration.get("do_not_send_to_device", False)
        offload = configuration.get("offload", None)

        if offload == "model":
            logger.debug("Enabling model CPU offload")
            component.enable_model_cpu_offload()
        elif offload == "sequential":
            logger.debug("Enabling sequential CPU offload")
            for component_name in configuration.get("exclude_from_cpu_offload", []):
                logger.debug(f"Excluding {component_name} from CPU offload")
                component._exclude_from_cpu_offload.append(component_name)
            component.enable_sequential_cpu_offload()
        elif hasattr(component, "to") and not do_not_send_to_device:
            logger.debug(f"Moving {component_name} to device: {device}")
            component = component.to(device)

        return component

    except (KeyError, ValueError, TypeError) as e:
        # Missing configuration, invalid values, type mismatches
        logger.error(
            f"Configuration error loading {component_name}: {e}", exc_info=True
        )
        raise
    except (OSError, IOError) as e:
        # Model file not found, download failures, disk errors
        logger.error(f"I/O error loading {component_name}: {e}", exc_info=True)
        raise
    except RuntimeError as e:
        # CUDA OOM during model loading, incompatible model format
        logger.error(f"Runtime error loading {component_name}: {e}", exc_info=True)
        raise
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(
            f"Unexpected error ({type(e).__name__}) loading {component_name}: {e}",
            exc_info=True,
        )
        raise
