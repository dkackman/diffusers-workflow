import torch
import copy
from .quantization import get_quantization_configuration


class Pipeline:
    def __init__(self, pipeline_definition, default_seed, pipeline=None):
        self.pipeline_definition = pipeline_definition
        self.default_seed = default_seed
        self.pipeline = pipeline

    @property
    def configuration(self):
        return self.pipeline_definition.get("configuration", {})

    @property
    def model_name(self):
        return self.from_pretrained_arguments.get("model_name", "")

    @property
    def from_pretrained_arguments(self):
        return self.pipeline_definition.get("from_pretrained_arguments", {})

    @property
    def argument_template(self):
        return self.pipeline_definition["arguments"]

    def load(self, device_identifier, shared_components):
        from_pretrained_arguments = self.from_pretrained_arguments

        # grab any previously shared components and put them into from_pretrained_arguments
        # if a pipeline asks for both a shared component and specifies a configuration for
        # that component, the configuration will take precedence
        for reused_component_name in self.pipeline_definition.get(
            "reused_components", []
        ):
            from_pretrained_arguments[reused_component_name] = shared_components[
                reused_component_name
            ]

        for optional_component_name in [
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
        ]:
            self.load_optional_component(
                optional_component_name, from_pretrained_arguments, device_identifier
            )

        # load and configure the pipeline
        pipeline = load_and_configure_pipeline(
            self.configuration,
            from_pretrained_arguments,
            device_identifier,
        )

        # load and configure any custom scheduler
        load_and_configure_scheduler(
            self.pipeline_definition.get("scheduler", None), pipeline
        )

        # store any shared pipeline components for future use by other steps
        for shared_component_name in self.pipeline_definition.get(
            "shared_components", []
        ):
            shared_components[shared_component_name] = getattr(
                pipeline, shared_component_name
            )

        # load loras and fuse them into the pipeline
        load_loras(self.pipeline_definition.get("loras", []), pipeline)

        # load ip adapter
        load_ip_adapter(self.pipeline_definition.get("ip_adapter", None), pipeline)

        # create a generator that will be used by the pipeline
        if not "no_generator" in self.configuration:
            self.argument_template["generator"] = torch.Generator(
                device_identifier
            ).manual_seed(self.pipeline_definition.get("seed", self.default_seed))

        self.pipeline = pipeline

    @torch.inference_mode()
    def run(self, arguments):
        if self.pipeline is None:
            raise ValueError(
                "Pipeline has not been initialized. Call load(device_identifier, shared_components) first."
            )

        # if it's an inversion step run the pipeline with the invert method
        if self.configuration.get("inversion", False):
            # this is very specific to the FluxRFInversion pipeline - think about generalizing
            invert_arguments = copy.deepcopy(arguments)
            invert_arguments.pop("generator", None)
            inverted_latents, image_latents, latent_image_ids = self.pipeline.invert(
                **invert_arguments
            )
            return {
                "inverted_latents": inverted_latents,
                "image_latents": image_latents,
                "latent_image_ids": latent_image_ids,
            }

        # run the pipeline
        return self.pipeline(**arguments)

    def load_optional_component(
        self, component_name, from_pretrained_arguments, device_identifier
    ):
        component = load_and_configure_component(
            self.pipeline_definition.get(component_name, None),
            component_name,
            device_identifier,
        )
        if component is not None:
            from_pretrained_arguments[component_name] = component


def load_loras(loras, pipeline):
    for lora in loras:
        model_name = lora.pop("model_name", None)
        print(f"Loading lora {model_name}...")
        scale = lora.pop("scale", None)
        pipeline.load_lora_weights(model_name, **lora)
        if scale is not None:
            pipeline.fuse_lora(lora_scale=scale)


def load_ip_adapter(ip_adapter_definition, pipeline):
    if ip_adapter_definition is not None:
        model_name = ip_adapter_definition.pop("model_name")
        print(f"Loading ip adapter {model_name}...")
        scale = ip_adapter_definition.pop("scale", None)
        pipeline.load_ip_adapter(model_name, **ip_adapter_definition)
        if scale is not None:
            pipeline.set_ip_adapter_scale(scale)


def load_and_configure_scheduler(scheduler_definition, pipeline):
    if scheduler_definition is not None:
        scheduler_configuration = scheduler_definition.get("configuration", None)
        from_config_args = scheduler_definition.get("from_config_args", {})
        scheduler_type = scheduler_configuration.get("scheduler_type", None)
        print(f"Loading scheduler {scheduler_type}...")

        pipeline.scheduler = scheduler_type.from_config(
            pipeline.scheduler.config, **from_config_args
        )


def load_and_configure_component(
    component_definition, component_name, device_identifier
):
    if component_definition is not None:
        print(f"Loading {component_name}...")
        component_configuration = component_definition["configuration"]
        component_from_pretrained_arguments = component_definition[
            "from_pretrained_arguments"
        ]

        quantization_configuration = get_quantization_configuration(
            component_definition
        )
        if quantization_configuration is not None:
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
    # load the pipeline
    pipeline_type = configuration["pipeline_type"]
    pipeline = None
    if "model_name" in from_pretrained_arguments:
        model_name = from_pretrained_arguments.pop("model_name")
        print(f"Loading pipeline {model_name}...")

        pipeline = pipeline_type.from_pretrained(
            model_name, **from_pretrained_arguments
        )
    else:
        from_single_file = from_pretrained_arguments.pop("from_single_file")
        print(f"Loading pipeline from {from_single_file}...")
        pipeline = pipeline_type.from_single_file(
            from_single_file, **from_pretrained_arguments
        )

    do_not_send_to_device = configuration.get("do_not_send_to_device", False)
    offload = configuration.get("offload", None)
    if offload == "model":
        pipeline.enable_model_cpu_offload()
    elif offload == "sequential":
        pipeline.enable_sequential_cpu_offload()
    elif hasattr(pipeline, "to") and not do_not_send_to_device:
        pipeline = pipeline.to(device_identifier)

    vae = configuration.get("vae", {})
    if vae.get("enable_slicing", False):
        pipeline.vae.enable_slicing()
    if vae.get("enable_tiling", False):
        pipeline.vae.enable_tiling()
    if vae.get("set_memory_format", False):
        pipeline.vae.to(memory_format=torch.channels_last)

    unet = configuration.get("unet", {})
    if unet.get("enable_forward_chunking", False):
        pipeline.unet.enable_forward_chunking()
    if unet.get("set_memory_format", False):
        pipeline.unet.to(memory_format=torch.channels_last)

    return pipeline
