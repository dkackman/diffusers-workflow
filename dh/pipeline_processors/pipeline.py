import torch
from diffusers import BitsAndBytesConfig
from .quantization import quantize
from ..result import Result

class Pipeline:
    def __init__(self, pipeline_definition):
        self.pipeline_definition = pipeline_definition

    @torch.inference_mode()
    def run(self, device_identifier, intermediate_results, shared_components):
        configuration = self.pipeline_definition.get("configuration", None)
        from_pretrained_arguments = self.pipeline_definition.get("from_pretrained_arguments", None)
        
        # grab any previously shared components and put them into from_pretrained_arguments
        # if a pipeline asks for both a shared component and specifies a configuration for 
        # that component, the configuration will take precedence
        for reused_component_name in self.pipeline_definition.get("reused_components", []):
            from_pretrained_arguments[reused_component_name] = shared_components[reused_component_name]

        for optional_component_name in ["controlnet", "transformer", "vae", "unet", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"]:
            self.load_optional_component(optional_component_name, from_pretrained_arguments, device_identifier)

        # load and configure the pipeline
        pipeline = load_and_configure_pipeline(configuration, from_pretrained_arguments, device_identifier)

        # load and configure any custom scheduler
        load_and_configure_scheduler(self.pipeline_definition.get("scheduler", None), pipeline)
        
        # store any shared pipeline components for future use by other steps
        for shared_component_name in self.pipeline_definition.get("shared_components", []):
            shared_components[shared_component_name] = getattr(pipeline, shared_component_name)

        # load loras and fuse them into the pipeline
        loras = self.pipeline_definition.get("loras", [])        
        for lora in loras:
            default_lora_scale = 0.7 / len(loras) # default to equally distributing lora weights
            lora_name = lora.pop("lora_name", None)
            print(f"Loading lora {lora_name}...")
            lora_scale = lora.pop("lora_scale", default_lora_scale)
            pipeline.load_lora_weights(lora_name, **lora)
            pipeline.fuse_lora(lora_scale=lora_scale)

        # create a generator that will be used by the pipeline
        default_generator = torch.Generator(device_identifier).manual_seed(self.pipeline_definition.get("seed", torch.seed()))

        # prepare and run pipeline
        arguments = self.pipeline_definition.get("arguments", {})

        # each iteration can use its own seed
        if not configuration.get("no_generator", False):
            arguments["generator"] = torch.Generator(device_identifier).manual_seed(self.pipeline_definition["seed"]) if "seed" in self.pipeline_definition else default_generator

        # if there are intermediate results requested, add them to the iteration
        intermediate_result_names = self.pipeline_definition.get("intermediate_results", {})
        for k, v in intermediate_result_names.items():
            if "." in v:
                # this is named property of the captured result
                # parse and get that property
                parts = v.split(".")
                arguments[k] = intermediate_results[parts[0]][parts[1]]
            else:
                arguments[k] = intermediate_results[v]

        # run the pipeline
        pipeline_output = pipeline(**arguments)
        result = Result(pipeline_output, self.pipeline_definition.get("result", {}))
        #
        # the presence of this key indicates that the output should be
        # stored as an intermediate result, not returned as an output
        #
        if "capture_intermediate_results" in self.pipeline_definition:
            intermediate_result_names = self.pipeline_definition["capture_intermediate_results"]
            for k, v in intermediate_result_names.items():
                raw_result = result.get_raw_result()
                # output can have different shapes, so we need to check if the key is present
                # if it is, capture that property of the result, otherwise just capture the result itself
                intermediate_results[k] = raw_result.get(v, result.get_primary_output())

        return result
    

    def load_optional_component(self, component_name, from_pretrained_arguments, device_identifier):
        # load the component if specified
        component = load_and_configure_component(self.pipeline_definition.get(component_name, None), component_name, device_identifier)
        if component is not None:
            from_pretrained_arguments[component_name] = component

            
def load_and_configure_scheduler(scheduler_definition, pipeline):
    if scheduler_definition is not None:
        scheduler_configuration = scheduler_definition.get("configuration", None)        
        from_config_args = scheduler_definition.get("from_config_args", {})                
        scheduler_type = scheduler_configuration.get("scheduler_type", None)
        print(f"Loading scheduler {scheduler_type}...")

        pipeline.scheduler = scheduler_type.from_config(pipeline.scheduler.config, **from_config_args)


def load_and_configure_component(component_definition, component_name, device_identifier):
    if component_definition is not None:
        print(f"Loading {component_name}...")
        component_configuration = component_definition["configuration"]
        component_from_pretrained_arguments = component_definition["from_pretrained_arguments"]

        # TODO - does this need to be generalized?
        if component_name != "controlnet":
            component_from_pretrained_arguments["subfolder"] = component_name
        component = load_and_configure_pipeline(component_configuration, component_from_pretrained_arguments, device_identifier)
        quantize(component, component_definition.get("quantization", None))

        return component
    
    return None


def load_and_configure_pipeline(configuration, from_pretrained_arguments, device_identifier):
    bits_and_bytes_configuration = configuration.get("bits_and_bytes_configuration", None)
    if bits_and_bytes_configuration is not None:
        print("Loading bits and bytes configuration")
        from_pretrained_arguments["quantization_config"] = BitsAndBytesConfig(**bits_and_bytes_configuration)

    # load the pipeline
    pipeline_type = configuration.get("pipeline_type", None)
    model_name = from_pretrained_arguments.pop("model_name", None)  
    print(f"Loading pipeline {model_name}...")

    pipeline = pipeline_type.from_pretrained(model_name, **from_pretrained_arguments)

    offload = configuration.get("offload", None)
    if offload == "full":
         pipeline.enable_model_cpu_offload()
    elif offload == "sequential":
        pipeline.enable_sequential_cpu_offload()
    else:
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
