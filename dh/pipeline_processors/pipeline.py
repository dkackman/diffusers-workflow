import torch
from diffusers import BitsAndBytesConfig
from ..pre_processors.controlnet import preprocess_image
from .quantization import quantize
from .result import Result

@torch.inference_mode()
def run_step(step_definition, device_identifier, intermediate_results, shared_components):
    pipeline_definition = step_definition.get("pipeline", None)
    configuration = pipeline_definition.get("configuration", None)
    from_pretrained_arguments = pipeline_definition.get("from_pretrained_arguments", None)
    
    # run all the prerpocessors first
    for preprocessor in pipeline_definition.get("preprocessors", []) :          
        preprocessor_output = preprocess_image(preprocessor["image"], preprocessor["name"], device_identifier, preprocessor.get("arguments", {}))
        intermediate_results[preprocessor["result_name"]] = preprocessor_output
    
    # grab any previously shared components and put them into from_pretrained_arguments
    # if a pipeline asks for both a shared component and specifies a configuration for 
    # that component, the configuration will take precedence
    for reused_component_name in pipeline_definition.get("reused_components", []):
        from_pretrained_arguments[reused_component_name] = shared_components[reused_component_name]

    for optional_component_name in ["controlnet", "transformer", "vae", "unet", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"]:
        load_optional_component(optional_component_name, pipeline_definition, from_pretrained_arguments, device_identifier)

    # load and configure the pipeline
    pipeline = load_and_configure_pipeline(configuration, from_pretrained_arguments, device_identifier)

    # load and configure any custom scheduler
    load_and_configure_scheduler(pipeline_definition.get("scheduler", None), pipeline)
    
    # store any shared pipeline components for future use by other steps
    for shared_component_name in pipeline_definition.get("shared_components", []):
        shared_components[shared_component_name] = getattr(pipeline, shared_component_name)

    # load loras and fuse them into the pipeline
    loras = pipeline_definition.get("loras", [])        
    for lora in loras:
        default_lora_scale = 0.7 / len(loras) # default to equally distributing lora weights
        lora_name = lora.pop("lora_name", None)
        print(f"Loading lora {lora_name}...")
        lora_scale = lora.pop("lora_scale", default_lora_scale)
        pipeline.load_lora_weights(lora_name, **lora)
        pipeline.fuse_lora(lora_scale=lora_scale)

    # create a generator that will be used by each iteration if they don't set their own seed
    default_generator = torch.Generator(device_identifier).manual_seed(step_definition.get("seed", torch.seed()))
    results = []

    # prepare and run pipeline iterations
    for iteration in step_definition.get("iterations", []):
        arguments = iteration.get("arguments", {})

        # each iteration can use its own seed
        if not configuration.get("no_generator", False):
            arguments["generator"] = torch.Generator(device_identifier).manual_seed(iteration["seed"]) if "seed" in iteration else default_generator

        # if there are intermediate results requested, add them to the iteration
        intermediate_result_names = iteration.get("intermediate_results", {})
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
        result = Result(pipeline_output, iteration)
        results.append(result)
        #
        # the presence of this key indicates that the output should be
        # stored as an intermediate result, not returned as an output
        #
        # NOTE - the capture key can be used to differentiate between different
        #        iterations of the same pipeline. It is not required.
        #
        if "capture_intermediate_results" in iteration:
            intermediate_result_names = iteration["capture_intermediate_results"]
            capture_key = iteration.get("capture_key", "")
            for k, v in intermediate_result_names.items():
                raw_result = result.get_raw_result()
                # output can have different shapes, so we need to check if the key is present
                # if it is, capture that property of the result, otherwise just capture the result itself
                intermediate_results[k + capture_key] = raw_result.get(v, result.get_primary_output())

    return results


def load_and_configure_scheduler(scheduler_definition, pipeline):
    if scheduler_definition is not None:
        scheduler_configuration = scheduler_definition.get("configuration", None)        
        from_config_args = scheduler_definition.get("from_config_args", {})                
        scheduler_type = scheduler_configuration.get("scheduler_type", None)
        print(f"Loading scheduler {scheduler_type}...")

        pipeline.scheduler = scheduler_type.from_config(pipeline.scheduler.config, **from_config_args)


def load_optional_component(component_name, pipeline_definition, from_pretrained_arguments, device_identifier):
    # load the component if specified
    component = load_and_configure_component(pipeline_definition.get(component_name, None), component_name, device_identifier)
    if component is not None:
        from_pretrained_arguments[component_name] = component


def load_and_configure_component(component_definition, component_name, device_identifier):
    if component_definition is not None:
        print(f"Loading {component_name}...")
        component_configuration = component_definition["configuration"]
        component_from_pretrained_arguments = component_definition["from_pretrained_arguments"]
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
