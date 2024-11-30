from .pipeline_processors.pipeline import Pipeline
from .image_processors.processor_dispatch import process_image
from .tasks.task import Task

class Step:
    def __init__(self, step_definition, default_seed):
        self.step_definition = step_definition
        self.step_definition["seed"] = self.step_definition.get("seed", default_seed)

    def run(self, intermediate_results, shared_components):
        try:
            name = self.step_definition["name"]
            print(f"Running step {name}...")
    
            results = []
            # run all the preprocessors
            for preprocessor in self.step_definition.get("preprocessors", []) :          
                preprocessor_output = process_image(preprocessor["image"], preprocessor["name"], "cuda", preprocessor.get("arguments", {}))
                intermediate_results[preprocessor["result_name"]] = preprocessor_output
            
            if "pipeline" in self.step_definition:
                pipeline = Pipeline(self.step_definition["pipeline"])
                step_result_list = pipeline.run("cuda", intermediate_results, shared_components)
                results.extend(step_result_list)  
            else:
                task = Task(self.step_definition["task"])
                task_result_list = task.run()
                results.extend(task_result_list)  

            # run all the postprocessors 
            for postprocessor in self.step_definition.get("postprocessors", []) :          
                postprocessor_output = process_image(postprocessor["image"], postprocessor["name"], "cuda", postprocessor.get("arguments", {}))
                intermediate_results[postprocessor["result_name"]] = postprocessor_output

        except Exception as e:
            print(f"Error running step {self.step_definition.get('name', 'unknown')}")
            print(e)
