from .pipeline_processors.pipeline import Pipeline
from .pre_processors.controlnet import process_image


class Step:
    def __init__(self, data, default_seed):
        self.data = data
        self.data["seed"] = self.data.get("seed", default_seed)
        self._results = []

    def run(self, intermediate_results, shared_components):
        try:
            name = self.data["name"]
            print(f"Running step {name}...")
    
            # run all the preprocessors
            for preprocessor in self.data.get("preprocessors", []) :          
                preprocessor_output = process_image(preprocessor["image"], preprocessor["name"], "cuda", preprocessor.get("arguments", {}))
                intermediate_results[preprocessor["result_name"]] = preprocessor_output
            
            pipeline = Pipeline(self.data["pipeline"])
            step_result_list = pipeline.run("cuda", intermediate_results, shared_components)
            self._results.extend(step_result_list)  

            # run all the postprocessors 
            for postprocessor in self.data.get("postprocessors", []) :          
                postprocessor_output = process_image(postprocessor["image"], postprocessor["name"], "cuda", postprocessor.get("arguments", {}))
                intermediate_results[postprocessor["result_name"]] = postprocessor_output

        except Exception as e:
            print(f"Error running step {self.data.get('name', 'unknown')}")
            print(e)

    @property
    def results(self):
        return self._results