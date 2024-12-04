from .pipeline_processors.pipeline import Pipeline
from .tasks.task import Task
from .result import Result

class Step:
    def __init__(self, step_definition, default_seed):
        self.step_definition = step_definition
        self.step_definition["seed"] = self.step_definition.get("seed", default_seed)

    def run(self, previous_results, shared_components):
        try:
            step_name = self.step_definition["name"]
    
            result = Result(self.step_definition.get("result", {}))
            if "pipeline" in self.step_definition:
                pipeline = Pipeline(self.step_definition["pipeline"])
                print(f"Running task {step_name}:{pipeline.model_name}...")   
                pipeline.load("cuda", shared_components)

                for arguments in pipeline.get_iterations(previous_results):                    
                    result.add_result(pipeline.run(arguments))

            else:
                task = Task(self.step_definition["task"])
                print(f"Running task {step_name}:{task.command}...")   

                for arguments in task.get_iterations(previous_results):
                    result.add_result(task.run("cuda", arguments))

            return result
        
        except Exception as e:
            print(f"Error running step {self.step_definition.get('name', 'unknown')}")
            print(e)
            raise e

    @property
    def name(self):
        return self.step_definition.get("name", "unknown")