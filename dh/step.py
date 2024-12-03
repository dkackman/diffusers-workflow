from .pipeline_processors.pipeline import Pipeline
from .tasks.task import Task
from .result import Result

class Step:
    def __init__(self, step_definition, default_seed):
        self.step_definition = step_definition
        self.step_definition["seed"] = self.step_definition.get("seed", default_seed)

    def run(self, previous_results, shared_components):
        try:
            name = self.step_definition["name"]
            print(f"Running step {name}...")
    
            result = None
            result_definition = self.step_definition.get("result", {})
            if "pipeline" in self.step_definition:
                pipeline = Pipeline(self.step_definition["pipeline"])
                pipeline.load("cuda", shared_components)
                result = Result(pipeline.run(previous_results), result_definition)

            else:
                task = Task(self.step_definition["task"])
                print(f"Running task {task.command}...")     
                result = Result(task.run("cuda"), result_definition)

            return result
        
        except Exception as e:
            print(f"Error running step {self.step_definition.get('name', 'unknown')}")
            print(e)

    @property
    def name(self):
        return self.step_definition.get("name", "unknown")