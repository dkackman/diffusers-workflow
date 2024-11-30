from .pipeline_processors.pipeline import Pipeline
from .image_processors.image_processor import ImageProcessor
from .tasks.task import Task

class Step:
    def __init__(self, step_definition, default_seed):
        self.step_definition = step_definition
        self.step_definition["seed"] = self.step_definition.get("seed", default_seed)

    def run(self, previous_results, shared_components):
        try:
            name = self.step_definition["name"]
            print(f"Running step {name}...")
    
            result = None

            if "pipeline" in self.step_definition:
                pipeline = Pipeline(self.step_definition["pipeline"])
                result = pipeline.run("cuda", previous_results, shared_components)

            elif "image_processor" in self.step_definition:
                processor = ImageProcessor(self.step_definition["image_processor"])
                result = processor.run("cuda")

            else:
                task = Task(self.step_definition["task"])
                print(f"Running task {task['task_name']}...")     
                result = task.run()

            return result
        
        except Exception as e:
            print(f"Error running step {self.step_definition.get('name', 'unknown')}")
            print(e)

    @property
    def name(self):
        return self.step_definition.get("name", "unknown")