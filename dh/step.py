from .pipeline_processors.pipeline import Pipeline
from .tasks.task import Task
from .result import Result
from .previous_results import get_iterations


class Step:
    def __init__(self, step_definition, default_seed):
        self.step_definition = step_definition
        self.default_seed = self.step_definition.get("seed", default_seed)

    @property
    def name(self):
        return self.step_definition.get("name", "unknown")

    def run(self, previous_results, shared_components, previous_pipelines):
        try:
            step_name = self.step_definition["name"]

            result = Result(self.step_definition.get("result", {}))
            if "pipeline" in self.step_definition:
                pipeline = Pipeline(self.step_definition["pipeline"], self.default_seed)
                print(f"Running task {step_name}:{pipeline.model_name}...")
                pipeline.load("cuda", shared_components)
                previous_pipelines[step_name] = pipeline

                for arguments in get_iterations(
                    pipeline.argument_template, previous_results
                ):
                    result.add_result(pipeline.run(arguments, "cuda"))

            elif "pipeline_reference" in self.step_definition:
                pipeline_reference = self.step_definition["pipeline_reference"]
                previous_pipeline = previous_pipelines[
                    pipeline_reference["reference_name"]
                ]
                pipeline = Pipeline(
                    pipeline_reference, self.default_seed, previous_pipeline.pipeline
                )
                for arguments in get_iterations(
                    pipeline.argument_template, previous_results
                ):
                    result.add_result(pipeline.run(arguments))

            else:
                task = Task(self.step_definition["task"])
                print(f"Running task {step_name}:{task.command}...")

                for arguments in get_iterations(
                    task.argument_template, previous_results
                ):
                    result.add_result(task.run("cuda", arguments, previous_pipelines))

            return result

        except Exception as e:
            print(f"Error running step {self.step_definition.get('name', 'unknown')}")
            print(e)
            raise e
