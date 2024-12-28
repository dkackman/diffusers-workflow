from .result import Result
from .previous_results import get_iterations


class Step:
    def __init__(self, step_definition, default_seed):
        self.step_definition = step_definition
        self.default_seed = self.step_definition.get("seed", default_seed)

    @property
    def name(self):
        return self.step_definition.get("name", "unknown")

    def run(self, previous_results, previous_pipelines, step_action):
        try:
            step_name = self.step_definition["name"]

            result = Result(self.step_definition.get("result", {}))

            print(f"Running task {step_name}:{step_action.name}...")
            for arguments in get_iterations(
                step_action.argument_template, previous_results
            ):
                result.add_result(step_action.run(arguments, previous_pipelines))

            return result

        except Exception as e:
            print(f"Error running step {self.step_definition.get('name', 'unknown')}")
            print(e)
            raise e
