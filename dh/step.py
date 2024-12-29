from .result import Result
from .previous_results import get_iterations
import logging

logger = logging.getLogger("dh")


class Step:
    def __init__(self, step_definition, default_seed):
        self.step_definition = step_definition
        self.default_seed = self.step_definition.get("seed", default_seed)
        logger.debug(f"Initialized step: {self.name} with seed: {self.default_seed}")

    @property
    def name(self):
        return self.step_definition.get("name", "unknown")

    def run(self, previous_results, previous_pipelines, step_action):
        try:
            step_name = self.step_definition["name"]

            result = Result(self.step_definition.get("result", {}))

            logger.info(f"Running step {step_name}:{step_action.name}...")
            for arguments in get_iterations(
                step_action.argument_template, previous_results
            ):
                logger.debug(f"Running iteration with arguments: {arguments}")
                result.add_result(step_action.run(arguments, previous_pipelines))

            return result

        except Exception as e:
            logger.error(f"Error running step {self.name}: {str(e)}", exc_info=True)
            raise e
