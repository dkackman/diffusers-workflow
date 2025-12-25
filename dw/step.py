import logging
from .result import Result
from .previous_results import get_iterations

logger = logging.getLogger("dw")


class Step:
    """
    Represents a single step in a workflow execution.
    Manages execution of pipelines, tasks, or sub-workflows with their configurations.
    """

    def __init__(self, step_definition, default_seed):
        """Initialize step with its configuration and seed value"""
        self.step_definition = step_definition

        # Get step-specific seed or use default if not specified
        self.default_seed = self.step_definition.get("seed", default_seed)
        logger.debug(f"Initialized step: {self.name} with seed: {self.default_seed}")

    @property
    def name(self):
        return self.step_definition.get("name", "unknown")

    def run(self, previous_results, previous_pipelines, step_action):
        """
        Execute the step's action with all possible argument combinations.

        Args:
            previous_results: Results from previous steps, used for argument generation
            previous_pipelines: Previously created pipelines that might be referenced
            step_action: The actual action to execute (Pipeline/Task/Workflow)
        """
        try:
            step_name = self.step_definition["name"]
            logger.debug(f"Starting execution of step: {step_name}")

            # Create result container with any special configuration from step definition
            # This handles how results should be saved/processed
            result = Result(self.step_definition.get("result", {}))

            # Log what type of action we're executing (Pipeline/Task/Workflow)
            action_type = type(step_action).__name__
            logger.info(f"Running {action_type} {step_name}:{step_action.name}...")

            # Get all possible argument combinations for this step
            # This expands any references to previous results into concrete values
            iterations = get_iterations(step_action.argument_template, previous_results)
            logger.debug(f"Generated {len(iterations)} argument combinations")

            # Execute the action for each set of arguments
            if not iterations:
                logger.warning(f"Step {step_name} has no iterations to execute")
                return result

            for i, arguments in enumerate(iterations, 1):
                logger.debug(
                    f"Running iteration {i}/{len(iterations)} with arguments: {arguments}"
                )

                try:
                    iteration_result = step_action.run(arguments, previous_pipelines)
                    result.add_result(iteration_result)

                except (KeyError, ValueError, TypeError) as e:
                    # Missing arguments, invalid values, type mismatches
                    logger.error(
                        f"Configuration error in iteration {i} of step {step_name}: {e}",
                        exc_info=True
                    )
                    raise
                except (OSError, IOError) as e:
                    # File operations, resource loading failures
                    logger.error(
                        f"I/O error in iteration {i} of step {step_name}: {e}",
                        exc_info=True
                    )
                    raise
                except RuntimeError as e:
                    # CUDA OOM, model loading failures, torch errors
                    logger.error(
                        f"Runtime error in iteration {i} of step {step_name}: {e}",
                        exc_info=True
                    )
                    raise
                except Exception as e:
                    # Catch-all for unexpected errors from pipeline/task/workflow execution
                    logger.error(
                        f"Unexpected error ({type(e).__name__}) in iteration {i} of step {step_name}: {e}",
                        exc_info=True
                    )
                    raise

            logger.debug(f"Successfully completed step: {step_name}")
            return result

        except (KeyError, ValueError, TypeError) as e:
            # Missing keys, invalid step configuration, type mismatches
            logger.error(
                f"Configuration error in step {self.name}: {e}",
                exc_info=True
            )
            raise
        except (OSError, IOError) as e:
            # File operations, resource loading errors
            logger.error(
                f"I/O error in step {self.name}: {e}",
                exc_info=True
            )
            raise
        except RuntimeError as e:
            # CUDA OOM, model loading failures, torch errors
            logger.error(
                f"Runtime error in step {self.name}: {e}",
                exc_info=True
            )
            raise
        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(
                f"Unexpected error ({type(e).__name__}) in step {self.name}: {e}",
                exc_info=True
            )
            raise
