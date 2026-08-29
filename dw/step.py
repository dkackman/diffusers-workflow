import logging
from .events import WorkflowCancelled, get_context
from .result import Result
from .previous_results import get_iterations, resolve_chain_prompts

logger = logging.getLogger("dw")


class Step:
    """
    Represents a single step in a workflow execution.
    Manages execution of pipelines, tasks, or sub-workflows with their configurations.
    """

    def __init__(self, step_definition, default_seed, workflow_definition=None):
        """Initialize step with its configuration and seed value.

        workflow_definition, when given, is the original (unsubstituted)
        definition of the workflow this step belongs to - embedded metadata
        carries it so a saved image can be reopened as the workflow that
        made it."""
        self.step_definition = step_definition
        self.workflow_definition = workflow_definition
        self.iteration = None

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

            # Collect metadata for embedding if enabled
            result_def = self.step_definition.get("result", {})
            if result_def.get("embed_metadata", False):
                metadata = self._collect_metadata()
                result.set_metadata(metadata)

            # Log what type of action we're executing (Pipeline/Task/Workflow)
            action_type = type(step_action).__name__
            logger.info(f"Running {action_type} {step_name}:{step_action.name}...")

            # A chained pipeline's per-segment prompts live outside the argument
            # template, so they are resolved here rather than by the pass below
            resolve_chain_prompts(step_action, previous_results)

            # Get all possible argument combinations for this step
            # This expands any references to previous results into concrete values
            iterations = get_iterations(step_action.argument_template, previous_results)
            logger.debug(f"Generated {len(iterations)} argument combinations")

            # Execute the action for each set of arguments
            if not iterations:
                logger.warning(f"Step {step_name} has no iterations to execute")
                return result

            run_context = get_context()
            for i, arguments in enumerate(iterations, 1):
                run_context.check_cancelled()
                logger.debug(
                    f"Running iteration {i}/{len(iterations)} with arguments: {arguments}"
                )
                run_context.emit(
                    "iteration_start",
                    step=step_name,
                    iteration=i,
                    total_iterations=len(iterations),
                )
                self.iteration = i
                iteration_result = step_action.run(arguments, previous_pipelines)
                result.add_result(iteration_result)

            logger.debug(f"Successfully completed step: {step_name}")
            return result

        except WorkflowCancelled:
            logger.info(f"Step {self.name} cancelled")
            raise
        except Exception as e:
            # One log line with the full traceback - callers decide handling.
            # (Configuration, I/O, runtime and unexpected errors all logged and
            # re-raised identically, so one clause replaces the old four)
            iteration = getattr(self, "iteration", None)
            where = f"iteration {iteration} of " if iteration else ""
            logger.error(
                f"{type(e).__name__} in {where}step {self.name}: {e}",
                exc_info=True,
            )
            raise

    def _collect_metadata(self):
        """Collect step metadata for embedding in saved images."""
        metadata = {"step_name": self.name}

        # The whole recipe, not just this step's slice of it - this is what
        # lets a gallery open an image as the workflow that produced it
        if self.workflow_definition is not None:
            metadata["workflow"] = self.workflow_definition

        if "pipeline" in self.step_definition:
            pipeline_def = self.step_definition["pipeline"]
            pretrained_args = pipeline_def.get("from_pretrained_arguments", {})
            if "model_name" in pretrained_args:
                metadata["model_name"] = pretrained_args["model_name"]
            metadata["arguments"] = dict(pipeline_def.get("arguments", {}))

        elif "task" in self.step_definition:
            task_def = self.step_definition["task"]
            metadata["task_command"] = task_def.get("command", "unknown")
            metadata["arguments"] = dict(task_def.get("arguments", {}))

        return metadata
