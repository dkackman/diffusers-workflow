from .task_dispatch import dispatch_task
from ..result import Result

class Task:
    def __init__(self, task_definition):
        self.task_definition = task_definition

    def run(self):
        task_output = dispatch_task(self.task_definition["task_name"], self.task_definition.get("arguments", {}))
        result = Result(task_output, self.task_definition.get("result", {}))
        return result