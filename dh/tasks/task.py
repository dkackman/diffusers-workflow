from .task_dispatch import dispatch_task

class Task:
    def __init__(self, task_definition):
        self.task_definition = task_definition

    def run(self, device_identifier):
        task_output = dispatch_task(self.task_definition["command"], self.task_definition.get("arguments", {}))
        return task_output
    
    @property
    def command(self):
        return self.task_definition.get("command", "unknown")