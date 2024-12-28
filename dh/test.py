import os
from .workflow import workflow_from_file
from . import startup

if __name__ == "__main__":
    workflow = workflow_from_file(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "workflows", "test.json"
        ),
        ".",
    )

    try:
        workflow.validate()
    except Exception as e:
        print(f"Error validating workflow: {e}")
        exit(1)

    try:
        startup()
        workflow.run()
    except Exception as e:
        print(f"Error running workflow: {e}")
        exit(1)
