import os
from .workflow import workflow_from_file
from . import startup

def main():
    workflow = workflow_from_file(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "workflows", "test.json"
        ),
        "./outputs",
    )

    try:
        startup("DEBUG")
        workflow.validate()
    except Exception as e:
        print(f"Error validating workflow: {e}")
        exit(1)

    try:
        workflow.run({})
    except Exception as e:
        print(f"Error running workflow: {e}")
        exit(1)


if __name__ == "__main__":
    main()
