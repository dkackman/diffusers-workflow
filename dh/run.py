import argparse
import os
from .project import create_project, load_json_file
from . import startup

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a workflow from a file.")
    parser.add_argument(
        "file_name", type=str, help="The filespec to of the workflow to run"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        nargs="?",
        default="./outputs",
        help="The folder to write the outputs to",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        raise FileNotFoundError(f"Output directory {args.output_dir} does not exist")

    if not os.path.exists(args.file_name):
        raise FileNotFoundError(f"File {args.file_name} does not exist")

    project = create_project(load_json_file(args.file_name))
    try:    
        project.validate()
    except Exception as e:
        print(f"Error validating project: {e}")
        exit(1)

    try:    
        startup()
        project.run(args.output_dir)
    except Exception as e:
        print(f"Error running project: {e}")
        exit(1)

