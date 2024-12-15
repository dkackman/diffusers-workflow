import argparse
import os
from .project import create_project, load_json_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a project from a file.")
    parser.add_argument(
        "file_name", type=str, help="The filespec to of the project to validate"
    )
    args = parser.parse_args()

    if not os.path.exists(args.file_name):
        raise FileNotFoundError(f"File {args.file_name} does not exist")

    try:
        project = create_project(load_json_file(args.file_name))
        project.validate()
    except Exception as e:
        print(f"Error validating project: {e}")
        exit(1)

    print("Project validated successfully")
