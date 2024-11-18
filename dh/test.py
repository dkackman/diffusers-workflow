import os
from .project import create_project, load_json_file
from . import startup

if __name__ == "__main__":
    job = load_json_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_job.json"))
    project = create_project(job)

    try:    
        project.validate()
    except Exception as e:
        print(f"Error validating project: {e}")
        exit(1)

    try:            
        startup()
        project.run("*", ".")
    except Exception as e:
        print(f"Error running project: {e}")
        exit(1)
