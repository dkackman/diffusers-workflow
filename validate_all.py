# FILE: validate_all.py
import os
import subprocess

examples_dir = "./examples"

for file_name in os.listdir(examples_dir):
    if file_name.endswith(".json"):
        file_path = os.path.join(examples_dir, file_name)
        subprocess.run(["python", "-m", "dh.validate", file_path])
