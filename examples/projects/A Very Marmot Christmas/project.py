import os
import json
import subprocess

# Open and read the 2024.json file
with open("./projects/A Very Marmot Christmas/2024.json", "r") as file:
    prompts = json.load(file)

# Iterate over all strings in the array and invoke python on the command line
for i, prompt in enumerate(prompts):
    subprocess.run(
        [
            "python",
            "-m",
            "dw.run",
            "-o",
            "./projects/A Very Marmot Christmas/outputs",
            "./projects/A Very Marmot Christmas/flux.json",
            f'prompt="{prompt}"',
            f"file_base_name=avm24-{i}",
            "num_images_per_prompt=5",
        ]
    )
