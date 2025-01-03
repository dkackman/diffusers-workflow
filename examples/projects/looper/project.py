import os
import json
import subprocess


# subprocess.run(
#     [
#         "python",
#         "-m",
#         "dw.run",
#         "-o",
#         "./projects/looper/outputs",
#         "./projects/looper/img2txt2img.json",
#         f"file_base_name=looper-{0}",
#         "num_inference_steps=25",
#         "image_glob=./projects/looper/start_img.jpg",
#     ]
# )

for i in range(29, 100):
    subprocess.run(
        [
            "python",
            "-m",
            "dw.run",
            "-o",
            "./examples/projects/looper/outputs",
            "./examples/projects/looper/img2txt2img.json",
            f"file_base_name={i}",
            "num_inference_steps=25",
            f"image_glob=./examples/projects/looper/outputs/{i-1}looper-*.jpg",
        ]
    )
