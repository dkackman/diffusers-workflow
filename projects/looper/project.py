import os
import json
import subprocess


subprocess.run(
    [
        "python",
        "-m",
        "dh.run",
        "-o",
        "./projects/looper/outputs",
        "./projects/looper/img2txt2img.json",
        f"file_base_name=looper-{0}",
        "num_inference_steps=25",
        "image_glob=./projects/looper/start_img.jpg",
    ]
)

for i in range(1, 100):
    subprocess.run(
        [
            "python",
            "-m",
            "dh.run",
            "-o",
            "./projects/looper/outputs",
            "./projects/looper/img2txt2img.json",
            f"file_base_name=looper-{i}",
            "num_inference_steps=25",
            f"image_glob=./projects/looper/outputs/looper-{i-1}-*.jpg",
        ]
    )
