from setuptools import setup, find_packages

setup(
    name="dh",
    version="0.37.0",  # Matches version in dh/__init__.py
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "diffusers",
        "transformers",
        "qrcode",
        "Pillow",
        "jsonschema",
        "concurrent-log-handler",
        "controlnet-aux",
        "soundfile",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
        ],
    },
    python_requires=">=3.10",
    author="dkackman",
    description="A helper library for diffusers workflows",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dkackman/diffusers-helper",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
