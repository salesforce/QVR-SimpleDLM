#!/usr/bin/env python3
import torch
from setuptools import find_packages, setup

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 4], "Requires PyTorch >= 1.4"

setup(
    name="QVR_SimpleDLM",
    version="1.0",
    description="Query value retrieval with SimpleDLM pretraining.",
    packages=find_packages(exclude=("configs", "tests")),
    python_requires=">=3.6",
    install_requires=[
        "transformers==2.9.0",
        "tensorboardX==2.0",
        "lxml==4.6.5",
        "seqeval==0.0.12",
        "Pillow==7.1.2",
    ],
    extras_require={
        "dev": ["flake8==3.8.2", "isort==4.3.21", "black==19.10b0", "pre-commit==2.4.0"]
    },
)
