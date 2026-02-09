from setuptools import setup, find_packages

setup(
    name="ren_counterpoint",
    version="0.1.0",
    packages=find_packages(),
    description="Transformer model and dataset tools for Renaissance counterpoint generation",
    python_requires=">=3.8",
    install_requires=[
        "miditok>=3.0.6",
        "mido>=1.3.3",
        "symusic>=0.5.9",
        "z3-solver==4.15.4.0",
        "tqdm>=4.67.1",
        "numpy>=2.0.2",
        "datasets",  # HuggingFace datasets
        "torch",     # Assumes user will install appropriate CUDA version separately
    ],
)
