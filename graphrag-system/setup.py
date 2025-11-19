"""Setup script for GraphRAG system."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="graphrag-system",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Knowledge Graph based Retrieval Augmented Generation system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/graphrag-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "graphrag-build=scripts.1_build_graph:main",
            "graphrag-communities=scripts.2_detect_communities:main",
            "graphrag-reports=scripts.3_generate_reports:main",
            "graphrag-embed=scripts.4_create_embeddings:main",
            "graphrag-local=scripts.run_local_search:main",
            "graphrag-global=scripts.run_global_search:main",
        ],
    },
)
