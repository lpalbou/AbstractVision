"""Setup configuration for abstractvision."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="abstractvision",
    version="0.1.0",
    author="Laurent-Philippe Albou",
    author_email="contact@abstractcore.ai",
    description="Generative vision capabilities for abstractcore.ai",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abstractcore/abstractvision",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies will be added as the project develops
    ],
)
