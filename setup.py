"""
Setup script for the separatrix_locator package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "A tool for locating separatrices in black-box dynamical systems using Koopman eigenfunctions."

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return [
        'torch>=1.9.0',
        'numpy>=1.21.0',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'scikit-learn>=1.0.0',
        'torchdiffeq>=0.2.0',
        'omegaconf>=2.1.0',
        'hydra-core>=1.1.0',
    ]

setup(
    name="separatrix_locator",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for locating separatrices in black-box dynamical systems using Koopman eigenfunctions",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/separatrix_locator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "ipykernel>=6.0",
            "nbformat>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "separatrix-locator=separatrix_locator.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "separatrix_locator": [
            "examples/*.ipynb",
            "examples/*.py",
        ],
    },
)
