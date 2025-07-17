"""
Configuration setup pour le package Reinforcement Learning.

Ce fichier permet d'installer le package avec pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Lire le README pour la description longue
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Lire les requirements
requirements = []
with open('requirements.txt') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('-'):
            # Ignorer les contraintes spécifiques à pip
            if 'python_requires' not in line:
                requirements.append(line)

setup(
    name="reinforcement-learning-algorithms",
    version="1.0.0",
    author="Votre nom",
    author_email="votre.email@example.com",
    description="Implémentation des algorithmes classiques de Reinforcement Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/votre-username/reinforcement-learning-algorithms",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
        "pandas>=1.3.0",
        "gym>=0.21.0",
        "joblib>=1.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "flake8>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "jupyterlab>=3.0.0",
            "ipywidgets>=7.6.0",
            "seaborn>=0.11.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "full": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "flake8>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "jupyterlab>=3.0.0",
            "ipywidgets>=7.6.0",
            "seaborn>=0.11.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Ajouter des scripts de ligne de commande si nécessaire
            # "rl-train=src.cli:main",
        ],
    },
    package_data={
        "src": ["*.txt", "*.md"],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="reinforcement learning, machine learning, AI, algorithms, RL",
    project_urls={
        "Bug Reports": "https://github.com/votre-username/reinforcement-learning-algorithms/issues",
        "Source": "https://github.com/votre-username/reinforcement-learning-algorithms",
        "Documentation": "https://github.com/votre-username/reinforcement-learning-algorithms/wiki",
    },
) 