#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The setup script."""

# Import modules
from setuptools import find_packages, setup

with open("readme.md", encoding="utf8") as readme_file:
    readme = readme_file.read()

with open("requirements.in") as f:
    requirements = f.read().splitlines()

with open("requirements-dev.txt") as f:
    dev_requirements = f.read().splitlines()

setup(
    name="pyswarms_plugins",
    version="0.0.0",
    description=
    "A package containing algorithms not included in pyswarms package.",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Hung-Tien Huang",
    author_email="hungtienhuang@gmail.com",
    url="https://github.com/busFred/pyswarms_plugins",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=requirements,
    tests_require=dev_requirements,
    license="MIT license",
    zip_safe=False,
    keywords="pyswarms_plugins",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
