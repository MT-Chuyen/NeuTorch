"""Setuptools entrypoint (metadata primarily lives in pyproject.toml)."""

from setuptools import find_packages, setup

setup(
    name="neutorch",
    version="0.1.1",
    packages=find_packages(include=["neutorch", "neutorch.*"]),
    include_package_data=True,
)
