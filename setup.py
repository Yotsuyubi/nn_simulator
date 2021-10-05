from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))


def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(
    name="nn_simulator",
    version="1.0.0",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    python_requires=">=3.5",
    install_requires=_requires_from_file('requirements.txt'),
    dependency_links=[],
    include_package_data=True,
)
