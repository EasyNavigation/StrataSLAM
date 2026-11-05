"""Shim for colcon (ament_python). The metadata and the dependencies come from pyproject.toml; the
package list and the `strataslam` entry point are repeated here because colcon runs setup.py in ways
(--symlink-install, setup.py develop) where pyproject.toml is not read."""
from setuptools import setup

setup(
    packages=["strataslam_tools"],
    entry_points={"console_scripts": ["strataslam = strataslam_tools.cli:main"]},
    # colcon picks pytest (instead of unittest, which finds nothing) from this declaration.
    extras_require={"test": ["pytest"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/strataslam_tools"]),
        ("share/strataslam_tools", ["package.xml"]),
    ],
)
