#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

setup(
    author="ansatz",
    description="video-process tookits",
    name='videoprocess',
    packages=find_packages(include=['videoprocess', 'videoprocess.*', 'videoprocess.*.*']),
    # package_data={'': ['data/param/*.csv']},
    include_package_data=True,
    url='https://github.com/ansatzX/video-process',
    version='0.0.1',
)