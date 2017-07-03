#! /usr/bin/env python

import os

from distutils.core import setup
from setuptools.command.test import test as TestCommand

NAME = 'DeDist'


class PyTest(TestCommand):

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        pytest.main(self.test_args)


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name=NAME,
    version="0.1",
    author="Sander Keemink",
    author_email="swkeemink@scimail.eu",
    description="A Python library for estimating decoding distributions",
    url="https://github.com/swkeemink/DeDist",
    download_url="NA",
    package_dir={NAME: "./dedist"},
    packages=[NAME],
    license="GNU",
    long_description=read('README.md'),
    classifiers=[
        "License :: GNU",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering"
    ],
    cmdclass={'test': PyTest},
)
