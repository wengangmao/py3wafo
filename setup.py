#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup file for wafo.

Usage:
Install the package and test dependencies:
  python -m pip install -e ".[test]"

Run all tests:
  python setup.py test
  python setup.py doctest

Build documentation
  python setup.py docs
  python setup.py latex

Install
  python setup.py install [, --prefix=$PREFIX]

Build
  python setup.py bdist_wininst
  python setup.py bdist_wheel
  python setup.py sdist

Recommended build
  git pull origin
  git shortlog v0.9.20..HEAD -w80 --format="* %s" --reverse > log.txt
# update CHANGELOG.rst with info from log.txt

  python build_package.py 0.10.0rc0
  git commit
  git tag v0.10.0rc0 master
  git push --tags

PyPi upload:
  twine check dist/*   # check
  twine upload dist/*  # wait until the travis report is OK before doing this step.

Notes
-----
Don't use package_data and/or data_files, use include_package_data=True and MANIFEST.in instead!
Don't hard-code the list of packages, use setuptools.find_packages() instead!


See also
--------
https://docs.pytest.org/en/latest/goodpractices.html
https://python-packaging.readthedocs.io/en/latest/
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
https://chriswarrick.com/blog/2014/09/15/python-apps-the-right-way-entry_points-and-scripts/
https://blog.ionelmc.ro/2014/05/25/python-packaging/#the-structure
https://ep2015.europython.eu/media/conference/slides/less-known-packaging-features-and-tricks.pdf
https://realpython.com/documenting-python-code/#public-and-open-source-projects

"""
import os
import re
import shlex
import sys
from setuptools import Command
from setuptools.command.sdist import sdist

BUILD_NATIVE = os.environ.get('WAFO_BUILD_EXTENSIONS', '').lower() in {
    '1', 'true', 'yes', 'on'}

if BUILD_NATIVE:
    try:
        from numpy.distutils.core import setup
        from numpy.distutils.misc_util import Configuration
    except ImportError as error:
        raise RuntimeError(
            'Native WAFO extensions require NumPy < 2 and a compatible '
            'C/Fortran compiler. Unset WAFO_BUILD_EXTENSIONS for the '
            'pure-Python installation.') from error
else:
    from setuptools import setup

    class Configuration:
        """No-op configuration used for the default pure-Python build."""

        def __init__(self, _package_name):
            pass

        def add_extension(self, *_args, **_kwargs):
            pass

        def add_library(self, *_args, **_kwargs):
            pass

        def add_data_dir(self, *_args, **_kwargs):
            pass

        def todict(self):
            return {}

ROOT = os.path.abspath(os.path.dirname(__file__))
PACKAGE_NAME = 'wafo'
DIST_NAME = 'py3wafo'


def read(file_path, lines=False):
    """Returns contents of file either as a string or list of lines."""
    with open(file_path, 'r') as fp:
        if lines:
            return fp.readlines()
        return fp.read()


def find_version(file_path):
    """Returns version given in the __version__ variable of a module file"""
    version_file = read(file_path)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)  # @UndefinedVariable
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


class Doctest(Command):
    description = 'Run doctests with Sphinx'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        from sphinx.application import Sphinx
        sph = Sphinx('./docs',  # source directory
                     './docs',  # directory containing conf.py
                     './docs/_build',  # output directory
                     './docs/_build/doctrees',  # doctree directory
                     'doctest')  # finally, specify the doctest builder
        sph.build()


class Latex(Command):
    description = 'Run latex with Sphinx'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        from sphinx.application import Sphinx
        sph = Sphinx('./docs',  # source directory
                     './docs',  # directory containing conf.py
                     './docs/_build/latex',  # output directory
                     './docs/_build/doctrees',  # doctree directory
                     'latex')  # finally, specify the latex builder
        sph.build()


class PyTest(Command):
    """Run the WAFO test suite without the obsolete pytest-runner package."""

    description = 'Run tests with pytest'
    user_options = [
        ('addopts=', 'a', 'arguments passed to pytest'),
    ]

    def initialize_options(self):
        self.addopts = None

    def finalize_options(self):
        self.pytest_args = (shlex.split(self.addopts)
                            if self.addopts else ['src/wafo'])

    def run(self):
        try:
            import pytest
        except ImportError as error:
            raise RuntimeError(
                'pytest is required to run tests. Install the development '
                'requirements with "python -m pip install -r '
                'requirements-dev.txt".') from error

        sys.path.insert(0, os.path.join(ROOT, 'src'))
        errno = pytest.main(self.pytest_args)
        if errno:
            raise SystemExit(errno)


def setup_package():
    extra_link_args = ["-static", "-static-libgfortran", "-static-libgcc"] if os.name == 'nt' else []
    config = Configuration(PACKAGE_NAME)

    # -------------------------------------------------------------------------
    # c_library
    config.add_extension('c_library',
                         sources=['source/c_library/c_library.pyf',
                                  'source/c_library/c_functions.c'])
    # -------------------------------------------------------------------------
    # mvn
    config.add_extension('mvn',
                         sources=['source/mvn/mvn.pyf',
                                  'source/mvn/mvndst.f'],
                         extra_link_args=extra_link_args)

    # -------------------------------------------------------------------------
    # mvnprdmod
    lib_mvnprdmod_src = ['source/mvnprd/mvnprd.f',
                         'source/mvnprd/mvnprodcorrprb.f']
    config.add_library('_mvnprdmod', sources=lib_mvnprdmod_src)
    config.add_extension('mvnprdmod',
                         sources=['source/mvnprd/mvnprd_interface.f'],
                         libraries=['_mvnprdmod'],
                         depends=(lib_mvnprdmod_src),
                         extra_link_args=extra_link_args)

    # -------------------------------------------------------------------------
    # cov2mod
    lib_cov2mod_src = ['source/mreg/dsvdc.f',
                       'source/mreg/mregmodule.f',
                       'source/mreg/intfcmod.f']
    config.add_library('_cov2mod', sources=lib_cov2mod_src)
    config.add_extension('cov2mod',
                         sources=['source/mreg/cov2mmpdfreg_intfc.f'],
                         libraries=['_cov2mod'],
                         include_dirs=['source/mreg/'],
                         depends=(lib_cov2mod_src),
                         extra_link_args=extra_link_args)

    # -------------------------------------------------------------------------
    # rindmod
    lib_rindmod_src = ['source/rind2007/intmodule.f',
                       'source/rind2007/jacobmod.f',
                       'source/rind2007/swapmod.f',
                       'source/rind2007/fimod.f',
                       'source/rind2007/rindmod.f',
                       'source/rind2007/rind71mod.f']
    config.add_library('_rindmod', sources=lib_rindmod_src)
    config.add_extension('rindmod',
                         sources=['source/rind2007/rind_interface.f'],
                         libraries=['_rindmod'],
                         include_dirs=['source/mreg/'],
                         depends=(lib_rindmod_src),
                         extra_link_args=extra_link_args)

    config.add_data_dir(('data', 'src/wafo/data'))

    version = find_version(os.path.join(ROOT, 'src', PACKAGE_NAME, "__init__.py"))
    print(("Version: {}".format(version)))

    sphinx_requires = ['sphinx>=7', 'numpydoc>=1.6',
                       'sphinx-rtd-theme>=2']
    test_requires = read(os.path.join(ROOT, 'requirements-dev.txt'),
                         lines=True)
    setup(
        name=DIST_NAME,
        version=version,
        python_requires='>=3.9',
        install_requires=read(os.path.join(ROOT, 'requirements.txt'), lines=True),
        extras_require={'build_sphinx': sphinx_requires,
                        'test': test_requires},
        package_data={  # If any package contains *.txt or *.rst files, include them:
                      "": ["*.dat"],},
        cmdclass={'test': PyTest,
                  'pytest': PyTest,
                  'doctest': Doctest,
                  'latex': Latex,
                  'sdist': sdist},
        **config.todict()
    )


if __name__ == "__main__":
    setup_package()
