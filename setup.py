import os
import pkg_resources
from setuptools import find_packages
from setuptools import setup
import sys


def get_version():
    version_filepath = os.path.join(os.path.dirname(__file__), 'optuna', 'version.py')
    with open(version_filepath) as f:
        for line in f:
            if line.startswith('__version__'):
                return line.strip().split()[-1][1:-1]
    assert False


def get_install_requires():
    install_requires = [
        'sqlalchemy>=1.1.0', 'numpy', 'scipy', 'six', 'typing', 'cliff', 'colorlog', 'pandas']
    if sys.version_info[0] == 2:
        install_requires.extend(['enum34'])
    return install_requires


def get_extras_require():
    extras_require = {
        'checking': ['pytest', 'hacking'],
        'testing': ['mock', 'bokeh', 'chainer>=5.0.0', 'xgboost'],
    }
    if sys.version_info[0] == 3:
        # TODO(Yanase): Setting mypy version to 0.620 as a temporal fix
        # for the library's problem in handling NamedTuple since 0.630.
        # The problem is tracked at https://github.com/python/mypy/issues/5640.
        extras_require['checking'].append('mypy==0.620')
    return extras_require


def find_any_distribution(pkgs):
    for pkg in pkgs:
        try:
            return pkg_resources.get_distribution(pkg)
        except pkg_resources.DistributionNotFound:
            pass
    return None


pfnopt_pkg = find_any_distribution(['pfnopt'])
if pfnopt_pkg is not None:
    msg = 'We detected that PFNOpt is installed in your environment.\n' \
        'PFNOpt has been renamed Optuna. Please uninstall the old\n' \
        'PFNOpt in advance (e.g. by executing `$ pip uninstall pfnopt`).'
    print(msg)
    exit(1)

setup(
    name='optuna',
    version=get_version(),
    description='',
    author='Takuya Akiba',
    author_email='akiba@preferred.jp',
    packages=find_packages(),
    install_requires=get_install_requires(),
    tests_require=get_extras_require()['testing'],
    extras_require=get_extras_require(),
    entry_points={
        'console_scripts': ['optuna = optuna.cli:main']
    }
)
