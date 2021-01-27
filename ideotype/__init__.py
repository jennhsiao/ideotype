"""
Setup namespace for project modules.

In addition, fetch current project version (git-hash).

"""
from setuptools_scm import get_version
from pathlib import Path
from pkg_resources import get_distribution, DistributionNotFound

from .read_sims import *  # noqa
from .utils import *  # noqa
from .log import * # noqa
from .sql_declarative import *  # noqa
from .sql_insert import *  # noqa

try:
    # get accurate version for developer installs
    version_str = get_version(Path(__file__).parent.parent)
    __version__ = version_str
except (LookupError, ImportError):
    try:
        # Set the version automatically from the package details.
        __version__ = get_distribution(__name__).version
    except DistributionNotFound:
        # package is not installed
        pass
