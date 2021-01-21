from setuptools_scm import get_version
from pathlib import Path
from pkg_resources import get_distribution, DistributionNotFound

from .read_sims import *
from .utils import *
from .sql_declarative import *
#from .sql_insert import *
#from . import sql_declarative
#from . import sql_insert

# from .sql_insert import * vs. from . import sql_insert
# are essentially the same thing but results in different namespace 
# option 1 (from .sql_insert import *):
#   you can access functions in modules by
#   directly calling the function name
#   e.g. from ideotype import CC_VPD
#   instead of from from ideotype.utils import CC_VPD

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
