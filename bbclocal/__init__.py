"""bbclocal: a fast local re-implementation of SNANA's BBC fit (SALT2mu) for one common configuration.

SNANA (SALT2mu.exe) remains the reference implementation; see README.md for what is supported and how it was validated.
"""
__version__ = "0.1.0"

from .config import BBCConfig, UnsupportedOptionError, parse_salt2mu_text, read_salt2mu_input  # noqa: E402,F401
from .core import BBC, BiasCorSims  # noqa: E402,F401
from .io import read_cid_list, read_fitres, read_survey_def  # noqa: E402,F401
