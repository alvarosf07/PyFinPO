"""
The ``expected_return_models`` module provides functions for estimating the expected returns of
the assets, which is a required input in mean-variance optimization.

By convention, the output of these methods is expected *annual* returns. It is assumed that
*daily* prices are provided, though in reality the functions are agnostic
to the time period (just change the ``frequency`` parameter). Asset prices must be given as
a pandas dataframe, as per the format described in the :ref:`user-guide`.

All of the functions process the price data into percentage returns data, before
calculating their respective estimates of expected returns.

Currently implemented:

    - general return model function, allowing you to run any return model from one function.
    - mean historical return
    - exponentially weighted mean historical return
    - CAPM estimate of returns

Additionally, we provide utility functions to convert from returns to prices and vice-versa (located under pypo.utils.returns.py)
"""

# Import all return models
from .mh_rm import mh_rm
from .ewmh_rm import ewmh_rm
from .capm_rm import capm_rm
from .compute_expected_return import compute_expected_return

# Define what is exported when using 'from models import *'
__all__ = ['compute_expected_return','mh_rm','ewmh_rm','capm_rm']