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

# Define what is exported when using 'from models import *'
__all__ = ['compute_expected_return','mh_rm','ewmh_rm','capm_rm']


# Master function that centralizes all expected return models.
def compute_expected_return(prices, method="mh_rm", **kwargs):
    """
    Compute an estimate of future returns, using the stock prices as input and the return model specified in ``method``.

    :param prices: adjusted closing prices of the asset, each row is a date
                   and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param returns_data: if true, the first argument is returns instead of prices.
    :type returns_data: bool, defaults to False.
    :param method: the return model to use. Should be one of:

        - ``mh_rm`` -> mean historical return model
        - ``ewmh_rm`` -> exponentially-weighted mean historical return model
        - ``capm_rm`` -> capm return model

    :type method: str, optional
    :raises NotImplementedError: if the supplied method is not recognised
    :return: annualised sample covariance matrix
    :rtype: pd.DataFrame
    """
    if method == "mh_rm":
        return mh_rm(prices, **kwargs)
    elif method == "ewmh_rm":
        return ewmh_rm(prices, **kwargs)
    elif method == "capm_rm":
        return capm_rm(prices, **kwargs)
    else:
        raise NotImplementedError("Return model {} not implemented".format(method))