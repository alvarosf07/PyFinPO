import warnings

import numpy as np
import pandas as pd

from pyfinpo.utils.return_utils import returns_from_prices
from pyfinpo.utils.risk_utils import fix_nonpositive_semidefinite


def sample_cov(prices, returns_data=False, frequency=252, log_returns=False, **kwargs):
    """
    Calculate the annualised sample covariance matrix of (daily) asset returns.

    :param prices: adjusted closing prices of the asset, each row is a date
                   and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param returns_data: if true, the first argument is returns instead of prices.
    :type returns_data: bool, defaults to False.
    :param frequency: number of time periods in a year, defaults to 252 (the number
                      of trading days in a year)
    :type frequency: int, optional
    :param log_returns: whether to compute using log returns
    :type log_returns: bool, defaults to False
    :return: annualised sample covariance matrix
    :rtype: pd.DataFrame
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("data is not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices, log_returns)
    return fix_nonpositive_semidefinite(
        returns.cov() * frequency, kwargs.get("fix_method", "spectral")
    )
