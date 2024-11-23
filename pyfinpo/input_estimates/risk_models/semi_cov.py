import warnings

import numpy as np
import pandas as pd

from pyfinpo.utils.return_utils import returns_from_prices
from pyfinpo.utils.risk_utils import fix_nonpositive_semidefinite


def semi_cov(
    prices,
    returns_data=False,
    benchmark=0.000079,
    frequency=252,
    log_returns=False,
    **kwargs
):
    """
    Estimate the semicovariance matrix, i.e the covariance given that
    the returns are less than the benchmark.

    .. semicov = E([min(r_i - B, 0)] . [min(r_j - B, 0)])

    :param prices: adjusted closing prices of the asset, each row is a date
                   and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param returns_data: if true, the first argument is returns instead of prices.
    :type returns_data: bool, defaults to False.
    :param benchmark: the benchmark return, defaults to the daily risk-free rate, i.e
                      :math:`1.02^{(1/252)} -1`.
    :type benchmark: float
    :param frequency: number of time periods in a year, defaults to 252 (the number
                      of trading days in a year). Ensure that you use the appropriate
                      benchmark, e.g if ``frequency=12`` use the monthly risk-free rate.
    :type frequency: int, optional
    :param log_returns: whether to compute using log returns
    :type log_returns: bool, defaults to False
    :return: semicovariance matrix
    :rtype: pd.DataFrame
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("data is not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices, log_returns)
    drops = np.fmin(returns - benchmark, 0)
    T = drops.shape[0]
    return fix_nonpositive_semidefinite(
        (drops.T @ drops) / T * frequency, kwargs.get("fix_method", "spectral")
    )