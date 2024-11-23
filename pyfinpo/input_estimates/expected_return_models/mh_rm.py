import warnings

import pandas as pd

from pyfinpo.utils.return_utils import _check_returns, returns_from_prices, prices_from_returns


def mh_rm(
    prices, returns_data=False, compounding=True, frequency=252, log_returns=False
):
    """
    Calculate annualised mean (daily) historical return from input (daily) asset prices.
    Use ``compounding`` to toggle between the default geometric mean (CAGR) and the
    arithmetic mean.

    :param prices: adjusted closing prices of the asset, each row is a date
                   and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param returns_data: if true, the first argument is returns instead of prices.
                         These **should not** be log returns.
    :type returns_data: bool, defaults to False.
    :param compounding: computes geometric mean returns if True,
                        arithmetic otherwise, optional.
    :type compounding: bool, defaults to True
    :param frequency: number of time periods in a year, defaults to 252 (the number
                      of trading days in a year)
    :type frequency: int, optional
    :param log_returns: whether to compute using log returns
    :type log_returns: bool, defaults to False
    :return: annualised mean (daily) return for each asset
    :rtype: pd.Series
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices are not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices, log_returns)

    _check_returns(returns)
    if compounding:
        return (1 + returns).prod() ** (frequency / returns.count()) - 1
    else:
        return returns.mean() * frequency