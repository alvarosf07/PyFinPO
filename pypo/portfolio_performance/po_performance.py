"""
This submodule defines a general utility function ``portfolio_performance`` to
evaluate return and risk for a given set of portfolio weights.
"""

import numpy as np
import pandas as pd

from pypo.portfolio_optimization.po_objective_functions import objective_functions


def portfolio_performance(
    weights, expected_returns, cov_matrix, verbose=False, risk_free_rate=0.02
):
    """
    After optimising, calculate (and optionally print) the performance of the optimal
    portfolio. Currently calculates expected return, volatility, and the Sharpe ratio.

    :param expected_returns: expected returns for each asset. Can be None if
                             optimising for volatility only (but not recommended).
    :type expected_returns: np.ndarray or pd.Series
    :param cov_matrix: covariance of returns for each asset
    :type cov_matrix: np.array or pd.DataFrame
    :param weights: weights or assets
    :type weights: list, np.array or dict, optional
    :param verbose: whether performance should be printed, defaults to False
    :type verbose: bool, optional
    :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02
    :type risk_free_rate: float, optional
    :raises ValueError: if weights have not been calculated yet
    :return: expected return, volatility, Sharpe ratio.
    :rtype: (float, float, float)
    """
    if isinstance(weights, dict):
        if isinstance(expected_returns, pd.Series):
            tickers = list(expected_returns.index)
        elif isinstance(cov_matrix, pd.DataFrame):
            tickers = list(cov_matrix.columns)
        else:
            tickers = list(range(len(expected_returns)))
        new_weights = np.zeros(len(tickers))

        for i, k in enumerate(tickers):
            if k in weights:
                new_weights[i] = weights[k]
        if new_weights.sum() == 0:
            raise ValueError("Weights add to zero, or ticker names don't match")
    elif weights is not None:
        new_weights = np.asarray(weights)
    else:
        raise ValueError("Weights is None")

    sigma = np.sqrt(objective_functions.portfolio_variance(new_weights, cov_matrix))

    if expected_returns is not None:
        mu = objective_functions.portfolio_return(
            new_weights, expected_returns, negative=False
        )

        sharpe = objective_functions.sharpe_ratio(
            new_weights,
            expected_returns,
            cov_matrix,
            risk_free_rate=risk_free_rate,
            negative=False,
        )
        if verbose:
            print("Expected annual return: {:.1f}%".format(100 * mu))
            print("Annual volatility: {:.1f}%".format(100 * sigma))
            print("Sharpe Ratio: {:.2f}".format(sharpe))
        return mu, sigma, sharpe
    else:
        if verbose:
            print("Annual volatility: {:.1f}%".format(100 * sigma))
        return None, sigma, None