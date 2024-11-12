"""

"""

# Import all return models
from .sample_cov import sample_cov
from .semi_cov import semicovariance
from .exp_cov import exp_cov

from pypo.input_estimates.robust_models.cov_shrinkage_rm import CovarianceShrinkage


# Define what is exported when using 'from models import *'
__all__ = ['compute_risk_matrix','sample_cov','semi_cov','exp_cov']


# Master function that centralizes all expected return models.
def compute_risk_matrix(prices, method="sample_cov", **kwargs):
    """
    Compute a covariance matrix, using the risk model supplied in the ``method``
    parameter.

    :param prices: adjusted closing prices of the asset, each row is a date
                   and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param returns_data: if true, the first argument is returns instead of prices.
    :type returns_data: bool, defaults to False.
    :param method: the risk model to use. Should be one of:

        - ``sample_cov``
        - ``semicovariance``
        - ``exp_cov``
        - ``ledoit_wolf``
        - ``ledoit_wolf_constant_variance``
        - ``ledoit_wolf_single_factor``
        - ``ledoit_wolf_constant_correlation``
        - ``oracle_approximating``

    :type method: str, optional
    :raises NotImplementedError: if the supplied method is not recognised
    :return: annualised sample covariance matrix
    :rtype: pd.DataFrame
    """
    if method == "sample_cov":
        return sample_cov(prices, **kwargs)
    elif method == "semicovariance" or method == "semivariance":
        return semicovariance(prices, **kwargs)
    elif method == "exp_cov":
        return exp_cov(prices, **kwargs)
    elif method == "ledoit_wolf" or method == "ledoit_wolf_constant_variance":
        return CovarianceShrinkage(prices, **kwargs).ledoit_wolf()
    elif method == "ledoit_wolf_single_factor":
        return CovarianceShrinkage(prices, **kwargs).ledoit_wolf(
            shrinkage_target="single_factor"
        )
    elif method == "ledoit_wolf_constant_correlation":
        return CovarianceShrinkage(prices, **kwargs).ledoit_wolf(
            shrinkage_target="constant_correlation"
        )
    elif method == "oracle_approximating":
        return CovarianceShrinkage(prices, **kwargs).oracle_approximating()
    else:
        raise NotImplementedError("Risk model {} not implemented".format(method))

    