from pypo.input_estimates.expected_return_models import mh_rm
from pypo.input_estimates.expected_return_models import ewmh_rm
from pypo.input_estimates.expected_return_models import capm_rm


def compute_expected_return(prices, method="mh_rm", **kwargs):
    """
    Compute an estimate of future returns, using the return model specified in ``method``.

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
