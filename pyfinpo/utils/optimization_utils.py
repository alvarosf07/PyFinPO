"""
The module pypo.utils.optimization_utils provides utility functions to support the module 'po_optimizers'.
"""

from collections.abc import Iterable
from typing import List

import cvxpy as cp


def _flatten(alist: Iterable) -> Iterable:
    # Helper method to flatten an iterable
    for v in alist:
        if isinstance(v, Iterable) and not isinstance(v, (str, bytes)):
            yield from _flatten(v)
        else:
            yield v


def _get_all_args(expression: cp.Expression) -> List[cp.Expression]:
    """
    Helper function to recursively get all arguments from a cvxpy expression

    :param expression: input cvxpy expression
    :type expression: cp.Expression
    :return: a list of cvxpy arguments
    :rtype: List[cp.Expression]
    """
    if expression.args == []:
        return [expression]
    else:
        return list(_flatten([_get_all_args(arg) for arg in expression.args]))