"""

"""

# Import all objective functions
from .objective_functions import portfolio_variance
from .objective_functions import portfolio_return
from .objective_functions import sharpe_ratio
from .objective_functions import L2_reg
from .objective_functions import quadratic_utility
from .objective_functions import transaction_cost
from .objective_functions import ex_ante_tracking_error
from .objective_functions import ex_post_tracking_error

# Define what is exported when using 'from models import *'
__all__ = [
    'objective_function',
    'portfolio_variance',
    'portfolio_return',
    'sharpe_ratio',
    'L2_reg',
    'quadratic_utility',
    'transaction_cost',
    'ex_ante_tracking_error',
    'ex_post_tracking_error'
]


# TODO: add objective_function that centralizes all objective_functions in a single function