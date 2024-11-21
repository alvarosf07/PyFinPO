"""
The ``robust_models`` module provides additional models that improve 
the robustness and predictive performance of return and risk models.

**Currently implemented:**

- Black Litterman Model (BlackLittermanModel)

"""

# Import all return models
from .black_litterman_rm import (
    BlackLittermanModel,
    market_implied_prior_returns,
    market_implied_risk_aversion,
)


# Define what is exported when using 'from models import *'
__all__ = [
    'BlackLittermanModel',
    'market_implied_prior_returns',
    'market_implied_risk_aversion',
]

