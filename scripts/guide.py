import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
#from pypfopt.efficient_frontier import EfficientFrontier
#from pypfopt import risk_models
#from pypfopt import expected_returns

sys.path.append('/Users/alvarosanchez/Documents/Projects/personal-projects/pyfinpo')
from pyfinpo.input_estimates import expected_return_models
from pyfinpo.input_estimates import risk_models
from pyfinpo.input_estimates.robust_models import black_litterman_rm, BlackLittermanModel
from pyfinpo.portfolio_optimization import po_models
from pyfinpo.utils import DiscreteAllocation, get_latest_prices

#df = pd.read_csv("data/raw/stock_prices_test.csv")


# 1) Get Data
tickers = ["ACN", "AMZN", "COST", "DIS", "F", "GILD", "JPM", "KO", "LUV", "MA", "MSFT", "PFE", "TSLA", "UNH", "XOM"]
ohlc = yf.download(tickers, period="max")
prices = ohlc["Adj Close"].dropna(how="all")


# 2) Inputs Estimation
# 2.1) Return Estimation
mu1 = expected_return_models.mh_rm(prices)
mu2 = expected_return_models.ewmh_rm(prices)
mu3 = expected_return_models.capm_rm(prices)
print (mu3)
# 2.2) Risk Estimation
s1 = risk_models.sample_cov(prices)
s2 = risk_models.semi_cov(prices)
s3 = risk_models.ew_cov(prices)
s4 = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
# 2.3) Robust Models
viewdict = {"AAPL": 0.20, "BBY": -0.30, "BAC": 0, "SBUX": -0.2, "T": 0.15}
bl = BlackLittermanModel(s4, absolute_views=viewdict)



# Example 1 - Mean-variance Portfolio Optimization with the objective of finding the minimum variance portfolio
# 3) Portfolio Optimization
po_11 = po_models.MeanVariancePO(expected_returns=mu1, cov_matrix=s1, weight_bounds=(0,1), verbose=True)
optimized_portfolio_weights_11 = po_11.min_volatility()
optimized_portfolio_clean_weights_11 = po_11.clean_weights() #TODO: integrate clean weights option as argument in the previous function, don't wanna call again the class MeanVariancePO

# 4) Portfolio Performance
po_11.portfolio_performance(verbose=True)

# 5) Transform portfolio weights into amount of shares of each security that must be bought (discrete allocation)
da_11 = DiscreteAllocation(weights=optimized_portfolio_weights_11, latest_prices=get_latest_prices(prices), total_portfolio_value=10000)



# Example 2 - Different objective functions within Mean-Variance Optimization
opw_12 = po_models.MeanVariancePO(expected_returns=mu1, cov_matrix=s1, weight_bounds=(0,1), verbose=True).efficient_risk()
opcw_12 = po_models.MeanVariancePO.clean_weights 
opw_13 = po_models.MeanVariancePO(expected_returns=mu1, cov_matrix=s1, weight_bounds=(0,1), verbose=True).efficient_return()
opw_14 = po_models.MeanVariancePO(expected_returns=mu1, cov_matrix=s1, weight_bounds=(0,1), verbose=True).max_sharpe()
opw_15 = po_models.MeanVariancePO(expected_returns=mu1, cov_matrix=s1, weight_bounds=(0,1), verbose=True).max_quadratic_utility()




# Example 3 - Black Litterman, Mean-CVaR Portfolio Optimization
rets = bl.bl_returns()
ef = po_models.MeanCVaRPO(rets, s4)

# OR use return-implied weights
delta = black_litterman_rm.market_implied_risk_aversion(prices)
bl.bl_weights(delta)
weights = bl.clean_weights()  # Are these really the Mean-CVaR optimized weights? or MV optimized weights?